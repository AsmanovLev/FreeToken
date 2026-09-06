// EXL3 (QTIP trellis) fused dequant kernel: packed expert rows -> bf16 matrices.
//
// One launch dequantizes M same-shape matrices (grid.y = matrix index), each
// processed in self-contained 128x128 blocks: decode 64 tiles (window extract
// + procedural codebook, tensor-core-fragment placement) into shared memory,
// then un-rotate in place: block-diagonal Sylvester H128 fast-Hadamard on the
// left (input dim) and right (output dim), then signed scales suh/svh, written
// transposed to the HF-orientation bf16 output [out, in].
//
// Math is bit-compatible with freetoken.models.exl3.dequant_exl3 (torch
// reference), which is bit-exact vs the exllamav3 CUDA oracle.

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace {

// 16-bit tail-biting window of element e (window low K bits = code[e]).
// `pay` = tile payload: 16*K uint16 words read as a LE uint32 stream.
// Matches models/exl3.py::_unpack_windows (modulo applies to indexing only,
// the shift uses the unwrapped bit positions).
template <int K>
__device__ __forceinline__ uint32_t exl3_window(const uint16_t *__restrict__ pay,
                                                int e) {
  const int b0 = e * K + K - 16 + 256 * K;
  const int b1 = b0 + 16;
  const int i0u = b0 >> 5, i1u = (b1 - 1) >> 5;
  const int s0 = ((i1u + 1) << 5) - b1;
  constexpr int n32 = 8 * K;
  const int i0 = i0u % n32, i1 = i1u % n32;
  const uint32_t a = static_cast<uint32_t>(pay[2 * i0]) |
                     (static_cast<uint32_t>(pay[2 * i0 + 1]) << 16);
  const uint32_t b = static_cast<uint32_t>(pay[2 * i1]) |
                     (static_cast<uint32_t>(pay[2 * i1 + 1]) << 16);
  return static_cast<uint32_t>(
             ((static_cast<uint64_t>(a) << 32) | b) >> s0) &
         0xFFFFu;
}

// Procedural QTIP codebooks (bit-exact with the torch reference).
__device__ __forceinline__ float exl3_decode_cb(uint32_t win, int cb) {
  if (cb == 2) {  // mul1
    const uint32_t x = win * 0x83DCD12Du;
    const uint32_t s = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) +
                       ((x >> 24) & 0xFFu) + 0x6400u;
    const __half h = __ushort_as_half(static_cast<uint16_t>(s & 0xFFFFu));
    const __half r = __hfma(h, __ushort_as_half(0x1EEE),
                            __ushort_as_half(0xC931));
    return __half2float(r);
  }
  uint32_t x;
  if (cb == 1) x = win * 0xCBAC1FEDu;        // mcg
  else x = win * 89226354u + 64248484u;      // legacy 3inst
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const __half lo = __ushort_as_half(static_cast<uint16_t>(x & 0xFFFFu));
  const __half hi = __ushort_as_half(static_cast<uint16_t>(x >> 16));
  return __half2float(__hadd(lo, hi));  // fp16 add, single rounding
}

// 128-point fast Walsh-Hadamard (Sylvester order) over 4 values per lane;
// element i lives on lane (i % 32) as register x_{i/32}. Unnormalized.
__device__ __forceinline__ void fwht128(float &x0, float &x1, float &x2,
                                        float &x3, int lane) {
#pragma unroll
  for (int d = 1; d <= 16; d <<= 1) {
    float t;
    t = __shfl_xor_sync(0xffffffffu, x0, d);
    x0 = (lane & d) ? t - x0 : x0 + t;
    t = __shfl_xor_sync(0xffffffffu, x1, d);
    x1 = (lane & d) ? t - x1 : x1 + t;
    t = __shfl_xor_sync(0xffffffffu, x2, d);
    x2 = (lane & d) ? t - x2 : x2 + t;
    t = __shfl_xor_sync(0xffffffffu, x3, d);
    x3 = (lane & d) ? t - x3 : x3 + t;
  }
  float a, b;  // stage 32 (register pairs), stage 64
  a = x0; b = x1; x0 = a + b; x1 = a - b;
  a = x2; b = x3; x2 = a + b; x3 = a - b;
  a = x0; b = x2; x0 = a + b; x2 = a - b;
  a = x1; b = x3; x1 = a + b; x3 = a - b;
}

struct Exl3DequantParams {
  const uint8_t *__restrict__ src;
  __nv_bfloat16 *__restrict__ out;
  std::size_t src_stride;      // bytes between matrices in src
  std::size_t out_mat_stride;  // elements between matrices in out
  std::size_t out_row_stride;  // elements between out rows (= in_dim)
  int64_t trellis_off;         // byte offsets within a row
  int64_t suh_off;
  int64_t svh_off;
  int in_dim;
  int out_dim;
  int codebook;
};

template <int K, int kNumThreads>
__global__ __launch_bounds__(kNumThreads) void //
    exl3_dequant_kernel(const __grid_constant__ Exl3DequantParams p) {
  constexpr int TK = 16 * K;  // uint16 words per tile payload
  extern __shared__ float smem_pool[];
  auto sW = reinterpret_cast<float (*)[129]>(smem_pool);

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;

  const uint8_t *row = p.src + static_cast<std::size_t>(blockIdx.y) * p.src_stride;
  const uint16_t *trellis =
      reinterpret_cast<const uint16_t *>(row + p.trellis_off);
  const __half *suh = reinterpret_cast<const __half *>(row + p.suh_off);
  const __half *svh = reinterpret_cast<const __half *>(row + p.svh_off);

  const int ib = p.in_dim >> 7, ob = p.out_dim >> 7;
  const int bi = blockIdx.x % ib, bo = blockIdx.x / ib;
  const int tn16 = p.out_dim >> 4;

  // Phase A: decode 64 tiles (16x16) of this 128x128 block into sW.
  for (int idx = tid; idx < 64 * 256; idx += kNumThreads) {
    const int t = idx >> 8, j = idx & 255;
    const int ti = t >> 3, tj = t & 7;
    const uint16_t *pay =
        trellis + (static_cast<std::size_t>(bi * 8 + ti) * tn16 + (bo * 8 + tj)) * TK;
    const uint32_t win = exl3_window<K>(pay, j);
    const float val = exl3_decode_cb(win, p.codebook);
    // tensor-core fragment placement (inverse of the bitstream order)
    const int t8 = j >> 3, s = j & 7;
    const int r0 = (t8 & 3) * 2, c0 = t8 >> 2;
    const int rr = r0 + (s & 1) + ((s >> 1) & 1) * 8;
    const int cc = c0 + ((s >> 2) << 3);
    sW[(ti << 4) + rr][(tj << 4) + cc] = val;
  }
  __syncthreads();

  // Phase B1: left hadamard, FWHT along the input dim (down the rows);
  // warp per column, 32 columns per warp.
  for (int c = warp; c < 128; c += kNumThreads / 32) {
    float x0 = sW[lane][c];
    float x1 = sW[lane + 32][c];
    float x2 = sW[lane + 64][c];
    float x3 = sW[lane + 96][c];
    fwht128(x0, x1, x2, x3, lane);
    sW[lane][c] = x0;
    sW[lane + 32][c] = x1;
    sW[lane + 64][c] = x2;
    sW[lane + 96][c] = x3;
  }
  __syncthreads();

  // Phase B2: right hadamard, FWHT along the output dim (across columns);
  // warp per row.
  for (int r = warp; r < 128; r += kNumThreads / 32) {
    float x0 = sW[r][lane];
    float x1 = sW[r][lane + 32];
    float x2 = sW[r][lane + 64];
    float x3 = sW[r][lane + 96];
    fwht128(x0, x1, x2, x3, lane);
    sW[r][lane] = x0;
    sW[r][lane + 32] = x1;
    sW[r][lane + 64] = x2;
    sW[r][lane + 96] = x3;
  }
  __syncthreads();

  // Phase C: signed scales + 1/128 (both unnormalized FWHTs) + bf16 write,
  // transposed to HF orientation out[out_dim, in_dim].
  constexpr float kInv128 = 1.0f / 128.0f;
  __nv_bfloat16 *out_m = p.out + static_cast<std::size_t>(blockIdx.y) * p.out_mat_stride;
  for (int co = warp; co < 128; co += kNumThreads / 32) {
    const float sv = __half2float(svh[bo * 128 + co]);
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int ri = lane + 32 * jj;
      const float v = sW[ri][co] *
                      __half2float(suh[bi * 128 + ri]) * sv * kInv128;
      out_m[(bo * 128 + co) * p.out_row_stride + bi * 128 + ri] =
          __float2bfloat16(v);
    }
  }
}

template <int k_bits, int num_threads = 128>
struct Exl3DequantKernel {
  static void run(const tvm::ffi::TensorView src, const tvm::ffi::TensorView out,
                  int64_t trellis_off, int64_t suh_off, int64_t svh_off,
                  int64_t in_dim, int64_t out_dim, int64_t codebook) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto R = SymbolicSize{"R"};
    auto O = SymbolicSize{"O"};
    auto I = SymbolicSize{"I"};
    auto MS = SymbolicSize{"MS"};
    auto RS = SymbolicSize{"RS"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({M, R})
        .with_strides({R, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(src);
    TensorMatcher({M, O, I})
        .with_strides({MS, RS, 1})
        .with_device<kDLCUDA>(device_)
        .verify(out);
    RuntimeCheck(in_dim == I.unwrap() && out_dim == O.unwrap(),
                 "in/out dim mismatch with out tensor");
    RuntimeCheck(in_dim % 128 == 0 && out_dim % 128 == 0,
                 "dims must be multiples of 128");

    const auto device = device_.unwrap();
    constexpr auto kSmem = static_cast<int>(128 * 129 * sizeof(float));
    static bool attr_set = [] {
      cudaFuncSetAttribute(exl3_dequant_kernel<k_bits, num_threads>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
      return true;
    }();
    (void)attr_set;
    const auto params = Exl3DequantParams{
        .src = static_cast<const uint8_t *>(src.data_ptr()),
        .out = static_cast<__nv_bfloat16 *>(out.data_ptr()),
        .src_stride = static_cast<std::size_t>(R.unwrap()),
        .out_mat_stride = static_cast<std::size_t>(MS.unwrap()),
        .out_row_stride = static_cast<std::size_t>(RS.unwrap()),
        .trellis_off = trellis_off,
        .suh_off = suh_off,
        .svh_off = svh_off,
        .in_dim = static_cast<int>(in_dim),
        .out_dim = static_cast<int>(out_dim),
        .codebook = static_cast<int>(codebook),
    };

    dim3 grid(static_cast<unsigned>((in_dim / 128) * (out_dim / 128)),
              static_cast<unsigned>(M.unwrap()));
    LaunchKernel(grid, num_threads, device, kSmem)(
        exl3_dequant_kernel<k_bits, num_threads>, params);
  }
};

} // namespace
