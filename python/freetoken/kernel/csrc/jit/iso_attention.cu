// IsoQuant (ISO3/ISO4) paged attention kernels (decode + extend).
//
// Decode: split-K over the packed pool (stage 1 per (request, kv_head, chunk),
// stage 2 merges chunks); Q heads of the GQA group share each dequantized K/V
// row. Online softmax in fp32. K/V are dequantized on the fly (iso_common.cuh).
//
// Extend (prefill): one threadblock per (request, kv_head, q_index); the query
// token attends the packed prefix [0, ctx) plus the bf16 extend tokens
// [0, q_index] (causal). New tokens are NOT read from the packed pool — the
// backend stores them after attention (deferred quantization), so prefill
// attention never consumes freshly-quantized values.

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "iso_common.cuh"

namespace {

using iso::IsoBlockBytes;

constexpr float kNegInf = -1e30f;

// Online-softmax state for one query head, kept in shared memory.
struct SoftmaxState {
  float m;      // running max
  float l;      // running sum
  float p;      // last token weight
  float alpha;  // last rescale factor
};

// Block reduce of per-thread partials for GT query heads, then softmax state
// update (by the first GT threads). kThreads = 64*NB (2 or 4 warps).
template <int GT, int kNumWarps>
__device__ __forceinline__ void block_softmax_update(
    float (&partial)[GT], float scale, SoftmaxState *states,
    float *red,  // [kNumWarps * GT]
    int tid) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int g = 0; g < GT; ++g) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial[g] += __shfl_xor_sync(0xffffffffu, partial[g], off);
  }
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < GT; ++g) red[warp * GT + g] = partial[g];
  __syncthreads();
  if (tid < GT) {
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) s += red[w * GT + tid];
    s *= scale;
    SoftmaxState &st = states[tid];
    const float m_new = fmaxf(st.m, s);
    const float alpha = expf(st.m - m_new);
    const float p = expf(s - m_new);
    st.l = st.l * alpha + p;
    st.m = m_new;
    st.p = p;
    st.alpha = alpha;
  }
  __syncthreads();
}

template <int GT>
__device__ __forceinline__ void accumulate_v(float (&acc0)[GT], float (&acc1)[GT],
                                             float vf0, float vf1,
                                             const SoftmaxState *states) {
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    acc0[g] = acc0[g] * states[g].alpha + states[g].p * vf0;
    acc1[g] = acc1[g] * states[g].alpha + states[g].p * vf1;
  }
}

template <int GT>
__device__ __forceinline__ void write_out_pair(__nv_bfloat16 *out_row, int g,
                                               int j0, float a0, float a1,
                                               float l) {
  const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
  __nv_bfloat162 val = __floats2bfloat162_rn(a0 * inv_l, a1 * inv_l);
  *reinterpret_cast<__nv_bfloat162 *>(out_row + j0) = val;
}

// ---------------------------------------------------------------------------
// Decode: q (R, Hq*D) bf16 against packed paged KV, split-K (flash-decoding).
// Stage 1: grid (R*Hkv, S); each block covers one KV chunk of one request,
// its warps take interleaved tokens and each lane dequantizes one 4D group per
// 128-block. Writes unnormalized partials (acc, max, sum) in the log2 domain.
// Stage 2: grid R*Hq; merges the S partials of one query head.
// ---------------------------------------------------------------------------

constexpr int kDecodeWarps = 4;

struct IsoAttnDecodeParams {
  const __nv_bfloat16 *__restrict__ q;
  const uint8_t *__restrict__ k_cache;
  const uint8_t *__restrict__ v_cache;
  const int32_t *__restrict__ kv_indptr;
  const int32_t *__restrict__ kv_indices;
  float *__restrict__ o_part;   // [R*S, Hq, D]
  float *__restrict__ ml_part;  // [R*S, Hq, 2]
  std::size_t q_row_bytes;
  std::size_t cache_row_bytes;
  int nheads_q;
  int nheads_kv;
  int num_splits;
  float scale_log2;
};

template <int FMT, int NB, int GT>
__global__ __launch_bounds__(32 * kDecodeWarps) void //
    iso_attn_decode_split_kernel(const __grid_constant__ IsoAttnDecodeParams params) {
  constexpr int D = NB * iso::kIsoBlock;
  constexpr int BB = IsoBlockBytes<FMT>::value;
  constexpr int kThreads = 32 * kDecodeWarps;

  const int req = blockIdx.x / params.nheads_kv;
  const int kvh = blockIdx.x % params.nheads_kv;
  const int split = blockIdx.y;
  const int gq = params.nheads_q / params.nheads_kv;
  if (gq > GT) return;  // host dispatches a sufficient GT

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // q (prescaled by scale*log2e), later reused as the cross-warp reduce buffer
  __shared__ __align__(16) float qs[GT * D];
  __shared__ float warp_m[kDecodeWarps][GT];
  __shared__ float warp_l[kDecodeWarps][GT];
  constexpr int kNumCent = FMT == 3 ? 8 : 16;
  __shared__ float cent[kNumCent];

  const __nv_bfloat16 *q_row = reinterpret_cast<const __nv_bfloat16 *>(
      reinterpret_cast<const uint8_t *>(params.q) + req * params.q_row_bytes);
  for (int idx = tid; idx < GT * D; idx += kThreads) {
    const int g = idx / D, e = idx % D;
    qs[idx] = g < gq ? __bfloat162float(q_row[(kvh * gq + g) * D + e]) * params.scale_log2
                     : 0.0f;
  }
  if (tid < kNumCent) cent[tid] = iso::centroid<FMT>(tid);
  float qc[4];
  iso::iso_conj_quat(lane, qc);  // lane always dequantizes group `lane`
  __syncthreads();

  const int kv_start = params.kv_indptr[req];
  const int kv_len = params.kv_indptr[req + 1] - kv_start;
  const int chunk = (kv_len + params.num_splits - 1) / params.num_splits;
  const int t_begin = split * chunk;
  const int t_end = min(kv_len, t_begin + chunk);

  float m[GT], l[GT], acc[GT][NB][4];
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    m[g] = kNegInf;
    l[g] = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[g][b][i] = 0.0f;
  }

  const std::size_t head_off = static_cast<std::size_t>(kvh) * NB * BB;
  const int32_t *slots = params.kv_indices + kv_start;
  // software pipeline: raw bytes of token t+W and the slot of token t+2W are
  // in flight while token t is computed, hiding the dependent global loads
  iso::IsoGroupRaw k_raw[NB], v_raw[NB];
  auto load_token = [&](int slot) {
    const uint8_t *krow =
        params.k_cache + static_cast<std::size_t>(slot) * params.cache_row_bytes + head_off;
    const uint8_t *vrow =
        params.v_cache + static_cast<std::size_t>(slot) * params.cache_row_bytes + head_off;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      k_raw[b] = iso::iso_load_group<FMT>(krow + b * BB, lane);
      v_raw[b] = iso::iso_load_group<FMT>(vrow + b * BB, lane);
    }
  };
  int t = t_begin + warp;
  if (t < t_end) load_token(slots[t]);
  int next_slot = t + kDecodeWarps < t_end ? slots[t + kDecodeWarps] : 0;
  for (; t < t_end; t += kDecodeWarps) {
    float kf[NB][4], vf[NB][4];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      iso::iso_decode_group<FMT>(k_raw[b], lane, cent, qc, kf[b]);
      iso::iso_decode_group<FMT>(v_raw[b], lane, cent, qc, vf[b]);
    }
    if (t + kDecodeWarps < t_end) {
      load_token(next_slot);
      if (t + 2 * kDecodeWarps < t_end) next_slot = slots[t + 2 * kDecodeWarps];
    }

    float s[GT];
#pragma unroll
    for (int g = 0; g < GT; ++g) {
      float dot = 0.0f;
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const float4 qv = reinterpret_cast<const float4 *>(
            qs + g * D + b * iso::kIsoBlock)[lane];
        dot += kf[b][0] * qv.x + kf[b][1] * qv.y + kf[b][2] * qv.z + kf[b][3] * qv.w;
      }
      s[g] = dot;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
      for (int g = 0; g < GT; ++g) s[g] += __shfl_xor_sync(0xffffffffu, s[g], off);

#pragma unroll
    for (int g = 0; g < GT; ++g) {
      // s is warp-reduced, so this branch is warp-uniform; the running max
      // rarely grows after the first tokens, so most steps skip the rescale
      if (s[g] > m[g]) {
        const float alpha = exp2f(m[g] - s[g]);
        m[g] = s[g];
        l[g] *= alpha;
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[g][b][i] *= alpha;
      }
      const float p = exp2f(s[g] - m[g]);
      l[g] += p;
#pragma unroll
      for (int b = 0; b < NB; ++b)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][b][i] += p * vf[b][i];
    }
  }

  // merge the per-warp partials through shared memory (q no longer needed)
  if (lane == 0) {
#pragma unroll
    for (int g = 0; g < GT; ++g) {
      warp_m[warp][g] = m[g];
      warp_l[warp][g] = l[g];
    }
  }
  __syncthreads();
  for (int idx = tid; idx < GT * D; idx += kThreads) qs[idx] = 0.0f;
  float blk_m[GT], blk_l[GT];
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    blk_m[g] = kNegInf;
#pragma unroll
    for (int w = 0; w < kDecodeWarps; ++w) blk_m[g] = fmaxf(blk_m[g], warp_m[w][g]);
    blk_l[g] = 0.0f;
#pragma unroll
    for (int w = 0; w < kDecodeWarps; ++w)
      blk_l[g] += warp_l[w][g] * exp2f(warp_m[w][g] - blk_m[g]);
  }
  __syncthreads();
#pragma unroll
  for (int w = 0; w < kDecodeWarps; ++w) {
    if (warp == w) {
#pragma unroll
      for (int g = 0; g < GT; ++g) {
        const float wscale = exp2f(m[g] - blk_m[g]);
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            qs[g * D + b * iso::kIsoBlock + 4 * lane + i] += acc[g][b][i] * wscale;
      }
    }
    __syncthreads();
  }

  const std::size_t row = static_cast<std::size_t>(req) * params.num_splits + split;
  for (int idx = tid; idx < gq * D; idx += kThreads) {
    const int g = idx / D, e = idx % D;
    const std::size_t qh = static_cast<std::size_t>(kvh) * gq + g;
    params.o_part[(row * params.nheads_q + qh) * D + e] = qs[idx];
  }
  if (tid < gq) {
    const std::size_t qh = static_cast<std::size_t>(kvh) * gq + tid;
    float blk_m_t = kNegInf, blk_l_t = 0.0f;
#pragma unroll
    for (int g = 0; g < GT; ++g)
      if (g == tid) { blk_m_t = blk_m[g]; blk_l_t = blk_l[g]; }
    params.ml_part[(row * params.nheads_q + qh) * 2 + 0] = blk_m_t;
    params.ml_part[(row * params.nheads_q + qh) * 2 + 1] = blk_l_t;
  }
}

struct IsoAttnDecodeMergeParams {
  const float *__restrict__ o_part;
  const float *__restrict__ ml_part;
  __nv_bfloat16 *__restrict__ out;
  std::size_t out_row_bytes;
  int nheads_q;
  int num_splits;
};

template <int NB>
__global__ __launch_bounds__(iso::kIsoBlock) void //
    iso_attn_decode_merge_kernel(const __grid_constant__ IsoAttnDecodeMergeParams params) {
  constexpr int D = NB * iso::kIsoBlock;
  const int req = blockIdx.x / params.nheads_q;
  const int qh = blockIdx.x % params.nheads_q;
  const int tid = threadIdx.x;
  const int S = params.num_splits;
  const std::size_t base = static_cast<std::size_t>(req) * S;

  float gm = kNegInf;
  for (int s = 0; s < S; ++s)
    gm = fmaxf(gm, params.ml_part[((base + s) * params.nheads_q + qh) * 2]);
  float gl = 0.0f, o[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) o[b] = 0.0f;
  for (int s = 0; s < S; ++s) {
    const std::size_t r = (base + s) * params.nheads_q + qh;
    const float w = exp2f(params.ml_part[r * 2] - gm);
    gl += params.ml_part[r * 2 + 1] * w;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      o[b] += params.o_part[r * D + b * iso::kIsoBlock + tid] * w;
  }
  const float inv_l = gl > 0.0f ? 1.0f / gl : 0.0f;
  __nv_bfloat16 *out_row = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<uint8_t *>(params.out) + req * params.out_row_bytes) + qh * D;
#pragma unroll
  for (int b = 0; b < NB; ++b)
    out_row[b * iso::kIsoBlock + tid] = __float2bfloat16(o[b] * inv_l);
}

// ---------------------------------------------------------------------------
// Extend: q (total_new, Hq*D) bf16 against packed prefix + bf16 extend tokens.
// gridDim.y = max extend length; q token i attends prefix + extend[0..=i].
// ---------------------------------------------------------------------------

struct IsoAttnExtendParams {
  const __nv_bfloat16 *__restrict__ q;
  __nv_bfloat16 *__restrict__ out;
  const uint8_t *__restrict__ k_cache;
  const uint8_t *__restrict__ v_cache;
  const __nv_bfloat16 *__restrict__ k_ext;
  const __nv_bfloat16 *__restrict__ v_ext;
  const int32_t *__restrict__ qo_indptr;
  const int32_t *__restrict__ kv_indptr;   // prefix only
  const int32_t *__restrict__ kv_indices;  // prefix slots
  std::size_t q_row_bytes;
  std::size_t out_row_bytes;
  std::size_t cache_row_bytes;
  std::size_t ext_row_bytes;
  int nheads_q;
  int nheads_kv;
  int nb;
  float scale;
};

template <int FMT, int NB, int GT>
__global__ __launch_bounds__(64 * NB) void //
    iso_attn_extend_kernel(const __grid_constant__ IsoAttnExtendParams params) {
  constexpr int D = NB * iso::kIsoBlock;
  constexpr int BB = IsoBlockBytes<FMT>::value;
  constexpr int kThreads = 64 * NB;
  constexpr int kNumWarps = kThreads / 32;

  const int req = blockIdx.x / params.nheads_kv;
  const int kvh = blockIdx.x % params.nheads_kv;
  const int qi = blockIdx.y;
  const int q_start = params.qo_indptr[req];
  const int q_end = params.qo_indptr[req + 1];
  const int n_new = q_end - q_start;
  if (qi >= n_new) return;
  const int gq = params.nheads_q / params.nheads_kv;
  if (gq > GT) return;

  const int tid = threadIdx.x;
  const int b = tid >> 6;
  const int u = tid & 63;
  const int j0 = b * iso::kIsoBlock + 2 * u;

  extern __shared__ float smem[];
  float *qs = smem;
  float *red = qs + GT * D;
  SoftmaxState *states =
      reinterpret_cast<SoftmaxState *>(red + kNumWarps * GT);

  const __nv_bfloat16 *q_row = reinterpret_cast<const __nv_bfloat16 *>(
      reinterpret_cast<const uint8_t *>(params.q) +
      static_cast<std::size_t>(q_start + qi) * params.q_row_bytes);
  for (int idx = tid; idx < gq * D; idx += kThreads) {
    const int g = idx / D, e = idx % D;
    qs[idx] = __bfloat162float(q_row[(kvh * gq + g) * D + e]);
  }
  if (tid < gq) states[tid] = {kNegInf, 0.0f, 0.0f, 0.0f};
  __syncthreads();

  float acc0[GT], acc1[GT];
#pragma unroll
  for (int g = 0; g < GT; ++g) { acc0[g] = 0.0f; acc1[g] = 0.0f; }

  // 1) packed prefix
  const int kv_start = params.kv_indptr[req];
  const int kv_end = params.kv_indptr[req + 1];
  for (int t = kv_start; t < kv_end; ++t) {
    const std::size_t slot = static_cast<std::size_t>(params.kv_indices[t]);
    const uint8_t *krow = params.k_cache + slot * params.cache_row_bytes +
                          kvh * (params.nb * BB);
    const uint8_t *vrow = params.v_cache + slot * params.cache_row_bytes +
                          kvh * (params.nb * BB);
    float kf0, kf1, vf0, vf1;
    iso::iso_dequant_pair<FMT>(krow + b * BB, u, kf0, kf1);
    iso::iso_dequant_pair<FMT>(vrow + b * BB, u, vf0, vf1);

    float partial[GT];
#pragma unroll
    for (int g = 0; g < GT; ++g)
      partial[g] = kf0 * qs[g * D + j0] + kf1 * qs[g * D + j0 + 1];
    block_softmax_update<GT, kNumWarps>(partial, params.scale, states, red, tid);
    accumulate_v<GT>(acc0, acc1, vf0, vf1, states);
    __syncthreads();
  }

  // 2) bf16 extend tokens [0..=qi] (causal)
  const __nv_bfloat16 *k_ext_rows = params.k_ext + kvh * D;
  const __nv_bfloat16 *v_ext_rows = params.v_ext + kvh * D;
  for (int t = 0; t <= qi; ++t) {
    const std::size_t row = static_cast<std::size_t>(q_start + t);
    const __nv_bfloat16 *krow = reinterpret_cast<const __nv_bfloat16 *>(
        reinterpret_cast<const uint8_t *>(k_ext_rows) + row * params.ext_row_bytes);
    const __nv_bfloat16 *vrow = reinterpret_cast<const __nv_bfloat16 *>(
        reinterpret_cast<const uint8_t *>(v_ext_rows) + row * params.ext_row_bytes);
    const float kf0 = __bfloat162float(krow[j0]);
    const float kf1 = __bfloat162float(krow[j0 + 1]);
    const float vf0 = __bfloat162float(vrow[j0]);
    const float vf1 = __bfloat162float(vrow[j0 + 1]);

    float partial[GT];
#pragma unroll
    for (int g = 0; g < GT; ++g)
      partial[g] = kf0 * qs[g * D + j0] + kf1 * qs[g * D + j0 + 1];
    block_softmax_update<GT, kNumWarps>(partial, params.scale, states, red, tid);
    accumulate_v<GT>(acc0, acc1, vf0, vf1, states);
    __syncthreads();
  }

  __nv_bfloat16 *out_row = reinterpret_cast<__nv_bfloat16 *>(
      reinterpret_cast<uint8_t *>(params.out) +
      static_cast<std::size_t>(q_start + qi) * params.out_row_bytes);
#pragma unroll
  for (int g = 0; g < GT; ++g) {
    if (g < gq)
      write_out_pair<GT>(out_row + (kvh * gq + g) * D, g, j0, acc0[g], acc1[g],
                         states[g].l);
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

inline int gt_dispatch_target(int gq) {
  int gt = 1;
  while (gt < gq) gt <<= 1;
  return gt;
}

template <int fmt_bits, int nb, int gt>
struct IsoAttnDecodeKernel {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView kv_indices,
                  const tvm::ffi::TensorView o_part,
                  const tvm::ffi::TensorView ml_part, int64_t nheads_q,
                  int64_t nheads_kv, int64_t head_dim, double scale,
                  int64_t num_splits) {
    using namespace host;
    auto R = SymbolicSize{"R"};
    auto E = SymbolicSize{"E"};
    auto B = SymbolicSize{"B"};
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto W = SymbolicSize{"W"};
    auto P = SymbolicSize{"P"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({R, E})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .verify(q);
    TensorMatcher({R, E})
        .with_strides({W, 1})
        .with_device<kDLCUDA>(device_)
        .verify(out);
    TensorMatcher({-1, B})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({-1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indptr);
    TensorMatcher({-1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indices);
    TensorMatcher({P, nheads_q, head_dim})
        .with_device<kDLCUDA>(device_)
        .with_dtype<float>()
        .verify(o_part);
    TensorMatcher({P, nheads_q, 2})
        .with_device<kDLCUDA>(device_)
        .with_dtype<float>()
        .verify(ml_part);
    RuntimeCheck(head_dim == nb * iso::kIsoBlock, "head_dim mismatch with template nb");
    RuntimeCheck(gt_dispatch_target(static_cast<int>(nheads_q / nheads_kv)) == gt,
                 "gt template mismatch");
    RuntimeCheck(num_splits >= 1 && num_splits <= 65535, "num_splits out of range");

    const auto device = device_.unwrap();
    const auto num_reqs = static_cast<std::size_t>(R.unwrap());
    RuntimeCheck(P.unwrap() >= static_cast<int64_t>(num_reqs) * num_splits,
                 "decode partial buffers too small");
    if (num_reqs == 0) return;

    constexpr float kLog2e = 1.4426950408889634f;
    const auto params = IsoAttnDecodeParams{
        .q = static_cast<const __nv_bfloat16 *>(q.data_ptr()),
        .k_cache = static_cast<const uint8_t *>(k_cache.data_ptr()),
        .v_cache = static_cast<const uint8_t *>(v_cache.data_ptr()),
        .kv_indptr = static_cast<const int32_t *>(kv_indptr.data_ptr()),
        .kv_indices = static_cast<const int32_t *>(kv_indices.data_ptr()),
        .o_part = static_cast<float *>(o_part.data_ptr()),
        .ml_part = static_cast<float *>(ml_part.data_ptr()),
        .q_row_bytes = static_cast<std::size_t>(Y.unwrap()) * sizeof(__nv_bfloat16),
        .cache_row_bytes = static_cast<std::size_t>(X.unwrap()),
        .nheads_q = static_cast<int>(nheads_q),
        .nheads_kv = static_cast<int>(nheads_kv),
        .num_splits = static_cast<int>(num_splits),
        .scale_log2 = static_cast<float>(scale) * kLog2e,
    };
    dim3 grid(static_cast<unsigned>(num_reqs * static_cast<std::size_t>(nheads_kv)),
              static_cast<unsigned>(num_splits));
    LaunchKernel(grid, 32 * kDecodeWarps, device)(
        iso_attn_decode_split_kernel<fmt_bits, nb, gt>, params);

    const auto merge = IsoAttnDecodeMergeParams{
        .o_part = static_cast<const float *>(o_part.data_ptr()),
        .ml_part = static_cast<const float *>(ml_part.data_ptr()),
        .out = static_cast<__nv_bfloat16 *>(out.data_ptr()),
        .out_row_bytes = static_cast<std::size_t>(W.unwrap()) * sizeof(__nv_bfloat16),
        .nheads_q = static_cast<int>(nheads_q),
        .num_splits = static_cast<int>(num_splits),
    };
    LaunchKernel(num_reqs * static_cast<std::size_t>(nheads_q), iso::kIsoBlock, device)(
        iso_attn_decode_merge_kernel<nb>, merge);
  }
};

template <int fmt_bits, int nb, int gt>
struct IsoAttnExtendKernel {
  static void run(const tvm::ffi::TensorView q, const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView k_ext,
                  const tvm::ffi::TensorView v_ext,
                  const tvm::ffi::TensorView qo_indptr,
                  const tvm::ffi::TensorView kv_indptr,
                  const tvm::ffi::TensorView kv_indices, int64_t nheads_q,
                  int64_t nheads_kv, int64_t head_dim, double scale,
                  int64_t max_extend) {
    using namespace host;
    auto R = SymbolicSize{"R"};
    auto E = SymbolicSize{"E"};
    auto B = SymbolicSize{"B"};
    auto X = SymbolicSize{"X"};
    auto Y = SymbolicSize{"Y"};
    auto Z = SymbolicSize{"Z"};
    auto W = SymbolicSize{"W"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, E})
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .verify(q);
    TensorMatcher({-1, E})
        .with_strides({W, 1})
        .with_device<kDLCUDA>(device_)
        .verify(out);
    TensorMatcher({-1, -1})
        .with_strides({Z, 1})
        .with_device<kDLCUDA>(device_)
        .verify(k_ext)
        .verify(v_ext);
    TensorMatcher({-1, B})
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<uint8_t>()
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({R}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(qo_indptr);
    TensorMatcher({R}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indptr);
    TensorMatcher({-1}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(kv_indices);
    RuntimeCheck(head_dim == nb * iso::kIsoBlock, "head_dim mismatch with template nb");
    RuntimeCheck(gt_dispatch_target(static_cast<int>(nheads_q / nheads_kv)) == gt,
                 "gt template mismatch");

    const auto device = device_.unwrap();
    constexpr int kThreads = 64 * nb;
    constexpr int kNumWarps = kThreads / 32;
    const std::size_t smem =
        sizeof(float) * (gt * head_dim + kNumWarps * gt) + sizeof(SoftmaxState) * gt;

    const auto params = IsoAttnExtendParams{
        .q = static_cast<const __nv_bfloat16 *>(q.data_ptr()),
        .out = static_cast<__nv_bfloat16 *>(out.data_ptr()),
        .k_cache = static_cast<const uint8_t *>(k_cache.data_ptr()),
        .v_cache = static_cast<const uint8_t *>(v_cache.data_ptr()),
        .k_ext = static_cast<const __nv_bfloat16 *>(k_ext.data_ptr()),
        .v_ext = static_cast<const __nv_bfloat16 *>(v_ext.data_ptr()),
        .qo_indptr = static_cast<const int32_t *>(qo_indptr.data_ptr()),
        .kv_indptr = static_cast<const int32_t *>(kv_indptr.data_ptr()),
        .kv_indices = static_cast<const int32_t *>(kv_indices.data_ptr()),
        .q_row_bytes = static_cast<std::size_t>(Y.unwrap()) * sizeof(__nv_bfloat16),
        .out_row_bytes = static_cast<std::size_t>(W.unwrap()) * sizeof(__nv_bfloat16),
        .cache_row_bytes = static_cast<std::size_t>(X.unwrap()),
        .ext_row_bytes = static_cast<std::size_t>(Z.unwrap()) * sizeof(__nv_bfloat16),
        .nheads_q = static_cast<int>(nheads_q),
        .nheads_kv = static_cast<int>(nheads_kv),
        .nb = static_cast<int>(nb),
        .scale = static_cast<float>(scale),
    };

    const auto num_reqs = static_cast<std::size_t>(R.unwrap() - 1);
    dim3 grid(static_cast<unsigned>(num_reqs * static_cast<std::size_t>(nheads_kv)),
              static_cast<unsigned>(max_extend));
    LaunchKernel(grid, kThreads, device, smem)(
        iso_attn_extend_kernel<fmt_bits, nb, gt>, params);
  }
};

} // namespace
