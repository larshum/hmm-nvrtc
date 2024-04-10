////////////////////
// GENERATED CODE //
////////////////////

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

typedef uint32_t state_t;
typedef uint8_t obs_t;
typedef float prob_t;

#define BATCH_SIZE 1024
#define BATCH_OVERLAP 128

#define NUM_STATES 262144
#define NUM_OBS 101
#define NUM_PREDS 5

#define INIT_PROB(x) (init_prob((initp), (x)))
#define OUTPUT_PROB(x,y) (output_prob((outp), (x), (y)))
#define FIND_MAX_PREDS(chi_prev,instance,state,maxs,maxp) \
  (generated_viterbi_max_predecessor((chi_prev), (instance), (state), (maxs), (maxp), gamma, gamma_inv, trans1, trans2))
#define FIND_PROB_PREDS(alpha_prev,instance,state,probs) \
  (generated_forward_prob_predecessors((alpha_prev), (instance), (state), (probs), gamma, gamma_inv, trans1, trans2))

__device__
prob_t init_prob(const prob_t *initp, state_t x) {
  return initp[((x / 16) % 16384) * 16 + x % 16];
}

__device__
prob_t output_prob(const prob_t *outp, state_t x, obs_t o) {
  return outp[o * 16384 + x / 16];
}

__device__
prob_t transp1(
    const prob_t *trans1, const prob_t *trans2,
    state_t x, state_t y) {
  return trans1[x / 16 * 4 + y / 16 % 4] + trans2[y % 16];
}

__device__
prob_t transp2(const prob_t gamma) {
  return gamma;
}

__device__
prob_t transp3(const prob_t gamma_inv) {
  return gamma_inv;
}

__device__
prob_t transp4() {
  return 0.0;
}

__device__
void generated_viterbi_max_predecessor(
    const prob_t *chi_prev, int instance, state_t state, state_t *maxs, prob_t *maxp,
    prob_t gamma, prob_t gamma_inv, const prob_t *trans1, const prob_t *trans2) {

  state_t s;
  prob_t p;

  // NOTE: reordering so that the loop takes place last allows us to skip one
  // max comparison. This has a significant impact on performance, so it should
  // be performed by the compiler.
  if (state % 16 == 15) {
    *maxs = state;
    *maxp = chi_prev[instance * NUM_STATES + *maxs] + transp2(gamma);
  }

  if (state % 16 == 14) {
    *maxs = state + 1;
    *maxp = chi_prev[instance * NUM_STATES + *maxs] + transp3(gamma_inv);
  }

  if (state % 16 != 14 && state % 16 != 15) {
    *maxs = state + 1;
    *maxp = chi_prev[instance * NUM_STATES + *maxs] + transp4();
  }

  for (int k = 0; k < 4; k++) {
    s = state / 64 % 4096 * 16 + k * 65536;
    p = chi_prev[instance * NUM_STATES + s] + transp1(trans1, trans2, s, state);
    if (p > *maxp) {
      *maxs = s;
      *maxp = p;
    }
  }
}

// NOTE: if the compiler fails to generate this more efficient approach, it
// could fall back to a version that uses precomputed predecessors.
__device__
int generated_forward_prob_predecessors(
    const prob_t *alpha_prev, int instance, state_t state, prob_t *probs,
    prob_t gamma, prob_t gamma_inv, const prob_t *trans1, const prob_t *trans2) {

  int pidx = 0;
  state_t pred;
  for (int k = 0; k < 4; k++) {
    pred = state / 64 % 4096 * 16 + k * 65536;
    probs[pidx] = alpha_prev[instance * NUM_STATES + pred] + transp1(trans1, trans2, pred, state);
    pidx += 1;
  }

  if (state % 16 == 15) {
    pred = state;
    probs[pidx] = alpha_prev[instance * NUM_STATES + pred] + transp2(gamma);
    pidx += 1;
  }

  if (state % 16 == 14) {
    pred = state + 1;
    probs[pidx] = alpha_prev[instance * NUM_STATES + pred] + transp3(gamma_inv);
    pidx += 1;
  }

  if (state % 16 != 14 && state % 16 != 15) {
    pred = state + 1;
    probs[pidx] = alpha_prev[instance * NUM_STATES + pred] + transp4();
    pidx += 1;
  }

  return pidx;
}

// We use the below macros to enable a clean separation between our generated
// model-specific code and a generic model-independent implementation of the
// Forward and Viterbi algorithms.

#define FORWARD_INIT_DECL(args...) \
  __global__ void forward_init(args, const prob_t *initp, const prob_t *outp)

#define FORWARD_STEP_DECL(args...) \
  __global__ void forward_step(args, const prob_t *outp, const prob_t *trans1, const prob_t *trans2, const prob_t gamma, const prob_t gamma_inv)

#define VITERBI_INIT(args...) \
  __global__ void viterbi_init(args, const prob_t *initp, const prob_t *outp)

#define VITERBI_INIT_BATCH(args...) \
  __global__ void viterbi_init_batch(args, const prob_t *outp)

#define VITERBI_FORWARD(args...) \
  __global__ void viterbi_forward(args, const prob_t *outp, const prob_t *trans1, const prob_t *trans2, const prob_t gamma, const prob_t gamma_inv)

/////////////////////////////
// GENERAL IMPLEMENTATIONS //
/////////////////////////////

const prob_t inf = 1.0 / 0.0;

extern "C" FORWARD_INIT_DECL(const obs_t *obs, int maxlen, prob_t *alpha_zero) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    obs_t x = obs[instance * maxlen];
    alpha_zero[instance * NUM_STATES + state] = INIT_PROB(state) + OUTPUT_PROB(state, x);
  }
}

__device__
prob_t log_sum_exp(const prob_t probs[NUM_PREDS]) {
  prob_t maxp = probs[0];
  for (int i = 1; i < NUM_PREDS; i++) {
    if (probs[i] > maxp) maxp = probs[i];
  }
  prob_t sum = 0.0;
  for (int i = 0; i < NUM_PREDS; i++) {
    sum += expf(probs[i] - maxp);
  }
  return maxp + logf(sum);
}

extern "C" FORWARD_STEP_DECL(
    const obs_t *obs, const int *obs_lens, int maxlen,
    const prob_t *alpha_prev, prob_t *alpha_curr, int t) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    size_t idx = instance * NUM_STATES + state;
    if (t < obs_lens[instance]) {
      obs_t x = obs[instance * maxlen + t];
      prob_t psum;
      prob_t probs[NUM_PREDS];
      int pidx = FIND_PROB_PREDS(alpha_prev, instance, state, probs);
      while (pidx < NUM_PREDS) probs[pidx++] = -inf;
      psum = log_sum_exp(probs) + OUTPUT_PROB(state, x);
      alpha_curr[idx] = psum;
    } else if (t == obs_lens[instance]) {
      alpha_curr[idx] = alpha_prev[idx];
    }
  }
}

__device__
void forward_max_warp_reduce(volatile prob_t *maxp, unsigned int tid) {
  if (maxp[tid + 32] > maxp[tid]) {
    maxp[tid] = maxp[tid + 32];
  }
  if (maxp[tid + 16] > maxp[tid]) {
    maxp[tid] = maxp[tid + 16];
  }
  if (maxp[tid + 8] > maxp[tid]) {
    maxp[tid] = maxp[tid + 8];
  }
  if (maxp[tid + 4] > maxp[tid]) {
    maxp[tid] = maxp[tid + 4];
  }
  if (maxp[tid + 2] > maxp[tid]) {
    maxp[tid] = maxp[tid + 2];
  }
  if (maxp[tid + 1] > maxp[tid]) {
    maxp[tid] = maxp[tid + 1];
  }
}

extern "C"
__global__
void forward_max(
    const prob_t *alpha, prob_t *result) {
  unsigned int idx = threadIdx.x;
  unsigned int instance = blockIdx.x;
  unsigned int lo = instance * NUM_STATES;

  __shared__ prob_t maxp[512];
  maxp[idx] = alpha[lo + idx];
  for (int i = lo + idx; i < lo + NUM_STATES; i += 512) {
    if (alpha[i] > maxp[idx]) {
      maxp[idx] = alpha[i];
    }
  }
  __syncthreads();
  
  if (idx < 256) {
    if (maxp[idx + 256] > maxp[idx]) {
      maxp[idx] = maxp[idx + 256];
    }
  }
  __syncthreads();
  if (idx < 128) {
    if (maxp[idx + 128] > maxp[idx]) {
      maxp[idx] = maxp[idx + 128];
    }
  }
  __syncthreads();
  if (idx < 64) {
    if (maxp[idx + 64] > maxp[idx]) {
      maxp[idx] = maxp[idx + 64];
    }
  }
  __syncthreads();
  if (idx < 32) forward_max_warp_reduce(maxp, idx);

  if (idx == 0) {
    result[instance] = maxp[0];
  }
}

__device__
void forward_sum_warp_reduce(volatile prob_t *psum, unsigned int tid) {
  psum[tid] = psum[tid] + psum[tid + 32];
  psum[tid] = psum[tid] + psum[tid + 16];
  psum[tid] = psum[tid] + psum[tid + 8];
  psum[tid] = psum[tid] + psum[tid + 4];
  psum[tid] = psum[tid] + psum[tid + 2];
  psum[tid] = psum[tid] + psum[tid + 1];
}

extern "C"
__global__
void forward_log_sum_exp(const prob_t *alpha, prob_t *result) {
  unsigned int idx = threadIdx.x;
  unsigned int instance = blockIdx.x;
  unsigned int lo = instance * NUM_STATES;

  // Retrieve the maximum value for the current instance, as computed in the
  // max kernel.
  prob_t maxp = result[instance];

  __shared__ prob_t psum[512];
  psum[idx] = expf(alpha[lo + idx] - maxp);
  for (int i = lo + idx; i < lo + NUM_STATES; i += 512) {
    psum[idx] = psum[idx] + expf(alpha[i] - maxp);
  }
  __syncthreads();

  // Compute the sum of all these exponents
  if (idx < 256) psum[idx] = psum[idx] + psum[idx + 256];
  __syncthreads();
  if (idx < 128) psum[idx] = psum[idx] + psum[idx + 128];
  __syncthreads();
  if (idx < 64) psum[idx] = psum[idx] + psum[idx + 64];
  __syncthreads();
  if (idx < 32) forward_sum_warp_reduce(psum, idx);

  // The first thread of each block writes the result
  if (idx == 0) {
    result[instance] = maxp + logf(psum[0]);
  }
}

extern "C" VITERBI_INIT(
    const obs_t *obs, int maxlen, prob_t *chi_zero) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    obs_t x = obs[instance * maxlen];
    chi_zero[instance * NUM_STATES + state] = INIT_PROB(state) + OUTPUT_PROB(state, x);
  }
}

extern "C" VITERBI_INIT_BATCH(
    const obs_t *obs, const int *obs_lens, int maxlen, const state_t *seq,
    prob_t *chi_zero, int t) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    if (t < obs_lens[instance]) {
      obs_t x = obs[instance * maxlen + t];
      state_t last_state = seq[instance * maxlen + t - 1];
      if (state == last_state) {
        chi_zero[instance * NUM_STATES + state] = OUTPUT_PROB(state, x);
      } else {
        chi_zero[instance * NUM_STATES + state] = -inf;
      }
    }
  }
}

extern "C" VITERBI_FORWARD(
    const obs_t *obs, const int *obs_lens, int maxlen, const prob_t *chi_prev,
    prob_t *chi_curr, state_t *zeta, int t, int k) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    size_t idx = instance * NUM_STATES + state;
    size_t zeta_idx = instance * BATCH_SIZE * NUM_STATES + (k-1) * NUM_STATES + state;
    if (t+k < obs_lens[instance]) {
      obs_t x = obs[instance * maxlen + t + k];
      state_t maxs;
      prob_t maxp = -inf;
      FIND_MAX_PREDS(chi_prev, instance, state, &maxs, &maxp);
      maxp += OUTPUT_PROB(state, x);
      chi_curr[idx] = maxp;
      zeta[zeta_idx] = maxs;
    } else if (t+k == obs_lens[instance]) {
      // We only need to copy over data once - past this point, we know both
      // chi vectors will contain identical information. We set the zeta matrix
      // as below to ensure we backtrack through it correctly.
      chi_curr[idx] = chi_prev[idx];
      zeta[zeta_idx] = state;
    } else {
      zeta[zeta_idx] = state;
    }
  }
}

__device__
void viterbi_backward_warp_reduce(volatile prob_t *maxp, volatile state_t *maxs, unsigned int tid) {
  if (maxp[tid + 32] > maxp[tid]) {
    maxp[tid] = maxp[tid + 32];
    maxs[tid] = maxs[tid + 32];
  }
  if (maxp[tid + 16] > maxp[tid]) {
    maxp[tid] = maxp[tid + 16];
    maxs[tid] = maxs[tid + 16];
  }
  if (maxp[tid + 8] > maxp[tid]) {
    maxp[tid] = maxp[tid + 8];
    maxs[tid] = maxs[tid + 8];
  }
  if (maxp[tid + 4] > maxp[tid]) {
    maxp[tid] = maxp[tid + 4];
    maxs[tid] = maxs[tid + 4];
  }
  if (maxp[tid + 2] > maxp[tid]) {
    maxp[tid] = maxp[tid + 2];
    maxs[tid] = maxs[tid + 2];
  }
  if (maxp[tid + 1] > maxp[tid]) {
    maxp[tid] = maxp[tid + 1];
    maxs[tid] = maxs[tid + 1];
  }
}

extern "C"
__global__
void viterbi_backward(
    const prob_t *chi, const state_t *zeta, state_t *out, int maxlen, int T) {
  size_t idx = threadIdx.x;
  size_t instance = blockIdx.x;
  size_t lo = instance * NUM_STATES;

  __shared__ state_t maxs[512];
  __shared__ prob_t maxp[512];
  maxs[idx] = idx;
  maxp[idx] = chi[lo + idx];
  for (int i = lo + idx; i < lo + NUM_STATES; i += 512) {
    if (chi[i] > maxp[idx]) {
      maxp[idx] = chi[i];
      maxs[idx] = i - lo;
    }
  }
  __syncthreads();

  if (idx < 256) {
    if (maxp[idx + 256] > maxp[idx]) {
      maxp[idx] = maxp[idx + 256];
      maxs[idx] = maxs[idx + 256];
    }
  }
  __syncthreads();
  if (idx < 128) {
    if (maxp[idx + 128] > maxp[idx]) {
      maxp[idx] = maxp[idx + 128];
      maxs[idx] = maxs[idx + 128];
    }
  }
  __syncthreads();
  if (idx < 64) {
    if (maxp[idx + 64] > maxp[idx]) {
      maxp[idx] = maxp[idx + 64];
      maxs[idx] = maxs[idx + 64];
    }
  }
  __syncthreads();
  if (idx < 32) viterbi_backward_warp_reduce(maxp, maxs, idx);

  // Run the backtracking sequentially from the maximum state using the first
  // thread for each instance.
  if (idx == 0) {
    state_t max_state = maxs[0];
    state_t *outptr = out + instance * maxlen + T;
    const state_t *zetaptr = zeta + instance * BATCH_SIZE * NUM_STATES;
    outptr[BATCH_SIZE-1] = max_state;
    for (int t = BATCH_SIZE-2; t >= 0; t--) {
      outptr[t] = zetaptr[t * NUM_STATES + outptr[t+1]];
    }
  }
}
