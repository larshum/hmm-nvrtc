from cuda import cuda, nvrtc
import numpy as np
import math

def cuda_check(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {cuda.cuGetErrorString(err)}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"CUDA error: {nvrtc.nvrtcGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")

class HMM:
    def compile_cuda(self, src):
        with open(src, "r") as f:
            cu = f.read()

        err, prog = nvrtc.nvrtcCreateProgram(str.encode(cu), bytes(src, 'utf-8'), 0, [], [])
        cuda_check(err)

        opts = [b"-default-device", b"-lineinfo"]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        cuda_check(err)

        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        cuda_check(err)
        ptx = b" " * ptxSize
        err, = nvrtc.nvrtcGetPTX(prog, ptx)
        cuda_check(err)

        err, = cuda.cuInit(0)
        cuda_check(err)
        err, cuDevice = cuda.cuDeviceGet(0)
        cuda_check(err)
        err, context = cuda.cuCtxCreate(0, cuDevice)
        cuda_check(err)
        ptx = np.char.array(ptx)
        err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
        cuda_check(err)

        self.ctx = context
        self.module = module

    def copy_to_gpu(self, arg, stream):
        sz = len(arg) * arg.itemsize
        err, gpu_data = cuda.cuMemAllocAsync(sz, stream)
        cuda_check(err)
        err, = cuda.cuMemcpyHtoDAsync(gpu_data, arg.ctypes.data, sz, stream)
        cuda_check(err)
        return gpu_data

    def load_cuda_function(self, name):
        err, fun = cuda.cuModuleGetFunction(self.module, name)
        cuda_check(err)
        return fun

    def copy_from_gpu(self, gpu_data, n, ty, stream):
        result = np.zeros(n, dtype=ty)
        sz = n * result.itemsize
        err, = cuda.cuMemcpyDtoHAsync(result.ctypes.data, gpu_data, sz, stream)
        cuda_check(err)
        return result

    def run_kernel(self, kernel, blockdim, threaddim, stream, args):
        # Convert arguments to a void** pointer
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        (bx, by, bz) = blockdim
        (tx, ty, tz) = threaddim
        err, = cuda.cuLaunchKernel(kernel, bx, by, bz, tx, ty, tz, 0, stream, args.ctypes.data, 0)
        cuda_check(err)

    def _forward(self, obs, obs_lens):
        num_instances = len(obs_lens)

        # Load the functions defined in the CUDA code
        forward_init = self.load_cuda_function(b"forward_init")
        forward_step = self.load_cuda_function(b"forward_step")
        forward_max = self.load_cuda_function(b"forward_max")
        forward_log_sum_exp = self.load_cuda_function(b"forward_log_sum_exp")

        err, stream = cuda.cuStreamCreate(0)
        cuda_check(err)

        # Copy data to the GPU
        maxlen = np.array(max(obs_lens), dtype=np.int32)
        cu_obs_lens = self.copy_to_gpu(obs_lens.astype(dtype=np.int32), stream)
        cu_obs = self.copy_to_gpu(obs.astype(dtype=self.obs_type), stream)

        # Allocate working data on the GPU
        alpha_sz = num_instances * self.num_states * np.dtype(self.prob_type).itemsize
        err, alpha_src = cuda.cuMemAllocAsync(alpha_sz, stream)
        cuda_check(err)
        err, alpha_dst = cuda.cuMemAllocAsync(alpha_sz, stream)
        cuda_check(err)
        result_sz = num_instances * np.dtype(self.prob_type).itemsize
        err, cu_result = cuda.cuMemAllocAsync(result_sz, stream)
        cuda_check(err)

        # Perform the Forward algorithm using the CUDA kernels
        tpb = 256
        xblocks = (self.num_states + tpb - 1) // tpb
        blockdim = (xblocks, num_instances, 1)
        threaddim = (tpb, 1, 1)

        # Initialization
        args = [
            np.array([int(cu_obs)], dtype=np.uint64),
            maxlen,
            np.array([int(alpha_src)], dtype=np.uint64),
        ] + self.table_ptrs
        self.run_kernel(forward_init, blockdim, threaddim, stream, args)

        # Forward steps
        for t in range(1, maxlen):
            args = [
                np.array([int(cu_obs)], dtype=np.uint64),
                np.array([int(cu_obs_lens)], dtype=np.uint64),
                maxlen,
                np.array([int(alpha_src)], dtype=np.uint64),
                np.array([int(alpha_dst)], dtype=np.uint64),
                np.array(t, dtype=np.int32),
            ] + self.table_ptrs
            self.run_kernel(forward_step, blockdim, threaddim, stream, args)
            alpha_src, alpha_dst = alpha_dst, alpha_src

        # LogSumExp step (compute max + logarithmic sum)
        reduce_blockdim = (num_instances, 1, 1)
        reduce_threaddim = (512, 1, 1)
        args = [
            np.array([int(alpha_src)], dtype=np.uint64),
            np.array([int(cu_result)], dtype=np.uint64)
        ]
        self.run_kernel(forward_max, reduce_blockdim, reduce_threaddim, stream, args)
        args = [
            np.array([int(alpha_src)], dtype=np.uint64),
            np.array([int(cu_result)], dtype=np.uint64)
        ]
        self.run_kernel(forward_log_sum_exp, reduce_blockdim, reduce_threaddim, stream, args)

        # Copy result from the GPU
        result = self.copy_from_gpu(cu_result, num_instances, self.prob_type, stream)

        # Free up allocated data on the GPU
        err, = cuda.cuMemFreeAsync(cu_obs_lens, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(cu_obs, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(alpha_src, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(alpha_dst, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(cu_result, stream)
        cuda_check(err)

        err, = cuda.cuStreamSynchronize(stream)
        cuda_check(err)

        err, = cuda.cuStreamDestroy(stream)
        cuda_check(err)

        return result

    def _viterbi(self, obs, obs_lens, padded_lens, num_parallel):
        num_instances = len(obs_lens)

        # If we run Viterbi on more than one sequence in parallel, we order
        # them by length to reduce the amount of padding. This can
        # significantly reduce the number of kernels we launch, especially if
        # there is a huge difference in the lengths of the observations,
        # thereby improving performance. For the kmer example, this results in
        # a 30% reduction in the number of kernels we launch.
        if num_parallel > 1 and num_parallel < num_instances:
            idxobs = [(i, x, y, z) for (i, x), y, z in zip(enumerate(obs), obs_lens, padded_lens)]
            ordered_idxobs = sorted(idxobs, key=lambda x: x[2])
            permutation, obs, obs_lens, padded_lens, = zip(*ordered_idxobs)
            obs_lens = np.array(obs_lens, dtype=np.int32)
            padded_lens = np.array(padded_lens, dtype=np.int32)

        # Flatten the observations after potentially reordering them based on
        # length.
        obs = np.array(obs).flatten()

        # Load the functions defined in the CUDA code
        viterbi_init = self.load_cuda_function(b"viterbi_init")
        viterbi_init_batch = self.load_cuda_function(b"viterbi_init_batch")
        viterbi_forward = self.load_cuda_function(b"viterbi_forward")
        viterbi_backward = self.load_cuda_function(b"viterbi_backward")

        err, stream = cuda.cuStreamCreate(0)
        cuda_check(err)

        # Allocate observation data on the GPU
        maxlen = np.array(max(padded_lens), dtype=np.int32)
        obs_sz = num_parallel * maxlen * np.dtype(self.obs_type).itemsize
        err, cu_obs = cuda.cuMemAllocAsync(obs_sz, stream)
        cuda_check(err)
        obs_lens_sz = num_parallel * obs_lens.itemsize
        err, cu_obs_lens = cuda.cuMemAllocAsync(obs_lens_sz, stream)
        cuda_check(err)

        # Allocate working data on the GPU
        chi_sz = num_parallel * self.num_states * np.dtype(self.prob_type).itemsize
        err, chi_src = cuda.cuMemAllocAsync(chi_sz, stream)
        cuda_check(err)
        err, chi_dst = cuda.cuMemAllocAsync(chi_sz, stream)
        cuda_check(err)
        zeta_sz = num_parallel * self.batch_size * self.num_states * np.dtype(self.state_type).itemsize
        err, zeta = cuda.cuMemAllocAsync(zeta_sz, stream)
        cuda_check(err)
        err, cu_result = cuda.cuMemAllocAsync(num_parallel * maxlen * np.dtype(self.state_type).itemsize, stream)
        cuda_check(err)

        result = np.zeros((num_instances, maxlen), dtype=self.state_type)

        # Pin the data for the result and the observations to ensure these are
        # pinned, meaning they can be asynchronously copied to/from.
        cuda.cuMemHostRegister(obs.ctypes.data, obs_sz, 0)
        cuda.cuMemHostRegister(result.ctypes.data, num_instances * maxlen * result.itemsize, 0)

        for i in range(0, num_instances, num_parallel):
            # Copy data to the GPU for the current batch
            slicelen = min(i + num_parallel, num_instances) - i
            obs_slice = obs[i * maxlen : (i + slicelen) * maxlen]
            sz = slicelen * maxlen * obs.itemsize
            err, = cuda.cuMemcpyHtoDAsync(cu_obs, obs_slice.ctypes.data, sz, stream)
            cuda_check(err)
            obs_lens_slice = obs_lens[i:i+slicelen]
            sz = slicelen * obs_lens.itemsize
            err, = cuda.cuMemcpyHtoDAsync(cu_obs_lens, obs_lens_slice.ctypes.data, sz, stream)
            cuda_check(err)

            tpb = 256
            xblocks = (self.num_states + tpb - 1) // tpb
            blockdim = (xblocks, slicelen, 1)
            threaddim = (tpb, 1, 1)

            bos = self.batch_size - self.batch_overlap
            slice_maxlen = np.array(max(obs_lens_slice), dtype=np.int32)
            nbatches = int(math.ceil(slice_maxlen / bos))
            for b in range(nbatches):
                t = np.array(b * bos, dtype=np.int32)
                if b == 0:
                    args = [
                        np.array([int(cu_obs)], dtype=np.uint64),
                        maxlen,
                        np.array([int(chi_src)], dtype=np.uint64),
                    ] + self.table_ptrs
                    self.run_kernel(viterbi_init, blockdim, threaddim, stream, args)
                else:
                    args = [
                        np.array([int(cu_obs)], dtype=np.uint64),
                        np.array([int(cu_obs_lens)], dtype=np.uint64),
                        maxlen,
                        np.array([int(cu_result)], dtype=np.uint64),
                        np.array([int(chi_src)], dtype=np.uint64),
                        np.array(t, dtype=np.int32),
                    ] + self.table_ptrs
                    self.run_kernel(viterbi_init_batch, blockdim, threaddim, stream, args)

                for k in range(1, self.batch_size):
                    args = [
                        np.array([int(cu_obs)], dtype=np.uint64),
                        np.array([int(cu_obs_lens)], dtype=np.uint64),
                        maxlen,
                        np.array([int(chi_src)], dtype=np.uint64),
                        np.array([int(chi_dst)], dtype=np.uint64),
                        np.array([int(zeta)], dtype=np.uint64),
                        t,
                        np.array(k, dtype=np.int32),
                    ] + self.table_ptrs
                    self.run_kernel(viterbi_forward, blockdim, threaddim, stream, args)

                backw_blockdim = (slicelen, 1, 1)
                backw_threaddim = (512, 1, 1)
                args = [
                    np.array([int(chi_src)], dtype=np.uint64),
                    np.array([int(zeta)], dtype=np.uint64),
                    np.array([int(cu_result)], dtype=np.uint64),
                    maxlen,
                    t
                ]
                self.run_kernel(viterbi_backward, backw_blockdim, backw_threaddim, stream, args)

            # Copy result from the GPU and add to the result
            sz = slicelen * maxlen * result.itemsize
            err, = cuda.cuMemcpyDtoHAsync(result[i:i+slicelen].ctypes.data, cu_result, sz, stream)
            cuda_check(err)

        # Free up allocated data on the GPU
        err, = cuda.cuMemFreeAsync(cu_obs_lens, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(cu_obs, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(chi_src, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(chi_dst, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(zeta, stream)
        cuda_check(err)
        err, = cuda.cuMemFreeAsync(cu_result, stream)
        cuda_check(err)

        err, = cuda.cuStreamSynchronize(stream)
        cuda_check(err)

        err, = cuda.cuStreamDestroy(stream)
        cuda_check(err)

        # Unpin the CPU data
        cuda.cuMemHostUnregister(obs.ctypes.data)
        cuda.cuMemHostUnregister(result.ctypes.data)

        result = [r[:obs_lens[i]] for i, r in enumerate(result)]

        # If we ran more than one instance in parallel, we restore the original
        # order here.
        if num_parallel > 1 and num_parallel < num_instances:
            result_tmp = result.copy()
            for i, p in enumerate(permutation):
                result_tmp[p] = result[i]
            return result_tmp

        return result

    def pad_signals(self, signals, lens):
        n = max(lens)
        ps = np.zeros((len(signals), n), dtype=self.obs_type)
        for i, s in enumerate(signals):
            ps[i][:len(s)] = s
        return ps

    def viterbi(self, signals, num_parallel=1):
        bos = self.batch_size - self.batch_overlap
        lens = np.array([len(x) for x in signals], dtype=np.int32)
        plens = np.array([(n + bos - 1) // bos * bos + self.batch_overlap for n in lens], dtype=np.int32)
        padded_obs = self.pad_signals(signals, plens)
        return self._viterbi(padded_obs, lens, plens, num_parallel)

    def forward(self, signals):
        lens = np.array([len(x) for x in signals])
        padded_signals = self.pad_signals(signals, lens)
        return self._forward(padded_signals.flatten(), lens)

# Generated code beyond this point

    def __init__(self, args):
        self.num_states = 262144
        self.batch_size = 1024
        self.batch_overlap = 128
        self.state_type = np.uint32
        self.prob_type = np.float32
        self.obs_type = np.uint8
        self.compile_cuda("hmm.cu")
        self.copy_args(args)

    def __del__(self):
        if hasattr(self, "tables_trans1"):
            err, = cuda.cuMemFree(self.table_trans1)
            cuda_check(err)
        if hasattr(self, "table_trans2"):
            err, = cuda.cuMemFree(self.table_trans2)
            cuda_check(err)
        if hasattr(self, "table_output_prob"):
            err, = cuda.cuMemFree(self.table_output_prob)
            cuda_check(err)
        if hasattr(self, "table_initial_prob"):
            err, = cuda.cuMemFree(self.table_initial_prob)
            cuda_check(err)
        if hasattr(self, "module"):
            err, = cuda.cuModuleUnload(self.module)
            cuda_check(err)
        if hasattr(self, "ctx"):
            err, = cuda.cuCtxDestroy(self.ctx)
            cuda_check(err)

    def copy_args(self, args):
        # Reshape the arrays for better locality. This could either be
        # performed automatically by the compiler or manually by us.
        args['trans1'] = args['trans1'].reshape(4,4,4,4,4,4,4,4).transpose(1,2,3,4,5,6,7,0).flatten()
        args['outputProb'] = args['outputProb'].reshape(4,4,4,4,4,4,4,101).transpose(7,0,1,2,3,4,5,6).flatten()
        args['initialProb'] = args['initialProb'].flatten()

        self.table_initial_prob = self.copy_to_gpu(np.array(args["initialProb"], dtype=self.prob_type), 0)
        self.table_output_prob = self.copy_to_gpu(np.array(args["outputProb"], dtype=self.prob_type), 0)
        self.table_trans1 = self.copy_to_gpu(np.array(args["trans1"], dtype=self.prob_type), 0)
        self.table_trans2 = self.copy_to_gpu(np.array(args["trans2"], dtype=self.prob_type), 0)
        self.table_gamma = np.array(args["gamma"], dtype=self.prob_type)
        # NOTE: the compiler needs to figure out that it can pre-compute the
        # gamma inverse instead of repeatedly recomputing it, which seems to be
        # rather inefficient.
        self.table_gamma_inv = np.array(np.log(1.0 - np.exp(args["gamma"])), dtype=self.prob_type)
        self.table_ptrs = [
            np.array([int(self.table_initial_prob)], dtype=np.uint64),
            np.array([int(self.table_output_prob)], dtype=np.uint64),
            np.array([int(self.table_trans1)], dtype=np.uint64),
            np.array([int(self.table_trans2)], dtype=np.uint64),
            self.table_gamma,
            self.table_gamma_inv
        ]
