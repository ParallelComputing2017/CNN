#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>   // std::min

#include <math.h>
#include <float.h>
#include <string>

#include "fc_layer_cuda.h"

#include "utils.cuh"
#include "cudaTensor.cuh"

float cudaActivate(tensor_t<float> in, tensor_t<float> weights, int n);

__host__ void fc_layer_cuda_t::activate(tensor_t<float>& in) {
	this->in = in;

	for (int n = 0; n < out.getSize().x; n++) {

		float inputv = cudaActivate(this->in, this->weights, n);

		input[n] = inputv;

		out(n, 0, 0) = fc_layer_t::activator_function(inputv);
	}

	// TODO
	//exit(EXIT_SUCCESS);

}

/**
 * CUDA Kernel Device code
 *
 */
/*****************************************************************************/
__device__ float warpReduceSum(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val += __shfl_down(val, offset);
	}
	return val;
}

__device__ float blockReduceSum(float val) {
	static __shared__ float shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);

	//write reduced value to shared memory
	if (lane == 0)
		shared[wid] = val;
	__syncthreads();

	//ensure we only grab a value from shared memory if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : float(0);
	if (wid == 0)
		val = warpReduceSum(val);

	return val;
}

__global__ void deviceReduceKernel(float *in, float* out, int N) {
	float sum = float(0);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
			i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0)
		out[blockIdx.x] = sum;
}

__global__ void activate_cuda(tensor_t<float> *d_in, tensor_t<float> *d_weights,
		int *d_n, float* d_input) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int i = ((int) index / (d_in->size.y * d_in->size.z)) % d_in->size.x;
	int j = ((int) index / d_in->size.z) % d_in->size.y;
	int k = index % d_in->size.z;

	/*printf("index: %i  (x, y, z)=(%i, %i, %i)  (i, j, k)=(%i, %i, %i) \n",
	 index, d_in->size.x, d_in->size.y, d_in->size.z, i, j, k);*/

	// map
	int m = k * (d_in->size.x * d_in->size.y) + j * (d_in->size.x) + i;

	float inputv = cudaTensor::get(d_in, i, j, k) * cudaTensor::get(d_weights, m, *d_n, 0);

	//printf("inputv: %f \n", inputv);

	*(d_input + index) = inputv;

}

/******************************************************************************
 * Host main routine
 */
void deviceReduce(float *in, float* out, int N) {
	int threads = 512;
	int blocks = min((N + threads - 1) / threads, 1024);

	float *d_in, *d_out;
	int* d_N;

	int in_mem_size = sizeof(float) * N;
	cudaMalloc((void **) &d_in, in_mem_size);
	cudaCheckError()
	;

	cudaMemcpy(d_in, in, in_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError()
	;

	// OUT
	int out_mem_size = sizeof(int) * N;
	cudaMalloc((void **) &d_out, out_mem_size);
	cudaCheckError()
	;

	cudaMemcpy(d_out, out, out_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError()
	;

	int n_mem_size = sizeof(int);

	cudaMalloc((void **) &d_N, n_mem_size);
	cudaCheckError()
	;

	cudaMemcpy(d_N, &N, n_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError()
	;

	deviceReduceKernel<<<blocks, threads>>>(d_in, d_out, N);
	deviceReduceKernel<<<1, 1024>>>(d_out, d_out, blocks);

	cudaDeviceSynchronize();
	cudaCheckError()
	;

	cudaMemcpy(out, d_out, out_mem_size, cudaMemcpyDeviceToHost);
	cudaCheckError()
	;

	cudaFree(d_in);
	cudaCheckError()
	;

	cudaFree(d_out);
	cudaCheckError()
	;

	cudaDeviceReset();
	cudaCheckError()
	;

	Logger::debug("Sum2 reduce: %f \n", out[0]);

}

float cudaActivate(tensor_t<float> in, tensor_t<float> weights, int n) {

	int blocksPerGrid, threadsPerBlock;
	int totalThreads;

	// Get device info
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	int requiredThreads = in.size.x * in.size.y * in.size.z;

	blocksPerGrid = std::min(deviceProp.multiProcessorCount, 2);

	threadsPerBlock = std::min(deviceProp.maxThreadsPerBlock,
			requiredThreads / blocksPerGrid);

	totalThreads = blocksPerGrid * threadsPerBlock;

	// IN
	cudaTensor inTensor(&in);

	inTensor.hostToDevice();

	// Weights

	cudaTensor weightsTensor(&weights);

	weightsTensor.hostToDevice();

	// N

	int* d_n;
	int* h_n = &n;
	long n_mem_size = sizeof(int);

	cudaMalloc((void **) &d_n, n_mem_size);
	//cudaCheckError("cudaMalloc N value");

	cudaMemcpy(d_n, h_n, n_mem_size, cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy to device N value");

	// INPUT array

	long input_mem_size = sizeof(float) * requiredThreads;
	float* d_input;
	float* h_input = (float*) malloc(input_mem_size);

	if (h_input == NULL) {
		fprintf(stderr, "Failed to allocate INPUT vector!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < requiredThreads; i++) {
		h_input[i] = 0.0;
	}

	cudaMalloc((void **) &d_input, input_mem_size);
	//cudaCheckError("cudaMalloc N value");

	cudaMemcpy(d_input, h_input, input_mem_size, cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy to device N value");

	// Launch KERNEL

	Logger::debug(
			"CUDA kernel launch with %d blocks of %d threads. Total: %i\n",
			blocksPerGrid, threadsPerBlock, totalThreads);

	activate_cuda<<<blocksPerGrid, threadsPerBlock>>>(inTensor.devicePointer(),
			weightsTensor.devicePointer(), d_n, d_input);

	cudaDeviceSynchronize();
	//cudaCheckError("Launch kernel");

	// get input array

	cudaMemcpy(h_input, d_input, input_mem_size, cudaMemcpyDeviceToHost);
	//cudaCheckError("cudaMemcpy to host Input array");

	// Free device memory
	inTensor.deviceFree();

	weightsTensor.deviceFree();

	cudaFree(d_n);
	//cudaCheckError("cudaFree N value");

	cudaFree(d_input);
	//cudaCheckError("cudaFree Input array");

	cudaDeviceReset();
	//cudaCheckError("cudaDeviceReset");

	//

	float sumR = 0.0;

	float* h_out = (float*) malloc(input_mem_size);

	if (h_out == NULL) {
		fprintf(stderr, "Failed to allocate INPUT vector!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < requiredThreads; i++) {
		h_out[i] = 0.0;
	}

	deviceReduce(h_input, h_out, requiredThreads);
	sumR = h_out[0];

	// Free host memory
	free(h_input);

	Logger::debug("sum: %f \n", sumR);

	return sumR;
}

