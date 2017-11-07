#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <vector>
#include <algorithm>   // std::min

#include <math.h>
#include <float.h>
#include <string.h>

#include "gradient_t.h"
#include "layer_t.h"
#include "optimization_method.h"

#include "fc_layer_cuda.h"

#include "CUDA/utils.cuh"

float cudaActivate(tensor_t<float> in, tensor_t<float> weights, int n);

__host__ void fc_layer_cuda_t::activate(tensor_t<float>& in) {
	this->in = in;

	for (int n = 0; n < out.getSize().x; n++) {

		float inputv = cudaActivate(this->in, this->weights, n);

		input[n] = inputv;

		out(n, 0, 0) = fc_layer_t::activator_function(inputv);
	}

	// TODO
	exit(EXIT_SUCCESS);

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

__device__ float& get(tensor_t<float> *t, int _x, int _y, int _z) {
	assert(_x >= 0 && _y >= 0 && _z >= 0);
	assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

	return t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x];
}

__device__ void set(tensor_t<float> *t, int _x, int _y, int _z, float value) {
	assert(_x >= 0 && _y >= 0 && _z >= 0);
	assert(_x < t->size.x && _y < t->size.y && _z < t->size.z);

	t->data[_z * (t->size.x * t->size.y) + _y * (t->size.x) + _x] = value;
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

	float inputv = get(d_in, i, j, k) * get(d_weights, m, *d_n, 0);

	//printf("inputv: %f \n", inputv);

	*(d_input + index) = inputv;

}

/******************************************************************************
 * Host main routine
 */
float deviceReduce(float *in, float* out, int N) {
	int threads = 512;
	int blocks = min((N + threads - 1) / threads, 1024);

	float *d_in, *d_out;
	int* d_N;

	int in_mem_size = sizeof(float) * N;
	cudaMalloc((void **) &d_in, in_mem_size);
	cudaCheckError();

	cudaMemcpy(d_in, in, in_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError();

	// OUT
	int out_mem_size = sizeof(int) * 1024;
	cudaMalloc((void **) &d_out, out_mem_size);
	cudaCheckError();

	cudaMemcpy(d_out, out, out_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError();

	int n_mem_size = sizeof(int);

	cudaMalloc((void **) &d_N, n_mem_size);
	cudaCheckError();

	cudaMemcpy(d_N, &N, n_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError();

	deviceReduceKernel<<<blocks, threads>>>(d_in, d_out, N);
	deviceReduceKernel<<<1, 1024>>>(d_out, d_out, blocks);

	cudaDeviceSynchronize();
	cudaCheckError();

	float sum;
	cudaMemcpy(&sum, d_out, out_mem_size, cudaMemcpyDeviceToHost);
	cudaCheckError();

	Logger::debug("Sum reduce: %f \n", sum);

	return sum;

}

float cudaActivate(tensor_t<float> in, tensor_t<float> weights, int n) {

	int blocksPerGrid, threadsPerBlock;
	int totalThreads;
	tensor_t<float> *h_in, *d_in;
	tensor_t<float> *h_weights, *d_weights;

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

	h_in = &in;
	int in_mem_size = sizeof(in);

	if (h_in == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void **) &d_in, in_mem_size);
	//cudaCheckError("cudaMalloc IN tensor");

	cudaMemcpy(d_in, h_in, in_mem_size, cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcopy to device IN tensor");

	// IN DATA

	float *d_in_data;
	long in_data_size = sizeof(*d_in_data) * in.getSize().x * in.getSize().y
			* in.getSize().z;

	//printf("sizeof(in)= %lu , in_data_size = %lu \n", sizeof(in), in_data_size);

	cudaMalloc((void **) &d_in_data, in_data_size);
	//cudaCheckError("cudaMalloc IN tensor data");

	cudaMemcpy(d_in_data, in.data, in_data_size, cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy to device IN tensor data");

	cudaMemcpy(&(d_in->data), &d_in_data, sizeof(d_in->data),
			cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy Binding pointers of IN tensor data");

	// Copy weights

	h_weights = &weights;
	long weights_mem_size = sizeof(weights);

	if (h_weights == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void **) &d_weights, weights_mem_size);
	//cudaCheckError("cudaMalloc Weights tensor");

	cudaMemcpy(d_weights, h_weights, weights_mem_size, cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy to device Weights tensor");

	// Weights DATA

	float *d_weights_data;
	long weights_data_size = sizeof(*d_weights_data) * weights.getSize().x
			* weights.getSize().y * weights.getSize().z;

	cudaMalloc((void **) &d_weights_data, weights_data_size);
	//cudaCheckError("cudaMalloc Weights tensor data");

	cudaMemcpy(d_weights_data, weights.data, weights_data_size,
			cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy to device Weights tensor data");

	cudaMemcpy(&(d_weights->data), &d_weights_data, sizeof(d_weights->data),
			cudaMemcpyHostToDevice);
	//cudaCheckError("cudaMemcpy Binding pointers of Weights tensor data");

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

	activate_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_weights, d_n,
			d_input);

	cudaDeviceSynchronize();
	//cudaCheckError("Launch kernel");

	// get input array

	cudaMemcpy(h_input, d_input, input_mem_size, cudaMemcpyDeviceToHost);
	//cudaCheckError("cudaMemcpy to host Input array");

	// Free device memory
	cudaFree(d_in);
	//cudaCheckError("cudaFree IN tensor");

	cudaFree(d_weights);
	//cudaCheckError("cudaFree Weights tensor");

	cudaFree(d_n);
	//cudaCheckError("cudaFree N value");

	cudaFree(d_input);
	//cudaCheckError("cudaFree Input array");

	cudaDeviceReset();
	//cudaCheckError("cudaDeviceReset");

	//

	float sum = 0.0, sumR = 0.0;

	for (int i = 0; i < requiredThreads; i++) {
		sum += h_input[i];
	}

	float* h_out = (float*) malloc(input_mem_size);

	if (h_out == NULL) {
		fprintf(stderr, "Failed to allocate INPUT vector!\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < requiredThreads; i++) {
		h_out[i] = 0.0;
	}

	sumR = deviceReduce(h_input, h_out, requiredThreads);

	// Free host memory
	free(h_input);

	Logger::debug("sum: %f == %f \n", sum, sumR);

	return sum;
}

