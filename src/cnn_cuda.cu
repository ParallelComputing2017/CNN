/**
 * calculate pi
 */

#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <vector>
#include <algorithm>   // std::min

#include "types.h"
#include "CNN/layer_t.h"

/**
 * CUDA Kernel Device code
 *
 */
/*****************************************************************************/

__global__ void training2(case_t *d_cases, long int batchSize, layer_t *d_slaves, long slaves_count, long layer_count) {

	int index = (blockDim.x * blockIdx.x) + threadIdx.x;

	printf("Index: %i, Cases: %lu, batchSize: %i, ", index, sizeof(d_cases[0]),
			batchSize);
	printf("sizeof(d_slaves[0]): %lu, slaves_count: %i, layer_count: %i \t", sizeof(d_slaves[0]),
			slaves_count, layer_count);

	__syncthreads();

}

/******************************************************************************
 * Host main routine
 */
std::vector<std::vector<layer_t*>> cuda_training(int maxBlocks,
		std::vector<case_t> cases, int batchSize,
		std::vector<std::vector<layer_t*>> slaves) {

	int blocksPerGrid, threadsPerBlock, size;
	int totalThreads;
	case_t *h_cases, *d_cases;

	// Get device info
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	blocksPerGrid = std::min(deviceProp.multiProcessorCount, maxBlocks);
	int cudaCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = deviceProp.maxThreadsPerBlock;
	totalThreads = blocksPerGrid * threadsPerBlock;

	cudaError_t err = cudaSuccess;

	h_cases = &cases[0];
	size = sizeof(case_t) * cases.size();

	// TODO remove
	printf("sizeof(h_cases[0]) * cases.size() = %lu * %lu \n",
			sizeof(h_cases[0]), cases.size());

	if (h_cases == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// copy vector to array
	copy(cases.begin(), cases.end(), h_cases);

	err = cudaMalloc((void **) &d_cases, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_cases, h_cases, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy slaves to device

	long slaves_count = slaves.size();
	long layer_count = slaves.at(0).size();

	layer_t *h_slaves, *d_slaves;

	h_slaves = (new layer_t[slaves_count*layer_count]);

	for (int s = 0; s < slaves_count; s++) {
		copy(slaves[s].begin(), slaves[s].end(), h_slaves);
		h_slaves += slaves[s].size();

	}

	int slaves_mem_size = sizeof(case_t) * slaves_count * layer_count;

	err = cudaMalloc((void **) &d_slaves, slaves_mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_slaves, h_slaves, slaves_mem_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Lanzar KERNEL

	printf("CUDA kernel launch with %d blocks of %d threads. Total: %i\n",
			blocksPerGrid, threadsPerBlock, totalThreads);

	training2<<<blocksPerGrid, threadsPerBlock>>>(d_cases, batchSize, d_slaves, slaves_count, layer_count);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(h_cases, d_cases, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr,
				"Failed to copy vector C from device to host (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_cases);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory

	//free(h_cases);
	err = cudaDeviceReset();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	return slaves;
}

