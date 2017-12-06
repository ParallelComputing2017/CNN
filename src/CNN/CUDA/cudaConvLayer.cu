/*
 * cudaConvLayer.cu
 *
 *  Created on: 12/11/2017
 *      Author: sebas
 */

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <math.h>

#include "cudaConvLayer.h"
#include "cudaTensor.cuh"

__device__ point_t map_to_input(int stride, point_t out, int z) {
	out.x *= stride;
	out.y *= stride;
	out.z = z;
	return out;
}

__global__ void activationKernel(tensor_t<float> *in, tensor_t<float> **filters,
		int *stride, tensor_t<float> *out) {

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int filterIdx = (blockIdx.z * blockDim.z) + threadIdx.z;

	if (x < out->size.x && y < out->size.y) {

		point_t p_out = (point_t ) { x, y, 0 };
		point_t mapped = map_to_input((*stride), p_out, 0);
		float sum = 0;

		tensor_t<float> *kernel = filters[filterIdx];

		for (int i = 0; i < kernel->size.x; i++) {
			for (int j = 0; j < kernel->size.y; j++) {
				for (int z = 0; z < in->size.z; z++) {

					float f = cudaTensor::get(kernel, i, j, z);
					float v = cudaTensor::get(in, mapped.x + i, mapped.y + j,
							z);
					sum += f * v;
				}
			}
		}
		cudaTensor::set(out, x, y, filterIdx, sum);

	}

}

void threadCalculator(const int &requiredThreads, cudaDeviceProp &deviceProp,
		int &blocksPerGrid, int &threadsPerBlock) {
	// calc the threads per block value
	threadsPerBlock = ceil((float) requiredThreads / blocksPerGrid);

	// If the value exess the max, then fixed
	if (threadsPerBlock > blocksPerGrid * deviceProp.maxThreadsPerBlock) {
		blocksPerGrid = ceil(
				(float) threadsPerBlock / deviceProp.maxThreadsPerBlock);
		threadsPerBlock = ceil((float) threadsPerBlock / blocksPerGrid);
	}
}

void CudaConvLayer::activate(tensor_t<float>& in) {

	this->in = in;

	// Get device info
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	int xThreads = out.size.x;
	int yThreads = out.size.y;
	int zThreads = filters.size();

	int xblocks = 5;
	int yblocks = 2;
	int zblocks = 1;

	threadCalculator(out.size.x, deviceProp, xblocks, xThreads);
	threadCalculator(out.size.y, deviceProp, yblocks, yThreads);
	threadCalculator(filters.size(), deviceProp, zblocks, zThreads);

	Logger::debug("Convolution, required threads: %i", xThreads);
	Logger::debug("Multiprocessors: %i", deviceProp.multiProcessorCount);
	Logger::debug("Max threads per block: %i", deviceProp.maxThreadsPerBlock);

	dim3 blocksPerGrid(xblocks, yblocks, zblocks);
	dim3 threadsPerBlock(xThreads, yThreads, zThreads);

	// IN

	cudaTensor inTensor(&in);
	inTensor.hostToDevice();

	// Out

	cudaTensor outTensor(&out);
	outTensor.hostToDevice();

	// Filters (array of pointers)

	tensor_t<float> **d_filters;
	long filter_mem_size = filters.size() * sizeof(tensor_t<float>*);

	cudaMalloc((void ***) &d_filters, filter_mem_size);
	cudaCheckError()

	for (int k = 0; k < filters.size(); k++) {
		// Kernel
		cudaTensor kernelTensor(&filters[k]);
		kernelTensor.hostToDevice();

		// cudaMemcpy needs a host pointer
		tensor_t<float> *d_kernel = kernelTensor.devicePointer();
		cudaMemcpy(&(d_filters[k]), &d_kernel, sizeof(d_filters[k]),
				cudaMemcpyHostToDevice);
		cudaCheckError()
	}

	// Stride

	int iStride = (int) stride;
	int* d_stride;
	long stride_mem_size = sizeof(int);

	cudaMalloc((void **) &d_stride, stride_mem_size);
	cudaCheckError()

	cudaMemcpy(d_stride, &iStride, stride_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError()

		// Launch KERNEL

	Logger::debug(
			"CUDA kernel launch with (%d, %d, %d) blocks of (%d, %d, %d) threads",
			blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z,
			threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

	activationKernel<<<blocksPerGrid, threadsPerBlock>>>(
			inTensor.devicePointer(), d_filters, d_stride,
			outTensor.devicePointer());
	cudaCheckError()

	cudaDeviceSynchronize();
	cudaCheckError()

		// get out

	outTensor.deviceToHost();

	// Free device memory
	inTensor.deviceFree();
	outTensor.deviceFree();

	cudaFree(d_filters);
	cudaCheckError()

	cudaFree(d_stride);
	cudaCheckError()

	cudaDeviceReset();
	cudaCheckError()

		// Free host memory

		// TODO
		//print_tensor(out);

		// TODO
		//exit (EXIT_SUCCESS);
}

