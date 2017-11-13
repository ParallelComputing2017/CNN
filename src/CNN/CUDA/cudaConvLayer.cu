/*
 * cudaConvLayer.cu
 *
 *  Created on: 12/11/2017
 *      Author: sebas
 */

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "cudaConvLayer.h"
#include "cudaTensor.cuh"

__device__ point_t map_to_input(int stride, point_t out, int z) {
	out.x *= stride;
	out.y *= stride;
	out.z = z;
	return out;
}

__global__ void convolutionKernel(tensor_t<float> *in, tensor_t<float> *kernel,
		int *filterIdx, int *stride, tensor_t<float> *out) {

	for (int x = 0; x < out->size.x; x++) {
		for (int y = 0; y < out->size.y; y++) {
			point_t p_out = (point_t ) { x, y, 0 };
			point_t mapped = map_to_input((*stride), p_out, 0);
			float sum = 0;
			for (int i = 0; i < kernel->size.x; i++) {
				for (int j = 0; j < kernel->size.y; j++) {
					for (int z = 0; z < in->size.z; z++) {
						float f = cudaTensor::get(kernel, i, j, z);
						float v = cudaTensor::get(in,mapped.x + i, mapped.y + j, z);
						sum += f * v;
					}
				}
			}
			cudaTensor::set(out, x, y, *filterIdx, sum);
		}
	}
}

void cudaConvolution(tensor_t<float> *in, tensor_t<float> *kernel,
		int *filterIdx, int *stride, tensor_t<float> *out) {

	int blocksPerGrid, threadsPerBlock;
	int totalThreads;

	// Get device info
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	int requiredThreads = 1;

	blocksPerGrid = std::min(deviceProp.multiProcessorCount, 1);

	threadsPerBlock = std::min(deviceProp.maxThreadsPerBlock,
			requiredThreads / blocksPerGrid);

	totalThreads = blocksPerGrid * threadsPerBlock;

	// IN

	cudaTensor inTensor(in);
	inTensor.hostToDevice();

	// Kernel
	cudaTensor kernelTensor(kernel);
	kernelTensor.hostToDevice();

	// Out

	cudaTensor outTensor(out);
	outTensor.hostToDevice();

	// Filter

	int* d_filter;
	long filter_mem_size = sizeof(int);

	cudaMalloc((void **) &d_filter, filter_mem_size);
	cudaCheckError();

	cudaMemcpy(d_filter, filterIdx, filter_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError();

	// Stride

	int* d_stride;
	long stride_mem_size = sizeof(int);

	cudaMalloc((void **) &d_stride, stride_mem_size);
	cudaCheckError();

	cudaMemcpy(d_stride, stride, stride_mem_size, cudaMemcpyHostToDevice);
	cudaCheckError();

	// Launch KERNEL

	Logger::debug(
			"CUDA kernel launch with %d blocks of %d threads. Total: %i\n",
			blocksPerGrid, threadsPerBlock, totalThreads);

	convolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(
			inTensor.devicePointer(), kernelTensor.devicePointer(), d_filter,
			d_stride, outTensor.devicePointer());

	cudaDeviceSynchronize();
	cudaCheckError();

	// get out

	outTensor.deviceToHost();

	// Free device memory

	inTensor.deviceFree();
	kernelTensor.deviceFree();
	outTensor.deviceFree();

	cudaFree(d_filter);
	cudaCheckError();

	cudaFree(d_stride);
	cudaCheckError();

	cudaDeviceReset();
	cudaCheckError();

	// Free host memory
}

void CudaConvLayer::activate(tensor_t<float>& in) {

	this->in = in;

	for (int filter = 0; filter < filters.size(); filter++) {
		tensor_t<float> *kernel = &filters[filter];

		tensor_t<float> *pIn = &in;
		int *pFilter = &filter;
		int iStride = (int) stride;
		int *pStride = &iStride;
		tensor_t<float> *pOut = &out;

		cudaConvolution(pIn, kernel, pFilter, pStride, pOut);

		// TODO
		//exit(EXIT_SUCCESS);
	}
}

