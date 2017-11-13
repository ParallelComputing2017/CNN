/*
 * cudaConvLayer.cu
 *
 *  Created on: 12/11/2017
 *      Author: sebas
 */

#include "cudaConvLayer.h"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

point_t map_to_input(int stride, point_t out, int z) {
	out.x *= stride;
	out.y *= stride;
	out.z = z;
	return out;
}

void convolution(tensor_t<float> *in, tensor_t<float> *kernel, int *filterIdx, int *stride,
		tensor_t<float> *out) {

	for (int x = 0; x < out->getSize().x; x++) {
		for (int y = 0; y < out->getSize().y; y++) {
			point_t p_out = (point_t ) { x, y, 0 };
			point_t mapped = map_to_input((*stride), p_out, 0);
			float sum = 0;
			for (int i = 0; i < kernel->size.x; i++) {
				for (int j = 0; j < kernel->size.y; j++) {
					for (int z = 0; z < in->getSize().z; z++) {
						float f = kernel->get(i, j, z);
						float v = in->get(mapped.x + i, mapped.y + j, z);
						sum += f * v;
					}
				}
			}
			out->set(sum, x, y, *filterIdx);
		}
	}
}

void CudaConvLayer::activate(tensor_t<float>& in) {

	this->in = in;

	for (int filter = 0; filter < filters.size(); filter++) {
		tensor_t<float> *kernel = &filters[filter];

		tensor_t<float> *pIn = &in;
		int *pFilter = &filter;
		int iStride = (int)stride;
		int *pStride = &iStride;
		tensor_t<float> *pOut = &out;

		convolution(pIn, kernel, pFilter, pStride, pOut);
	}
}

