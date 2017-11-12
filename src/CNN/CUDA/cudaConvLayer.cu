/*
 * cudaConvLayer.cu
 *
 *  Created on: 12/11/2017
 *      Author: sebas
 */

#include "cudaConvLayer.h"

void CudaConvLayer::activate(tensor_t<float>& in) {

	this->in = in;
	for (int filter = 0; filter < filters.size(); filter++) {
		tensor_t<float>& filter_data = filters[filter];
		for (int x = 0; x < out.getSize().x; x++) {
			for (int y = 0; y < out.getSize().y; y++) {
				point_t p_out = (point_t ) { x, y, 0 };
				point_t mapped = map_to_input(p_out, 0);
				float sum = 0;
				for (int i = 0; i < extend_filter; i++) {
					for (int j = 0; j < extend_filter; j++) {
						for (int z = 0; z < in.getSize().z; z++) {
							float f = filter_data(i, j, z);
							float v = in(mapped.x + i, mapped.y + j, z);
							sum += f * v;
						}
					}
				}
				out(x, y, filter) = sum;
			}
		}
	}
}

