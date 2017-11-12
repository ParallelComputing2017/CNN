/*
 * cudaConvLayer.cuh
 *
 *  Created on: 12/11/2017
 *      Author: sebas
 */

#pragma once

#include "../conv_layer_t.h"

class CudaConvLayer: public conv_layer_t {

public:

	CudaConvLayer(uint16_t stride, uint16_t extend_filter,
			uint16_t number_filters, tdsize in_size) :
			conv_layer_t(stride, extend_filter, number_filters, in_size) {

	}

	void activate(tensor_t<float>& in);

};

