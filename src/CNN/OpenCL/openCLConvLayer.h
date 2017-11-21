

#pragma once

#include "../conv_layer_t.h"

class OpenCLConvLayer: public conv_layer_t {

public:

	OpenCLConvLayer(uint16_t stride, uint16_t extend_filter,
			uint16_t number_filters, tdsize in_size) :
			conv_layer_t(stride, extend_filter, number_filters, in_size) {

	}

	void activate(tensor_t<float>& in);

};

