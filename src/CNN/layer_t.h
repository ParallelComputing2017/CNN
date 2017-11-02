#pragma once

#include "tensor_t.h"
#include "types.h"

#pragma pack(push, 1)

class layer_t {

protected:


public:
	layer_type type = layer_type::null;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;

	layer_t(tdsize in_size, int out_size) :
			in(in_size.x, in_size.y, in_size.z), out(out_size, 1, 1), grads_in(
					in_size.x, in_size.y, in_size.z) {

	}

	virtual void activate(tensor_t<float>& in) = 0;

	virtual ~layer_t() = default;

};
#pragma pack(pop)
