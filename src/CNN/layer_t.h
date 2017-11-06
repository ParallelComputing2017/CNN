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

	layer_t(tdsize in_size, tdsize out_size) :
			in(in_size), out(out_size), grads_in(in_size) {

	}

	virtual void activate(tensor_t<float>& in) = 0;

	virtual void calc_grads(tensor_t<float>& grad_next_layer) = 0;

	virtual void fix_weights() = 0;

	virtual ~layer_t() = default;

};
#pragma pack(pop)
