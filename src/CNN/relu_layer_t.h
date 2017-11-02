#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
class relu_layer_t: public layer_t {

public:
	relu_layer_t(tdsize in_size) :
			layer_t(in_size, in_size) {
		type = layer_type::relu;
	}

	void activate(tensor_t<float>& in) {
		this->in = in;
		activate();
	}

	void activate() {
		for (int i = 0; i < in.getSize().x; i++)
			for (int j = 0; j < in.getSize().y; j++)
				for (int z = 0; z < in.getSize().z; z++) {
					float v = in(i, j, z);
					if (v < 0)
						v = 0;
					out(i, j, z) = v;
				}

	}

	void fix_weights() {

	}

	void calc_grads(tensor_t<float>& grad_next_layer) {
		for (int i = 0; i < in.getSize().x; i++)
			for (int j = 0; j < in.getSize().y; j++)
				for (int z = 0; z < in.getSize().z; z++) {
					grads_in(i, j, z) =
							(in(i, j, z) < 0) ?
									(0) : (1 * grad_next_layer(i, j, z));
				}
	}
};
#pragma pack(pop)
