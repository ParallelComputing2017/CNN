#pragma once
#include <math.h>
#include <float.h>
#include <string.h>

#include "gradient_t.h"
#include "layer_t.h"
#include "optimization_method.h"

#pragma pack(push, 1)

class fc_layer_t: public layer_t {

public:
	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_t(tdsize in_size, int out_size) :
			layer_t(in_size, (tdsize){out_size, 1, 1}), weights(
					in_size.x * in_size.y * in_size.z, out_size, 1) {

		type = layer_type::fc;

		input = std::vector<float>(out_size);
		gradients = std::vector<gradient_t>(out_size);

		int maxval = in_size.x * in_size.y * in_size.z;

		for (int i = 0; i < out_size; i++)
			for (int h = 0; h < in_size.x * in_size.y * in_size.z; h++)
				weights(h, i, 0) = 2.19722f / maxval * rand() / float(RAND_MAX);
		// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
	}

	~fc_layer_t() {

	}

	float activator_function(float x) {
		//return tanhf( x );
		float sig = 1.0f / (1.0f + exp(-x));
		return sig;
	}

	float activator_derivative(float x) {
		//float t = tanhf( x );
		//return 1 - t * t;
		float sig = 1.0f / (1.0f + exp(-x));
		return sig * (1 - sig);
	}

	void activate(tensor_t<float>& in) {
		this->in = in;
		activate();
	}

	int map(point_t d) {
		return d.z * (in.getSize().x * in.getSize().y) + d.y * (in.getSize().x)
				+ d.x;
	}

	void activate() {
		for (int n = 0; n < out.getSize().x; n++) {
			float inputv = 0;

			for (int i = 0; i < in.getSize().x; i++)
				for (int j = 0; j < in.getSize().y; j++)
					for (int z = 0; z < in.getSize().z; z++) {
						int m = map( { i, j, z });
						inputv += in(i, j, z) * weights(m, n, 0);
					}

			input[n] = inputv;

			out(n, 0, 0) = activator_function(inputv);
		}
	}

	void fix_weights() {
		for (int n = 0; n < out.getSize().x; n++) {
			gradient_t& grad = gradients[n];
			for (int i = 0; i < in.getSize().x; i++)
				for (int j = 0; j < in.getSize().y; j++)
					for (int z = 0; z < in.getSize().z; z++) {
						int m = map( { i, j, z });
						float& w = weights(m, n, 0);
						w = update_weight(w, grad, in(i, j, z));
					}

			update_gradient(grad);
		}
	}

	void calc_grads(tensor_t<float>& grad_next_layer) {
		memset(grads_in.data, 0,
				grads_in.getSize().x * grads_in.getSize().y
						* grads_in.getSize().z * sizeof(float));
		for (int n = 0; n < out.getSize().x; n++) {
			gradient_t& grad = gradients[n];
			grad.grad = grad_next_layer(n, 0, 0)
					* activator_derivative(input[n]);

			for (int i = 0; i < in.getSize().x; i++)
				for (int j = 0; j < in.getSize().y; j++)
					for (int z = 0; z < in.getSize().z; z++) {
						int m = map( { i, j, z });
						grads_in(i, j, z) += grad.grad * weights(m, n, 0);
					}
		}
	}

	void setWeights(tensor_t<float> newWeights) {
		weights = newWeights;
	}
	void updateWeights(tensor_t<float> newWeights) {
		weights = (weights + newWeights) / 2;
	}
};
#pragma pack(pop)
