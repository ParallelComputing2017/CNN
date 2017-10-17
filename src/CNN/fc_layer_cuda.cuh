/*
 * fc_layer_cuda.cuh
 *
 *  Created on: 17/10/2017
 *      Author: sebas
 */

#ifndef FC_LAYER_CUDA_CUH_
#define FC_LAYER_CUDA_CUH_

struct fc_layer_cuda_t {

	layer_type type = layer_type::fc_cuda;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_cuda_t(point_t in_size, int out_size);

	void activate(tensor_t<float>& in);

	int map(point_t d);

	void activate();

	void fix_weights();

	void calc_grads(tensor_t<float>& grad_next_layer);

	void setWeights(tensor_t<float> newWeights);

	void updateWeights(tensor_t<float> newWeights);
};



#endif /* FC_LAYER_CUDA_CUH_ */
