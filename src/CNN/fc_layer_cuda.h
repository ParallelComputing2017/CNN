/*
 * fc_layer_cuda.cuh
 *
 *  Created on: 17/10/2017
 *      Author: sebas
 */

#ifndef FC_LAYER_CUDA_H_
#define FC_LAYER_CUDA_H_

#include "fc_layer.h"

class fc_layer_cuda_t{

private:

	fc_layer_t* fc_layer;

public:

	layer_type type = layer_type::fc_cuda;

	fc_layer_cuda_t(point_t in_size, int out_size){
		fc_layer = new fc_layer_t(in_size, out_size);
	}

	void activate(tensor_t<float>& in);

	int map(point_t d){
		return fc_layer->map(d);
	}

	void fix_weights(){
		fc_layer->fix_weights();
	}

	void calc_grads(tensor_t<float>& grad_next_layer){
		fc_layer->calc_grads(grad_next_layer);
	}

	void setWeights(tensor_t<float> newWeights){
		fc_layer->setWeights(newWeights);
	}

	void updateWeights(tensor_t<float> newWeights){
		fc_layer->updateWeights(newWeights);
	}
};



#endif /* FC_LAYER_CUDA_H_ */
