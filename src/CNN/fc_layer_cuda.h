/*
 * fc_layer_cuda.cuh
 *
 *  Created on: 17/10/2017
 *      Author: sebas
 */

#ifndef FC_LAYER_CUDA_H_
#define FC_LAYER_CUDA_H_

#include "fc_layer.h"

class fc_layer_cuda_t: public fc_layer_t {

private:


public:

	fc_layer_cuda_t(tdsize in_size, int out_size) :
			fc_layer_t(in_size, out_size) {
		type = layer_type::fc_cuda;
	}

	void activate(tensor_t<float>& in);
};

#endif /* FC_LAYER_CUDA_H_ */
