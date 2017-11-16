/*
 * fc_layer_cuda.cuh
 *
 *  Created on: 17/10/2017
 *      Author: sebas
 */

#ifndef CUDAFCLAYER_H_
#define CUDAFCLAYER_H_

#include "../fc_layer.h"


class cudaFCLayer: public fc_layer_t {

private:


public:

	cudaFCLayer(tdsize in_size, int out_size) :
			fc_layer_t(in_size, out_size) {
		type = layer_type::fc_cuda;
	}

	void activate(tensor_t<float>& in);
};

#endif /* CUDAFCLAYER_H_ */
