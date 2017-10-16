/*
 * cnn_cuda.cuh
 *
 *  Created on: 16/10/2017
 *      Author: sebas
 */

#ifndef CNN_CUDA_CUH_
#define CNN_CUDA_CUH_


vector<vector<layer_t*>> cuda_training(vector<case_t> cases, int batchSize,
		vector<vector<layer_t*>> slaves);


#endif /* CNN_CUDA_CUH_ */
