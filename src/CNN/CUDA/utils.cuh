/*
 * utils.cuh
 *
 *  Created on: 20/10/2017
 *      Author: sebas
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <string>

/*
 * Helper function to check if a CUDA error happen.
 */
#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                 \
  if(e!=cudaSuccess) {                                              \
	  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
	  exit(0);														\
  }                                                                 \
}

#endif /* UTILS_CUH_ */
