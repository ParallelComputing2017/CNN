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
void cudaCheckError(std::string msg);


#endif /* UTILS_CUH_ */
