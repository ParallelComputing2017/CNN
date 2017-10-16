/*
 * types.h
 *
 *  Created on: 16/10/2017
 *      Author: sebas
 */

#ifndef TYPES_H_
#define TYPES_H_

#include "CNN/tensor_t.h"

struct case_t {
	tensor_t<float> data;
	tensor_t<float> out;
};


#endif /* TYPES_H_ */
