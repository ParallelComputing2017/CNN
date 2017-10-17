#pragma once
#include "tensor_t.h"
#include "types.h"

#pragma pack(push, 1)
struct layer_t
{

	layer_type type;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
};
#pragma pack(pop)
