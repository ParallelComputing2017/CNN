
#include <string>

#include "utils.cuh"


void cudaCheckError(std::string msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to: %s (error code %s)!\n", msg.c_str(),
				cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}
}

