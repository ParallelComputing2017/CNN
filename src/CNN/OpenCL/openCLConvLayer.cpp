/*
 * openCLConvLayer.cpp
 *
 *  Created on: 21/11/2017
 *      Author: sebas
 */

#include <CL/cl.h>

#include "openCLConvLayer.h"
#include "err_code.h"
#include "device_info.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern int output_device_info(cl_device_id);

const char *activationKernel =
		"\n"
				"point_t map_to_input(int stride, point_t out, int z) {\n"
				"	out.x *= stride;\n"
				"	out.y *= stride;\n"
				"	out.z = z;\n"
				"	return out;\n"
				"}\n"
				""
				"__kernel void activationKernel(tensor_t<float> *in, tensor_t<float> **filters, int *stride, tensor_t<float> *out) {\n"
				""
				"	int x = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
				"	int y = (blockIdx.y * blockDim.y) + threadIdx.y;\n"
				"	int filterIdx = (blockIdx.z * blockDim.z) + threadIdx.z;\n"
				""
				"	if (x < out->size.x && y < out->size.y) {\n"
				""
				"		point_t p_out = (point_t ) { x, y, 0 };\n"
				"		point_t mapped = map_to_input((*stride), p_out, 0);\n"
				"		float sum = 0;\n"
				""
				"		tensor_t<float> *kernel = filters[filterIdx];\n"
				""
				"		for (int i = 0; i < kernel->size.x; i++) {\n"
				"			for (int j = 0; j < kernel->size.y; j++) {\n"
				"				for (int z = 0; z < in->size.z; z++) {\n"
				""
				"					float f = cudaTensor::get(kernel, i, j, z);\n"
				"					float v = cudaTensor::get(in, mapped.x + i, mapped.y + j, z);\n"
				"					sum += f * v;\n"
				"				}\n"
				"			}\n"
				"		}\n"
				"		cudaTensor::set(out, x, y, filterIdx, sum);\n"
				"		\n"
				"	}\n"
				"	\n"
				"}\n"
				"__kernel void activation(                                                 \n"
				"   __global float* a,                                                  \n"
				"   __global float* b,                                                  \n"
				"   __global float* c,                                                  \n"
				"   const unsigned int count){                                           \n"
				"                                                                      \n"
				"	uint y = get_global_id(0);			\n"
				"	if( y < count){			\n"
				"   	// Row pointer		\n"
				"		const __global float* row = a + y * count;		\n"
				"   	for (int j = 0; j < count; j++) {       		\n"
				"   		for (int k = 0; k < count; k++) {   		\n"
				"   			c[y * count + j] += row[k] * b[k * count + j];    \n"
				"   		}                                    \n"
				"   	}                                        \n"
				"   }                                            \n"
				"}                                                                      \n"
				"\n";

void OpenCLConvLayer::activate(tensor_t<float>& in) {
	this->in = in;

	float* h_a;
	float* h_b;
	float* h_c;

	int err;               // error code returned from OpenCL calls

	int count = 1;

	size_t global = count;                  // global domain size

	cl_device_id device_id;     // compute device id
	cl_context context;       // compute context
	cl_command_queue commands;      // compute command queue
	cl_program program;       // compute program
	cl_kernel ko_vadd;       // compute kernel

	cl_mem d_a;                    // device memory used for the input  a vector
	cl_mem d_b;                    // device memory used for the input  b vector
	cl_mem d_c;                    // device memory used for the output c vector

	// Set up platform and GPU device

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkError(err, "Finding platforms");
	if (numPlatforms == 0) {
		printf("Found 0 platforms!\n");
		exit(EXIT_FAILURE);
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	checkError(err, "Getting platforms");

	// Secure a GPU
	for (int i = 0; i < numPlatforms; i++) {
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS) {
			break;
		}
	}

	if (device_id == NULL)
		checkError(err, "Finding a device");

	err = output_device_info(device_id);
	checkError(err, "Printing device output");

	// Create a compute context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	checkError(err, "Creating context");

	// Create a command queue
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	checkError(err, "Creating command queue");

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1,
			(const char **) &activationKernel,
			NULL, &err);
	checkError(err, "Creating program");

	// Build the program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n%s\n",
				err_code(err));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
				sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(EXIT_FAILURE);
	}

	// Create the compute kernel from the program
	ko_vadd = clCreateKernel(program, "activate", &err);
	checkError(err, "Creating kernel");

	long size = sizeof(float);

	// Create the input (a, b) and output (c) arrays in device memory
	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
	checkError(err, "Creating buffer d_a");

	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
	checkError(err, "Creating buffer d_b");

	d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size,
	NULL, &err);
	checkError(err, "Creating buffer d_c");

	// Write a and b vectors into compute device memory

	err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, size, h_a, 0,
	NULL, NULL);
	checkError(err, "Copying h_a to device at d_a");

	err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, size, h_b, 0,
	NULL, NULL);
	checkError(err, "Copying h_b to device at d_b");

	// Set the arguments to our compute kernel
	err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
	err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
	err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
	checkError(err, "Setting kernel arguments");

	// Execute the kernel over the entire range of our 1d input data set
	// letting the OpenCL runtime choose the work-group size

	err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0,
	NULL, NULL);
	checkError(err, "Enqueueing kernel");

	// Wait for the commands to complete before stopping the timer
	err = clFinish(commands);
	checkError(err, "Waiting for kernel to finish");

	// Read back the results from the compute device
	err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, size, h_c, 0,
	NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array!\n%s\n", err_code(err));
		exit(1);
	}

	// cleanup then shutdown
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);
	clReleaseProgram(program);
	clReleaseKernel(ko_vadd);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
}

