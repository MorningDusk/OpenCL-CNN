#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cnn.h"

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }  //에러 확인 함수

void build_error(cl_program program, cl_device_id device, cl_int err)
{
	if (err == CL_BUILD_PROGRAM_FAILURE)
	{
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, (void*)0, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, (void*)0);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
}

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */

cl_int err;					// Variable for Error check 
cl_platform_id platform;	// Platform ID
cl_device_id device;		// Device ID
cl_context context;
cl_program program;
cl_kernel convolution_layer, pooling_layer, fc_layer;

void cnn_init(void) {
	
	/* GET DEVICE INFO */
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	/* CREATE CONTEXT */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	/* CREATE KERNEL*/
	convolution_layer = creat_kernel("convolution_layer");
	pooling_layer = creat_kernel("pooling_layer");
	fc_layer = creat_kernel("fc_layer");

}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	/*
	 * TODO
	 * Implement here.
	 * Write classification results to labels and confidences.
	 * See "cnn_seq.c" if you don't know what to do.
	 */




	
}

cl_command_queue* creat_queue(int count)
{
	cl_int err = 0;
	cl_command_queue* queue = (cl_command_queue*)malloc(sizeof(cl_command_queue) * count);
	for (int i = 0; i < count; i++) {
		queue[i] = clCreateCommandQueue(context, device, 0, &err);
		CHECK_ERROR(err);
	}

	return queue;
}

void creat_program(void)
{
	cl_int err;
	size_t source_code_len;
	char* source_code = get_source_code("kernel.cl", &source_code_len);
	program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_code_len, &err);
	CHECK_ERROR(err);
}

void build_program(void)
{
	cl_int err;
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	build_error(program, device, err);
	CHECK_ERROR(err);
}

cl_kernel creat_kernel(const char* kernel_name)
{
	cl_int err;
	cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
	CHECK_ERROR(err);

	return kernel;
}
