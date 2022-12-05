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

void cnn_init() {
	
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	/* Create a context */
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	/*
	 * TODO
	 * Implement here.
	 * Write classification results to labels and confidences.
	 * See "cnn_seq.c" if you don't know what to do.
	 */

}