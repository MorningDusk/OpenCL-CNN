#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}

#define CHECK_BUILD_ERROR(err) \
	if (err == CL_BUILD_PROGRAM_FAILURE) { \
		size_t log_size; \
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); \
		char *log = (char*)malloc(log_size); \
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL); \
		printf("%s\n", log); \
		free(log); \
	}

void softmax(float* output, int N);
int find_max(float* fc, int N);

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;

	FILE* file = fopen(file_name, "r");

	if (file == NULL) {
		printf("[%s:%d] Failed to open %s ", __FILE__, __LINE__, file_name);
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
	*len = length - cnt;

	fclose(file);

	return source_code;

}

/* For Sequantial */

float* alloc_layer_sq(size_t n) {
	return (cl_float*)malloc(n * sizeof(cl_float));
}

void convolution3x3(float* input, float* output, float* filter, int N) {
	int i, j, k, l;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			float sum = 0;
			for (k = 0; k < 3; k++) {
				for (l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N)
						sum += input[x * N + y] * filter[k * 3 + l];
				}
			}
			output[i * N + j] += sum;
		}
	}
}

#define ReLU(x) (((x)>0)?(x):0)
void convolution_layer_seq(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
	int i, j;

	memset(outputs, 0, sizeof(float) * N * N * D2);

	for (j = 0; j < D2; j++) {
		for (i = 0; i < D1; i++) {
			float* input = inputs + N * N * i;
			float* output = outputs + N * N * j;
			float* filter = filters + 3 * 3 * (j * D1 + i);
			convolution3x3(input, output, filter, N);
		}
	}

	for (i = 0; i < D2; i++) {
		float* output = outputs + N * N * i;
		float bias = biases[i];
		for (j = 0; j < N * N; j++) {
			output[j] = ReLU(output[j] + bias);
		}
	}
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
cl_command_queue queue;
cl_kernel convolution_layer, pooling_layer, fc_layer;

void cnn_init() {

	/* GET DEVICE INFO */
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	/* CREATE CONTEXT */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	/* CREATE QUEUE */
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

	/* CREATE PROGRAM*/
	size_t kernel_source_size;
	char* kernel_source_code = get_source_code("kernel.cl", &kernel_source_size);

	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source_code, &kernel_source_size, &err);
	CHECK_ERROR(err);

	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	CHECK_BUILD_ERROR(err);
	CHECK_ERROR(err);

	/* CREATE KERNEL*/
	//convolution_layer = clCreateKernel(program, "convolution_layer", &err);
	//CHECK_ERROR(err);
	pooling_layer = clCreateKernel(program, "pooling_layer", &err);
	CHECK_ERROR(err);
	fc_layer = clCreateKernel(program, "fc_layer", &err);
	CHECK_ERROR(err);
	

}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {

	int i, j, h, cnt;

	const int num_of_conv_layers[] = { 2, 2, 3, 3, 3 };

	const int NETWORK_SIZE[] = {
	64 * 3 * 3 * 3, 64,
	64 * 64 * 3 * 3, 64,
	128 * 64 * 3 * 3, 128,
	128 * 128 * 3 * 3, 128,
	256 * 128 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	512 * 256 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512 * 3 * 3, 512,
	512 * 512, 512,
	512 * 512, 512,
	10 * 512, 10
	};

	const int size_of_conv_layers[5] = {
		64 * 32 * 32,
		128 * 16 * 16,
		256 * 8 * 8,
		512 * 4 * 4,
		512 * 2 * 2
	};

	const int conv_layer_args[5][3] = {
		{ 64, 3, 32 },
		{ 128, 64, 16 },
		{ 256, 128, 8 },
		{ 512, 256, 4 },
		{ 512, 512, 2 }
	};

	const int size_of_pooling_layers[5][3] = {
		{ 64, 16, 16 },
		{ 128, 8, 8 },
		{ 256, 4, 4 },
		{ 512, 2, 2 },
		{ 512, 1, 1 }
	};

	const int size_of_fc_layers[3] = {
		512,
		512,
		10
	};

	size_t global_size, global_size2[2], global_size3[3];
	size_t local_size, local_size2[2], local_size3[3];

	cl_mem W[5][3], B[5][3];
	cl_mem C[5][3];
	cl_mem P[5];
	cl_mem FC[3];

	cnt = 0;

	cl_float** conv[5][3];
	cl_float** weight[5][3];
	cl_float** bias[5][3];
	cl_float* pool[5];
	cl_float* fc[3];

	/* Create Weight, bias Buffer */
	for (i = 0; i < 5; i++)
		for (j = 0; j < num_of_conv_layers[i]; j++) {

			weight[i][j] = network[cnt];
			bias[i][j] = network[cnt++];

			//W[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NETWORK_SIZE[cnt], network[cnt++], &err);
			//CHECK_ERROR(err);
			//B[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NETWORK_SIZE[cnt], network[cnt++], &err);
			//CHECK_ERROR(err);
		}

	/* Create Convolution Buffer */
	//for (i = 0; i < 5; i++)
	//	for (j = 0; j < num_of_conv_layers[i]; j++) {
	//		C[i][j] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_of_conv_layers[i], NULL, &err);
	//		CHECK_ERROR(err);
	//	}

	/* Create Pooling Buffer */
	for (i = 0; i < 5; i++) {
		P[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_of_pooling_layers[i][0] *
			size_of_pooling_layers[i][1] * size_of_pooling_layers[i][2], NULL, &err);
		CHECK_ERROR(err);
	}

	/* Create FC layer Buffer */
	for (i = 0; i < 3; i++) {
		FC[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size_of_fc_layers[i], NULL, &err);
		CHECK_ERROR(err);
	}

	

	/* Run CNN */
	for (i = 0; i < num_images; ++i) {

		float* image = images + i * 3 * 32 * 32;

		/* Start Convolution Layer*/
		for (j = 0; j < 5; j++) {

			for (h = 0; h < num_of_conv_layers[j]; h++) {
				conv[j][h] = alloc_layer_sq(size_of_conv_layers[j]);
				
				if (j == 0 && h == 0) {
					convolution_layer_seq(image, conv[j][h], weight[j][h], bias[j][h], conv_layer_args[i][0], conv_layer_args[i][1], conv_layer_args[i][2]);
				} else {
					convolution_layer_seq(conv[j][h - 1], conv[j][h], weight[j][h], bias[j][h], conv_layer_args[i][0], conv_layer_args[i][1], conv_layer_args[i][2]);
				}

				C[j][h] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size_of_conv_layers[i], conv[j][h], &err);
				CHECK_ERROR(err);

			}

			
			/* Start Pooling Layer*/
			// pooling_layer(__global float* input, __global float* output, const int N, const int Nsquare)

			err = clSetKernelArg(pooling_layer, 0, sizeof(cl_mem), &C[j][h - 1]);
			CHECK_ERROR(err);
			err = clSetKernelArg(pooling_layer, 1, sizeof(cl_mem), P + j);
			CHECK_ERROR(err);
			err = clSetKernelArg(pooling_layer, 2, sizeof(cl_int), &size_of_pooling_layers[j][0]);
			CHECK_ERROR(err);
			err = clSetKernelArg(pooling_layer, 3, sizeof(cl_int), &size_of_pooling_layers[j][1]);
			CHECK_ERROR(err);

			global_size3[0] = size_of_pooling_layers[j][0]; global_size3[1] = size_of_pooling_layers[j][1]; global_size3[2] = size_of_pooling_layers[j][2];
			local_size3[0] = 4; local_size3[1] = 4; local_size3[2] = 4;

			clEnqueueNDRangeKernel(queue, pooling_layer, 3, NULL, global_size3, local_size3, 0, NULL, NULL);
			CHECK_ERROR(err);

			err = clFinish(queue);
			CHECK_ERROR(err);
		}

		/* Start FC Layer */
		// fc_layer(input, output, wegiht, bias, in, out);
		float* w[3] = {network[26], network[28], network[30]};
		float* b[3] = {network[27], network[29], network[31]};
		for (i = 0; i < 3; i++) {

			err = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), P + 5);
			if (i != 0) {
				err = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), FC + (i - 1));
			} CHECK_ERROR(err);

			err = clSetKernelArg(fc_layer, 1, sizeof(cl_mem), FC + i);
			CHECK_ERROR(err);
			err = clSetKernelArg(fc_layer, 2, sizeof(cl_float), &w[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(fc_layer, 3, sizeof(cl_float), &b[i]);
			CHECK_ERROR(err);

			int N = 512;
			err = clSetKernelArg(fc_layer, 4, sizeof(int), N);
			CHECK_ERROR(err);

			err = clSetKernelArg(fc_layer, 5, sizeof(int), size_of_fc_layers + i);
			CHECK_ERROR(err);

			global_size2[0] = N;
			global_size2[1] = size_of_fc_layers[i];	
			local_size2[0] = 64;	
			local_size2[1] = 1;		

			clEnqueueNDRangeKernel(queue, fc_layer, 2, NULL, global_size2, local_size2, 0, NULL, NULL);			
			CHECK_ERROR(err);
			err = clFinish(queue);			
			CHECK_ERROR(err);
		}

		float fc3[10];

		err = clEnqueueReadBuffer(queue, FC[3], CL_TRUE, 0, sizeof(float) * size_of_fc_layers[3], fc3, 0, NULL, NULL);
		CHECK_ERROR(err);
		softmax(fc3, 10);
		labels[i] = find_max(fc3, 10);
		confidences[i] = fc3[labels[i]];

	}

	for (i = 0; i < 5; i++)
		for (j = 0; j < num_of_conv_layers[i]; j++) {
			clReleaseMemObject(W[i][j]);
			clReleaseMemObject(B[i][j]);
			if (0 < i) { err = clReleaseMemObject(C[i][j]); }
			if (i == 0) { err = clReleaseMemObject(FC[j]); } // i == 0 일때 1~3 까지 총 3번 실행됨
		}

	clReleaseMemObject(P[i]);
	clReleaseKernel(convolution_layer);
	clReleaseKernel(pooling_layer);
	clReleaseKernel(fc_layer);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(device);
	free(platform);

}

void softmax(float* output, int N) {
	int i;
	float max = output[0];
	float sum = 0;

	for (i = 1; i < N; i++)
		max = (output[i] > max) ? output[i] : max;

	for (i = 0; i < N; i++)
		sum += exp(output[i] - max);

	for (i = 0; i < N; i++)
		output[i] = exp(output[i] - max) / sum;
}

int find_max(float* fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;

	for (i = 0; i < N; i++)
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}

	return maxid;
}

