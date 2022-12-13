#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define CHECK_ERROR(err)                                          \
  if (err != CL_SUCCESS)                                          \
  {                                                               \
	printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
	exit(EXIT_FAILURE);                                           \
  } // 에러 확인 함수

#define CHECK_BUILD_ERROR(err)                                                         \
  if (err == CL_BUILD_PROGRAM_FAILURE)                                                 \
  {                                                                                    \
	size_t log_size;                                                                   \
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);  \
	char *log = (char *)malloc(log_size);                                              \
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL); \
	printf("%s\n", log);                                                               \
	free(log);                                                                         \
  }

char* get_source_code(const char* file_name, size_t* len)
{
	FILE* file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}

/*
 * Define global variables
 */

const int NETWORK_SIZES[] = {
	64 * 3 * 3 * 3, 64,
	64 * 64 * 3 * 3, 64,
	128 * 64 * 3 * 3, 128,
	128 * 128 * 3 * 3, 128,
	256 * 128 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256,
	256 * 256 * 3 * 3, 256, // 12 13
	512 * 256 * 3 * 3, 512, // 14 15
	512 * 512 * 3 * 3, 512, // 16 17
	512 * 512 * 3 * 3, 512, // 18 19
	512 * 512 * 3 * 3, 512, // 20 21
	512 * 512 * 3 * 3, 512, // 22 23
	512 * 512 * 3 * 3, 512, // 24 25
	512 * 512, 512,
	512 * 512, 512,
	10 * 512, 10 };

const int CONV_LAYERS_ARGS[5][3] = { // D1, D2, N
	{3, 64, 32},
	{64, 128, 16},
	{128, 256, 8},
	{256, 512, 4},
	{512, 512, 2} };

const int POOL_LAYER_SIZES[5][3] = { // DIM, N, N
	{64, 16, 16},
	{128, 8, 8},
	{256, 4, 4},
	{512, 2, 2},
	{512, 1, 1} };

const int FC_LAYER_SIZES[3] = {
	512,
	512,
	10 };

const int TS = 16;
const int IMAGE_SIZE = 3072;
const int NETWORK_BUFFER_SIZE = 60980520;
const int BUFFER_SIZE = 65536;
const int CONV_TEMP_BUFFER_SIZE = 589824;
const int LAYER_BUFFER_SIZE = 512;
const int OUT_NEURON_SIZE = 10;

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

cl_kernel conv1_layer, conv2_layer, p_layer, f_layer;

cl_mem imageBuffer, networkBuffer;
cl_mem convInputBuffer, convTempBuffer, convOutputBuffer;
cl_mem poolInputBuffer, poolOutputBuffer;
cl_mem fcInputBuffer, fcOutputBuffer;

size_t global_size;
size_t local_size;
size_t global_size2[2];
size_t local_size2[2];

int imageOffset, networkOffset;

size_t kernel_source_size;
size_t global_size;
size_t local_size;
size_t global_size2[2];
size_t local_size2[2];


void convolution_layer(cl_mem* inputs, cl_mem* outputs, cl_mem* networks, int inDim, int outDim, int N) {

	// (inputs, imageOffset, outputs, outDim, N)
	err = clSetKernelArg(conv1_layer, 0, sizeof(cl_mem), inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv1_layer, 1, sizeof(cl_int), &imageOffset);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv1_layer, 2, sizeof(cl_mem), &convTempBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv1_layer, 3, sizeof(cl_int), &outDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv1_layer, 4, sizeof(cl_int), &N);
	CHECK_ERROR(err);

	global_size2[0] = N * N * outDim;
	global_size2[1] = 1;
	local_size2[0] = TS * TS;
	local_size2[1] = 1;

	err = clEnqueueNDRangeKernel(queue, conv1_layer, 2, NULL, global_size2, local_size2, 0, NULL, NULL);
	CHECK_ERROR(err);

	// (inputs, outputs, networks, networkOffset, inDim, outDim, N)
	err = clSetKernelArg(conv2_layer, 0, sizeof(cl_mem), &convTempBuffer);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv2_layer, 1, sizeof(cl_mem), outputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv2_layer, 2, sizeof(cl_mem), networks);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv2_layer, 3, sizeof(int), &networkOffset);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv2_layer, 4, sizeof(int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv2_layer, 5, sizeof(int), &outDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(conv2_layer, 6, sizeof(int), &N);
	CHECK_ERROR(err);

	global_size2[0] = N * N;
	global_size2[1] = outDim;
	local_size2[0] = TS;
	local_size2[1] = TS;

	if (global_size2[0] < TS) {
		global_size2[0] = TS;
	}

	err = clEnqueueNDRangeKernel(queue, conv2_layer, 2, NULL, global_size2, local_size2, 0, NULL, NULL);
	CHECK_ERROR(err);
}

void max_pooling_layer(cl_mem* inputs, cl_mem* outputs, int inDim, int N)
{

	// (inputs, outputs, inDim, N)
	err = clSetKernelArg(p_layer, 0, sizeof(cl_mem), inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(p_layer, 1, sizeof(cl_mem), outputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(p_layer, 2, sizeof(cl_int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(p_layer, 3, sizeof(cl_int), &N);
	CHECK_ERROR(err);

	global_size2[0] = inDim * N;
	global_size2[1] = N;

	err = clEnqueueNDRangeKernel(queue, p_layer, 2, NULL, global_size2, NULL, 0, NULL, NULL);
}

void fc_layer(cl_mem* inputs, cl_mem* outputs, cl_mem* networks, int inDim, int outDim)
{

	// (inputs, outputs, filters, networkOffset, inDim, outDim)
	err = clSetKernelArg(f_layer, 0, sizeof(cl_mem), inputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(f_layer, 1, sizeof(cl_mem), outputs);
	CHECK_ERROR(err);
	err = clSetKernelArg(f_layer, 2, sizeof(cl_mem), networks);
	CHECK_ERROR(err);
	err = clSetKernelArg(f_layer, 3, sizeof(cl_int), &networkOffset);
	CHECK_ERROR(err);
	err = clSetKernelArg(f_layer, 4, sizeof(cl_int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(f_layer, 5, sizeof(cl_int), &outDim);
	CHECK_ERROR(err);

	global_size = outDim;
	local_size = (outDim != 10 ? TS : 2);

	err = clEnqueueNDRangeKernel(queue, f_layer, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
}

float* alloc_layer(size_t n)
{
	return (float*)malloc(n * sizeof(float));
}

void softmax(float* output, int N)
{
	int i;
	float sum = 0;
	float max = output[0];

	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}

	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}

	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

int find_max(float* input, int classNum)
{
	int i;
	int maxIndex = 0;
	float max = 0;

	for (i = 0; i < classNum; i++)
	{
		if (max < input[i]) {
			max = input[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

void cnn_init()
{

	/* GET DEVICE INFO */
	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);

	/* CREATE CONTEXT */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	/* CREATE QUEUE */
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
	CHECK_ERROR(err);

	/* CREATE PROGRAM */
<<<<<<< HEAD
	char* kernel_source;
	size_t kernel_source_size;
=======
>>>>>>> 362a559d54bc7db66657f7fd07bfb28d9342d74a
	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);
	err = clBuildProgram(program, 1, &device, "-cl-no-signed-zeros -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only", NULL, NULL);
	CHECK_BUILD_ERROR(err);
	CHECK_ERROR(err);

	/* CREATE KERNEL */
	conv1_layer = clCreateKernel(program, "convolution_1", &err);
	CHECK_ERROR(err);
	conv2_layer = clCreateKernel(program, "convolution_2", &err);
	CHECK_ERROR(err);
	p_layer = clCreateKernel(program, "pooling_max", &err);
	CHECK_ERROR(err);
	f_layer = clCreateKernel(program, "fc_layer", &err);
	CHECK_ERROR(err);
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_images)
{

	int i;

	// Create Buffer
	imageBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num_images * IMAGE_SIZE, NULL, &err);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, imageBuffer, CL_FALSE, 0, sizeof(float) * num_images * IMAGE_SIZE, images, 0, NULL, NULL);
	CHECK_ERROR(err);

	networkBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NETWORK_BUFFER_SIZE, network, &err);
	CHECK_ERROR(err);

	convInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);
	convTempBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * CONV_TEMP_BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);
	convOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);

	poolInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);
	poolOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);

	fcInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * LAYER_BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);
	fcOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * LAYER_BUFFER_SIZE, NULL, &err);
	CHECK_ERROR(err);

	// Allocate memory for fc layer
	float* fc = alloc_layer(10);

	for (i = 0; i < num_images; i++) {

		imageOffset = i * IMAGE_SIZE;
		networkOffset = 0;

		convolution_layer(&imageBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[0][0], CONV_LAYERS_ARGS[0][1], CONV_LAYERS_ARGS[0][2]);
		networkOffset += NETWORK_SIZES[0] + NETWORK_SIZES[1];
		imageOffset = 0;
		convolution_layer(&convOutputBuffer, &convInputBuffer, &networkBuffer, CONV_LAYERS_ARGS[0][1], CONV_LAYERS_ARGS[0][1], CONV_LAYERS_ARGS[0][2]);
		networkOffset += NETWORK_SIZES[2] + NETWORK_SIZES[3];

		max_pooling_layer(&convInputBuffer, &poolOutputBuffer, POOL_LAYER_SIZES[0][0], POOL_LAYER_SIZES[0][1]);

		convolution_layer(&poolOutputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[1][0], CONV_LAYERS_ARGS[1][1], CONV_LAYERS_ARGS[1][2]);
		networkOffset += NETWORK_SIZES[4] + NETWORK_SIZES[5];
		convolution_layer(&convOutputBuffer, &convInputBuffer, &networkBuffer, CONV_LAYERS_ARGS[1][1], CONV_LAYERS_ARGS[1][1], CONV_LAYERS_ARGS[1][2]);
		networkOffset += NETWORK_SIZES[6] + NETWORK_SIZES[7];
		max_pooling_layer(&convInputBuffer, &poolOutputBuffer, POOL_LAYER_SIZES[1][0], POOL_LAYER_SIZES[1][1]);

		convolution_layer(&poolOutputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[2][0], CONV_LAYERS_ARGS[2][1], CONV_LAYERS_ARGS[2][2]);
		networkOffset += NETWORK_SIZES[8] + NETWORK_SIZES[9];
		convolution_layer(&convOutputBuffer, &convInputBuffer, &networkBuffer, CONV_LAYERS_ARGS[2][1], CONV_LAYERS_ARGS[2][1], CONV_LAYERS_ARGS[2][2]);
		networkOffset += NETWORK_SIZES[10] + NETWORK_SIZES[11];
		convolution_layer(&convInputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[2][1], CONV_LAYERS_ARGS[2][1], CONV_LAYERS_ARGS[2][2]);
		networkOffset += NETWORK_SIZES[12] + NETWORK_SIZES[13];

		max_pooling_layer(&convOutputBuffer, &poolOutputBuffer, POOL_LAYER_SIZES[2][0], POOL_LAYER_SIZES[2][1]);

		convolution_layer(&poolOutputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[3][0], CONV_LAYERS_ARGS[3][1], CONV_LAYERS_ARGS[3][2]);
		networkOffset += NETWORK_SIZES[14] + NETWORK_SIZES[15];
		convolution_layer(&convOutputBuffer, &convInputBuffer, &networkBuffer, CONV_LAYERS_ARGS[3][1], CONV_LAYERS_ARGS[3][1], CONV_LAYERS_ARGS[3][2]);
		networkOffset += NETWORK_SIZES[16] + NETWORK_SIZES[17];
		convolution_layer(&convInputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[3][1], CONV_LAYERS_ARGS[3][1], CONV_LAYERS_ARGS[3][2]);
		networkOffset += NETWORK_SIZES[18] + NETWORK_SIZES[19];

		max_pooling_layer(&convOutputBuffer, &poolOutputBuffer, POOL_LAYER_SIZES[3][0], POOL_LAYER_SIZES[3][1]);

		convolution_layer(&poolOutputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[4][0], CONV_LAYERS_ARGS[4][1], CONV_LAYERS_ARGS[4][2]);
		networkOffset += NETWORK_SIZES[20] + NETWORK_SIZES[21];
		convolution_layer(&convOutputBuffer, &convInputBuffer, &networkBuffer, CONV_LAYERS_ARGS[4][1], CONV_LAYERS_ARGS[4][1], CONV_LAYERS_ARGS[4][2]);
		networkOffset += NETWORK_SIZES[22] + NETWORK_SIZES[23];
		convolution_layer(&convInputBuffer, &convOutputBuffer, &networkBuffer, CONV_LAYERS_ARGS[4][1], CONV_LAYERS_ARGS[4][1], CONV_LAYERS_ARGS[4][2]);
		networkOffset += NETWORK_SIZES[24] + NETWORK_SIZES[25];

		max_pooling_layer(&convOutputBuffer, &poolOutputBuffer, POOL_LAYER_SIZES[04][0], POOL_LAYER_SIZES[4][1]);

		fc_layer(&poolOutputBuffer, &fcOutputBuffer, &networkBuffer, FC_LAYER_SIZES[0], FC_LAYER_SIZES[0]);
		networkOffset += NETWORK_SIZES[26] + NETWORK_SIZES[27];
		fc_layer(&fcOutputBuffer, &fcInputBuffer, &networkBuffer, FC_LAYER_SIZES[1], FC_LAYER_SIZES[1]);
		networkOffset += NETWORK_SIZES[28] + NETWORK_SIZES[29];
		fc_layer(&fcInputBuffer, &fcOutputBuffer, &networkBuffer, FC_LAYER_SIZES[1], FC_LAYER_SIZES[2]);

		err = clEnqueueReadBuffer(queue, fcOutputBuffer, CL_TRUE, 0, sizeof(float) * OUT_NEURON_SIZE, fc, 0, NULL, NULL);
		CHECK_ERROR(err);

		softmax(fc, OUT_NEURON_SIZE);

		labels[i] = find_max(fc, OUT_NEURON_SIZE);
		confidences[i] = fc[labels[i]];
	}

	// Release for Build Info
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseProgram(program);

	// Release Kernel
	clReleaseKernel(conv1_layer);
	clReleaseKernel(conv2_layer);
	clReleaseKernel(f_layer);
	clReleaseKernel(p_layer);

	// Release Buffer
	clReleaseMemObject(imageBuffer);
	clReleaseMemObject(networkBuffer);
	clReleaseMemObject(convInputBuffer);
	clReleaseMemObject(convOutputBuffer);
	clReleaseMemObject(convTempBuffer);
	clReleaseMemObject(fcInputBuffer);
	clReleaseMemObject(fcOutputBuffer);
	clReleaseMemObject(poolInputBuffer);
	clReleaseMemObject(poolOutputBuffer);

	free(fc);
}