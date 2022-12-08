#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include "cnn.h"

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
cl_kernel c_layer, p_layer, f_layer;

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
    CHECK_BUILD_ERROR(err);
    CHECK_ERROR(err);
}

cl_kernel creat_kernel(const char* kernel_name)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    CHECK_ERROR(err);

    return kernel;
}

static void pl_make_buffers(cl_kernel kernel, cl_command_queue* queue, float* inputs, float* outputs, cl_mem* input_result,
    cl_mem* output_result, int D, int N)
{
    cl_int err;

    size_t size = sizeof(float) * D * N * N;

    // This should be removed after the completion of Convolution OpenCL Porting
    *input_result = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    CHECK_ERROR(err);

    clEnqueueWriteBuffer(*queue, *input_result, CL_FALSE, 0, size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(*queue);

    *output_result = clCreateBuffer(context, CL_MEM_READ_WRITE, size / 4, NULL, &err);
    CHECK_ERROR(err);
}

// Temporary function
static void pl_get_result(cl_command_queue* queue, cl_mem* result, float* output, int D, int N)
{
    cl_int err = clEnqueueReadBuffer(*queue, *result, CL_FALSE, 0, (sizeof(float) * N * N * D) / 4,
        output, 0, NULL, NULL);
    CHECK_ERROR(err);

    clFinish(*queue);
}

static void pooling_layer(cl_command_queue* queue, int queue_count, cl_kernel kernel, cl_mem* inputs, cl_mem* outputs, int D, int N)
{
    // Set kernel arguments
    cl_int err;
    int size = N * N;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_int), &N);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &size);
    CHECK_ERROR(err);

    size_t global_size[3] = { D, N, N };
    size_t local_size[3] = { 4, 4, 4 };

    err = clEnqueueNDRangeKernel(*queue, kernel, 3, NULL, global_size, local_size,
        0, NULL, NULL);
    CHECK_ERROR(err);

    clFinish(*queue);
    clReleaseMemObject(*inputs);
}

static void convolution3x3(float* input, float* output, float* filter, int N) {
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

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {
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

void cnn_init(void) {

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

    /* CREATE PROGRAM */
    creat_program();
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_BUILD_ERROR(err);
    CHECK_ERROR(err);

    /* CREATE KERNEL*/
    c_layer = creat_kernel("convolution_layer");
    p_layer = creat_kernel("pooling_layer");
    f_layer = creat_kernel("fc_layer");

}


void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
    // slice the network into weights and biases
    float* w1_1, * b1_1, * w1_2, * b1_2;
    float* w2_1, * b2_1, * w2_2, * b2_2;
    float* w3_1, * b3_1, * w3_2, * b3_2, * w3_3, * b3_3;
    float* w4_1, * b4_1, * w4_2, * b4_2, * w4_3, * b4_3;
    float* w5_1, * b5_1, * w5_2, * b5_2, * w5_3, * b5_3;
    float* w1, * b1, * w2, * b2, * w3, * b3;
    w1_1 = network[0]; b1_1 = network[1];
    w1_2 = network[2]; b1_2 = network[3];
    w2_1 = network[4]; b2_1 = network[5];
    w2_2 = network[6]; b2_2 = network[7];
    w3_1 = network[8]; b3_1 = network[9];
    w3_2 = network[10]; b3_2 = network[11];
    w3_3 = network[12]; b3_3 = network[13];
    w4_1 = network[14]; b4_1 = network[15];
    w4_2 = network[16]; b4_2 = network[17];
    w4_3 = network[18]; b4_3 = network[19];
    w5_1 = network[20]; b5_1 = network[21];
    w5_2 = network[22]; b5_2 = network[23];
    w5_3 = network[24]; b5_3 = network[25];
    w1 = network[26]; b1 = network[27];
    w2 = network[28]; b2 = network[29];
    w3 = network[30]; b3 = network[31];

    // allocate memory for output of each layer
    float* c1_1, * c1_2, * p1;
    float* c2_1, * c2_2, * p2;
    float* c3_1, * c3_2, * c3_3, * p3;
    float* c4_1, * c4_2, * c4_3, * p4;
    float* c5_1, * c5_2, * c5_3, * p5;
    float* fc1, * fc2, * fc3;
    c1_1 = alloc_layer(64 * 32 * 32);
    c1_2 = alloc_layer(64 * 32 * 32);
    p1 = alloc_layer(64 * 16 * 16);
    c2_1 = alloc_layer(128 * 16 * 16);
    c2_2 = alloc_layer(128 * 16 * 16);
    p2 = alloc_layer(128 * 8 * 8);
    c3_1 = alloc_layer(256 * 8 * 8);
    c3_2 = alloc_layer(256 * 8 * 8);
    c3_3 = alloc_layer(256 * 8 * 8);
    p3 = alloc_layer(256 * 4 * 4);
    c4_1 = alloc_layer(512 * 4 * 4);
    c4_2 = alloc_layer(512 * 4 * 4);
    c4_3 = alloc_layer(512 * 4 * 4);
    p4 = alloc_layer(512 * 2 * 2);
    c5_1 = alloc_layer(512 * 2 * 2);
    c5_2 = alloc_layer(512 * 2 * 2);
    c5_3 = alloc_layer(512 * 2 * 2);
    p5 = alloc_layer(512 * 1 * 1);
    fc1 = alloc_layer(512);
    fc2 = alloc_layer(512);
    fc3 = alloc_layer(10);

    // Make the host ready to use OpenCL

    cl_command_queue* queue = creat_queue(1);
    cl_kernel pl_kernel = creat_kernel("pooling_layer");

    cl_mem pl_input, pl_output;

    // run network
    for (int i = 0; i < num_images; ++i)
    {
        float* image = images + i * 3 * 32 * 32;

        convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);
        convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);

        // pooling_layer(c1_2, p1, 64, 16);
        pl_make_buffers(pl_kernel, queue, c1_2, p1, &pl_input, &pl_output, 64, 16);
        pooling_layer(queue, 1, pl_kernel, &pl_input, &pl_output, 64, 16);
        pl_get_result(queue, &pl_output, p1, 64, 16);

        convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
        convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);

        // pooling_layer(c2_2, p2, 128, 8);
        pl_make_buffers(pl_kernel, queue, c2_2, p2, &pl_input, &pl_output, 128, 8);
        pooling_layer(queue, 1, pl_kernel, &pl_input, &pl_output, 128, 8);
        pl_get_result(queue, &pl_output, p2, 128, 8);
        
        convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
        convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
        convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);

        // pooling_layer(c3_3, p3, 256, 4);
        pl_make_buffers(pl_kernel, queue, c3_3, p3, &pl_input, &pl_output, 256, 4);
        pooling_layer(queue, 1, pl_kernel, &pl_input, &pl_output, 256, 4);
        pl_get_result(queue, &pl_output, p3, 256, 4);

        convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
        convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
        convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);

        // pooling_layer(c4_3, p4, 512, 2);
        pl_make_buffers(pl_kernel, queue, c4_3, p4, &pl_input, &pl_output, 512, 2);
        pooling_layer(queue, 1, pl_kernel, &pl_input, &pl_output, 512, 2);
        pl_get_result(queue, &pl_output, p4, 512, 2);

        convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
        convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
        convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);

        // pooling_layer(c5_3, p5, 512, 1);
        pl_make_buffers(pl_kernel, queue, c5_3, p5, &pl_input, &pl_output, 512, 1);
        pooling_layer(queue, 1, pl_kernel, &pl_input, &pl_output, 512, 1);
        pl_get_result(queue, &pl_output, p5, 512, 1);

        fc_layer(p5, fc1, w1, b1, 512, 512);
        fc_layer(fc1, fc2, w2, b2, 512, 512);
        fc_layer(fc2, fc3, w3, b3, 10, 512);

        softmax(fc3, 10);

        labels[i] = find_max(fc3, 10);
        confidences[i] = fc3[labels[i]];
    }

    free(c1_1); free(c1_2); free(p1);
    free(c2_1); free(c2_2); free(p2);
    free(c3_1); free(c3_2); free(c3_3); free(p3);
    free(c4_1); free(c4_2); free(c4_3); free(p4);
    free(c5_1); free(c5_2); free(c5_3); free(p5);
    free(fc1); free(fc2); free(fc3);
}

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


static void softmax(float* output, int N) {
    int i;
    float max = output[0];
    for (i = 1; i < N; i++) {
        max = (output[i] > max) ? output[i] : max;
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(output[i] - max);
    }
    for (i = 0; i < N; i++) {
        output[i] = exp(output[i] - max) / sum;
    }
}

static int find_max(float* fc, int N) {
    int i;
    int maxid = 0;
    float maxval = 0;
    for (i = 0; i < N; i++) {
        if (maxval < fc[i]) {
            maxval = fc[i];
            maxid = i;
        }
    }
    return maxid;
}

/*
void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {

    int i, j, h, cnt;

    const int CONV_LAYER_NUMS[] = { 2, 2, 3, 3, 3 };

    const int NETWORK_SIZES[] = {
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

    const int CONV_LAYER_SIZES[5] = {
        64 * 32 * 32,
        128 * 16 * 16,
        256 * 8 * 8,
        512 * 4 * 4,
        512 * 2 * 2
    };

    const int CONV_LAYERS_ARGS[5][3] = {
        { 64, 3, 32 },
        { 128, 64, 16 },
        { 256, 128, 8 },
        { 512, 256, 4 },
        { 512, 512, 2 }
    };

    const int POOL_LAYER_SIZES[5][3] = {
        { 64, 16, 16 },
        { 128, 8, 8 },
        { 256, 4, 4 },
        { 512, 2, 2 },
        { 512, 1, 1 }
    };

    const int FC_LAYER_SIZES[3] = {
        512,
        512,
        10
    };

    float fc3[10];

    size_t global_size, global_size2[2], global_size3[3];
    size_t local_size, local_size2[2], local_size3[3];

    cl_mem W[5][3], B[5][3];
    cl_mem C[5][3];
    cl_mem P[5];
    cl_mem FC[3];

    cnt = 0;

    // Create Weight, bias Buffer 
    for (i = 0; i < 5; i++) {
        for (j = 0; j < CONV_LAYER_NUMS[i]; j++) {
            W[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NETWORK_SIZES[cnt], network[cnt++], &err);
            CHECK_ERROR(err);
            B[i][j] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NETWORK_SIZES[cnt], network[cnt++], &err);
            CHECK_ERROR(err);
        }
    }

    // Create Convolution Buffer 
    for (i = 0; i < 5; i++) {
        for (j = 0; j < CONV_LAYER_NUMS[i]; j++) {
            C[i][j] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * CONV_LAYER_SIZES[i], NULL, &err);
            CHECK_ERROR(err);
        }
    }

    // Create Pooling Buffer 
    for (i = 0; i < 5; i++) {
        P[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * POOL_LAYER_SIZES[i][0] *
            POOL_LAYER_SIZES[i][1] * POOL_LAYER_SIZES[i][2], NULL, &err);
        CHECK_ERROR(err);
    }

    // Create FC layer Buffer 
    for (i = 0; i < 3; i++) {
        FC[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * FC_LAYER_SIZES[i], NULL, &err);
        CHECK_ERROR(err);
    }

    // Run CNN 
    for (i = 0; i < num_images; ++i) {

        float* image = images + i * 3 * 32 * 32;

        // Start Convolution Layer
        // convolution_layer(__global float* inputs, __global float* outputs, __global float* filters, __global float* biases, int D2, int D1, int N)
        for (j = 0; j < 5; j++) {

            for (h = 0; h < CONV_LAYER_NUMS[j]; h++) {

                if (j == 0 && h == 0) {
                    err = clSetKernelArg(c_layer, 0, sizeof(cl_mem), image);
                    CHECK_ERROR(err);
                }
                else if (h == 0) {
                    err = clSetKernelArg(c_layer, 0, sizeof(cl_mem), &P[j - 1]);
                    CHECK_ERROR(err);
                }
                else {
                    err = clSetKernelArg(c_layer, 0, sizeof(cl_mem), &C[j][h - 1]);
                    CHECK_ERROR(err);

                }

                err = clSetKernelArg(c_layer, 1, sizeof(cl_mem), &C[j][h]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 2, sizeof(cl_mem), &W[j][h]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 3, sizeof(cl_mem), &B[j][h]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 4, sizeof(cl_int), &CONV_LAYERS_ARGS[j][0]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 5, sizeof(cl_int), &CONV_LAYERS_ARGS[j][1]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 6, sizeof(cl_int), &CONV_LAYERS_ARGS[j][2]);
                CHECK_ERROR(err);


            }


        }

        // Start Pooling Layer
        // pooling_layer(__global float* input, __global float* output, const int N, const int Nsquare)

        global_size3[0] = POOL_LAYER_SIZES[j][0];
        global_size3[1] = POOL_LAYER_SIZES[j][1];
        global_size3[2] = POOL_LAYER_SIZES[j][2];

        local_size3[0] = 4;
        local_size3[1] = 4;
        local_size3[2] = 4;

        clEnqueueNDRangeKernel(queue, p_layer, 3, NULL, global_size3, local_size3, 0, NULL, NULL);
        CHECK_ERROR(err);

        err = clFinish(queue);
        CHECK_ERROR(err);

    }

    // Start FC Layer
    // (__global float* input, __global float* output, __local float * l_sum, __global float* weights, biases, inDim, outDim)
    int N = 512;

    float* w[3] = { network[26], network[28], network[30] };
    float* b[3] = { network[27], network[29], network[31] };

    for (i = 0; i < 3; i++) {

        err = clSetKernelArg(f_layer, 0, sizeof(cl_mem), P + 5);
        if (i != 0) {
            err = clSetKernelArg(f_layer, 0, sizeof(cl_mem), FC + (i - 1));
            CHECK_ERROR(err);
        }

        err = clSetKernelArg(f_layer, 1, sizeof(cl_mem), FC + i);
        CHECK_ERROR(err);
        err = clSetKernelArg(f_layer, 2, sizeof(cl_float) * N, NULL);
        CHECK_ERROR(err);
        err = clSetKernelArg(f_layer, 3, sizeof(cl_float), &w[i]);
        CHECK_ERROR(err);
        err = clSetKernelArg(f_layer, 4, sizeof(cl_float), &b[i]);
        CHECK_ERROR(err);

        err = clSetKernelArg(f_layer, 5, sizeof(int), &N);
        CHECK_ERROR(err);

        err = clSetKernelArg(f_layer, 6, sizeof(int), FC_LAYER_SIZES + i);
        CHECK_ERROR(err);

        global_size2[0] = N;
        global_size2[1] = FC_LAYER_SIZES[i];
        local_size2[0] = 64;
        local_size2[1] = 1;

        clEnqueueNDRangeKernel(queue, f_layer, 2, NULL, global_size2, local_size2, 0, NULL, NULL);
        CHECK_ERROR(err);
        err = clFinish(queue);
        CHECK_ERROR(err);

    }

    err = clEnqueueReadBuffer(queue, FC[3], CL_TRUE, 0, sizeof(float) * FC_LAYER_SIZES[3], fc3, 0, NULL, NULL);
    CHECK_ERROR(err);

    softmax(fc3, 10);
    labels[i] = find_max(fc3, 10);
    confidences[i] = fc3[labels[i]];

    for (i = 0; i < 5; i++) {

        for (j = 0; j < CONV_LAYER_NUMS[i]; j++) {

            clReleaseMemObject(W[i][j]);
            clReleaseMemObject(B[i][j]);
            if (0 < i)
                err = clReleaseMemObject(C[i][j]);
            if (i == 0)
                err = clReleaseMemObject(FC[j]); // i == 0 일때 1~3 까지 총 3번 실행됨

        }

        clReleaseMemObject(P[i]);
    }

    clReleaseKernel(c_layer);
    clReleaseKernel(p_layer);
    clReleaseKernel(f_layer);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(device);
    free(platform);

}
*/