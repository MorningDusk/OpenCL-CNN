#pragma warning(disable:4996)
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

    char* get_source_code(const char* file_name, size_t * len) {
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

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
float* alloc_layer_sq(size_t n) {
    return (float*)malloc(n * sizeof(float));
}

#define ReLU(x) (((x)>0)?(x):0)
void convolution_layer_seq(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {

    int i, j, k, l;

    memset(outputs, 0, sizeof(float) * N * N * D2);

    for (j = 0; j < D2; j++) {
        for (i = 0; i < D1; i++) {
            float* input = inputs + N * N * i;
            float* output = outputs + N * N * j;
            float* filter = filters + 3 * 3 * (j * D1 + i);

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
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
    }

    for (i = 0; i < D2; i++) {
        float* output = outputs + N * N * i;
        float bias = biases[i];
        for (j = 0; j < N * N; j++) {
            output[j] = ReLU(output[j] + bias);
        }
    }

}

void softmax(float* output, int N) {
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

int find_max(float* fc, int N) {
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

time_t start;

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
    size_t source_code_len;
    char* source_code = get_source_code("kernel.cl", &source_code_len);

    program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_code_len, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "-cl-no-signed-zeros -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only", NULL, NULL);
    CHECK_BUILD_ERROR(err);
    CHECK_ERROR(err);

    /* CREATE KERNEL*/
    c_layer = clCreateKernel(program, "convolution_layer", &err);
    CHECK_ERROR(err);

    p_layer = clCreateKernel(program, "pooling_layer", &err);
    CHECK_ERROR(err);

    f_layer = clCreateKernel(program, "fc_layer", &err);
    CHECK_ERROR(err);

}

void cnn(float* images, float* network, int* labels, float* confidences, int num_images) {

    int i, j, h;

    const int CONV_LAYER_NUMS[] = { 2, 2, 3, 3, 3 };

    const int CONV_LAYER_SIZES[5] = {
        64 * 32 * 32,
        128 * 16 * 16,
        256 * 8 * 8,
        512 * 4 * 4,
        512 * 2 * 2
    };

    const int CONV_LAYERS_ARGS[5][3] = { // D2, D1, N
        { 64, 3, 32 },
        { 128, 64, 16 },
        { 256, 128, 8 },
        { 512, 256, 4 },
        { 512, 512, 2 }
    };

    const int POOL_LAYER_SIZES[5][3] = { // DIM, N, N
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

    size_t global_size, global_size2[2], global_size3[3];
    size_t local_size, local_size2[2], local_size3[3];

    cl_mem W[5][3], B[5][3];
    cl_mem w[3], b[3];
    cl_mem C[5][3];
    cl_mem P[5];
    cl_mem FC[3];

    // Create Weight and bias buffers; Buffers are validated.

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

    // Make a networkBuffer for Debug 
    cl_mem networkBuffer;
    networkBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 60980520, network, &err);
    CHECK_ERROR(err);

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
    cl_mem imageBuffer;

    imageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 3 * 3 * 32, NULL, &err);
    CHECK_ERROR(err);

    float fc3[10];
    float * conv;
    float* p;

    for (i = 0; i < num_images; ++i) {
        
        int offset = 0, cnt = 0;    
        float* image = images + i * 3 * 32 * 32;

        err = clEnqueueWriteBuffer(queue, imageBuffer, CL_TRUE, 0, sizeof(float) * 3 * 3 * 32, image, 0, NULL, NULL);
        CHECK_ERROR(err);

        // Start Convolution Layer
        for (j = 0; j < 5; j++) {
            
            for (h = 0; h < CONV_LAYER_NUMS[j]; h++) {
                 conv = alloc_layer_sq(CONV_LAYER_SIZES[j]);
                // start = clock();
                // convolution_layer
                // inputs, outputs, networks, filterOffset, D2, D1, N, Nsquare
                // input = N * N * inDim
                if (j == 0 && h == 0) {
                    err = clSetKernelArg(c_layer, 0, sizeof(cl_mem), &imageBuffer);
                    CHECK_ERROR(err);

                    err = clSetKernelArg(c_layer, 5, sizeof(int), &CONV_LAYERS_ARGS[j][1]);
                    CHECK_ERROR(err);

                    global_size = CONV_LAYERS_ARGS[j][1] * CONV_LAYERS_ARGS[j][2] * CONV_LAYERS_ARGS[j][2];
                    local_size = CONV_LAYERS_ARGS[j][1];

                }
                else if (h == 0) {
                    err = clSetKernelArg(c_layer, 0, sizeof(cl_mem), &P[j - 1]);
                    CHECK_ERROR(err);

                    err = clSetKernelArg(c_layer, 5, sizeof(int), &CONV_LAYERS_ARGS[j][1]);
                    CHECK_ERROR(err);

                    global_size = CONV_LAYERS_ARGS[j][0] * CONV_LAYERS_ARGS[j][2] * CONV_LAYERS_ARGS[j][2];
                    local_size = CONV_LAYERS_ARGS[j][0];

                }
                else {
                    err = clSetKernelArg(c_layer, 0, sizeof(cl_mem), &C[j][h - 1]);
                    CHECK_ERROR(err);

                    err = clSetKernelArg(c_layer, 5, sizeof(int), &CONV_LAYERS_ARGS[j][0]);
                    CHECK_ERROR(err);

                    global_size = CONV_LAYERS_ARGS[j][0] * CONV_LAYERS_ARGS[j][2] * CONV_LAYERS_ARGS[j][2];
                    local_size = CONV_LAYERS_ARGS[j][0];
                }

                err = clSetKernelArg(c_layer, 1, sizeof(cl_mem), &C[j][h]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 2, sizeof(cl_mem), &networkBuffer);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 3, sizeof(int), &offset);
                CHECK_ERROR(err);
                offset += (NETWORK_SIZES[cnt++] + NETWORK_SIZES[cnt++]);

                err = clSetKernelArg(c_layer, 4, sizeof(int), &CONV_LAYERS_ARGS[j][0]);
                CHECK_ERROR(err);

                err = clSetKernelArg(c_layer, 6, sizeof(int), &CONV_LAYERS_ARGS[j][2]);
                CHECK_ERROR(err);

                int Nsquare = CONV_LAYERS_ARGS[j][2] * CONV_LAYERS_ARGS[j][2];
                err = clSetKernelArg(c_layer, 7, sizeof(int), &Nsquare);
                CHECK_ERROR(err);

                clEnqueueNDRangeKernel(queue, c_layer, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
                CHECK_ERROR(err);

                err = clFinish(queue);
                CHECK_ERROR(err);

                printf("image%d [%d][%d] time : %lf\n", i, j, h, (clock() - start) / (double)CLOCKS_PER_SEC);

                err = clEnqueueReadBuffer(queue, C[j][h], CL_TRUE, 0, sizeof(float) * CONV_LAYER_SIZES[j], conv, 0, NULL, NULL);
                CHECK_ERROR(err);

                // Debug Only
                //for (int g = 0; g < 5; g++) {
                //    printf(" %.8f ", conv[g]);
                //}
                
                //exit(0);
            }

            // Start Pooling Layer
            // pooling_layer(__global float* input, __global float* output, const int N, const int Nsquare)
            // printf("====Excute Pooling Max====\n");
            p = alloc_layer_sq(POOL_LAYER_SIZES[j][0] * POOL_LAYER_SIZES[j][1] * POOL_LAYER_SIZES[j][2]);

            err = clSetKernelArg(p_layer, 0, sizeof(cl_mem), &C[j][h - 1]);
            CHECK_ERROR(err);
            err = clSetKernelArg(p_layer, 1, sizeof(cl_mem), &P[j]);
            CHECK_ERROR(err);
            err = clSetKernelArg(p_layer, 2, sizeof(int), &POOL_LAYER_SIZES[j][1]);
            CHECK_ERROR(err);

            int Nsquare = POOL_LAYER_SIZES[j][1] * POOL_LAYER_SIZES[j][1];
            err = clSetKernelArg(p_layer, 3, sizeof(int), &Nsquare);
            CHECK_ERROR(err);

            global_size3[0] = POOL_LAYER_SIZES[j][0];
            global_size3[1] = POOL_LAYER_SIZES[j][1];
            global_size3[2] = POOL_LAYER_SIZES[j][2];

            local_size3[0] = 4;
            local_size3[1] = 4;
            local_size3[2] = 4;

            start = clock();
            clEnqueueNDRangeKernel(queue, p_layer, 3, NULL, global_size3, local_size3, 0, NULL, NULL);
            CHECK_ERROR(err);

            err = clEnqueueReadBuffer(queue, P[j], CL_FALSE, 0, sizeof(float) * POOL_LAYER_SIZES[j][0] * POOL_LAYER_SIZES[j][1] * POOL_LAYER_SIZES[j][2],
                p, 0, NULL, NULL);
            CHECK_ERROR(err);

            err = clFinish(queue);
            CHECK_ERROR(err);
            //printf("time : %lf\n", (clock() - start) / (double)CLOCKS_PER_SEC);

            for (int i = 0; i < 50; i++) {
                printf(" %.4f ", p[i]);
            }
            putchar('\n');
        
        }

        // Start FC Layer 
        // 
        // (input, output, networks, offset, inDim, outDim)
        int N = 512;

        for (int k = 0; k < 3; k++) {

            err = clSetKernelArg(f_layer, 0, sizeof(cl_mem), &P[4]);
            if (k != 0) {
                err = clSetKernelArg(f_layer, 0, sizeof(cl_mem), &FC[k - 1]);
                CHECK_ERROR(err);
            }

            err = clSetKernelArg(f_layer, 1, sizeof(cl_mem), &FC[k]);
            CHECK_ERROR(err);
            err = clSetKernelArg(f_layer, 2, sizeof(cl_mem), &networkBuffer);
            CHECK_ERROR(err);
            err = clSetKernelArg(f_layer, 3, sizeof(int), &offset);
            CHECK_ERROR(err);
            offset += (NETWORK_SIZES[cnt++] + NETWORK_SIZES[cnt++]);
            err = clSetKernelArg(f_layer, 4, sizeof(int), &N);
            CHECK_ERROR(err);
            err = clSetKernelArg(f_layer, 5, sizeof(int), &FC_LAYER_SIZES[k]); // diff
            CHECK_ERROR(err);

            global_size2[0] = N;
            global_size2[1] = FC_LAYER_SIZES[k];
            local_size2[0] = N;
            local_size2[1] = 1;

            clEnqueueNDRangeKernel(queue, f_layer, 2, NULL, global_size2, local_size2, 0, NULL, NULL);
            CHECK_ERROR(err);

            err = clFinish(queue);
            CHECK_ERROR(err);
            
            float fc0[512];
            err = clEnqueueReadBuffer(queue, FC[0], CL_TRUE, 0, sizeof(float) * FC_LAYER_SIZES[0], fc0, 0, NULL, NULL);
            CHECK_ERROR(err);

            for (int w = 0; w < 512; w++) {
                printf(" %.4f ", fc0[w]);
            }
            
            exit(0);
        }

        err = clEnqueueReadBuffer(queue, FC[2], CL_TRUE, 0, sizeof(float) * FC_LAYER_SIZES[2], fc3, 0, NULL, NULL);
        CHECK_ERROR(err);

        softmax(fc3, 10);
        labels[i] = find_max(fc3, 10);
        confidences[i] = fc3[labels[i]];
    
        // Test Print
        printf("labels[%d] : %d | confidences[%d] = %f\n", i, labels[i], i, confidences[i]);
    }


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