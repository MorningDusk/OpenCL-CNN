#ifndef _CNN_H
#define _CNN_H

void cnn_init(void);
void cnn(float* images, float** network, int* labels, float* confidences, int num_images);

static void pooling2x2(float* input, float* output, int N);
static void pooling_layer(float* inputs, float* outputs, int D, int N);
static void convolution3x3(float* input, float* output, float* filter, int N);
#define ReLU(x) (((x)>0)?(x):0);
static void convolution_layer(float* inputs, float* outputs, float* filters,
    float* biases, int D2, int D1, int N);
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N);
static void softmax(float* output, int N);
static int find_max(float* fc, int N);
float* alloc_layer(size_t n);


#endif 
