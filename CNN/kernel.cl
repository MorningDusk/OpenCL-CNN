#define ReLU(x) (((x) > 0) ? (x) : 0)

__kernel void convolution_1 (__global float *inputs, const int imageOffset, __global float *outputs, const int outDim, const int N, const int batchCount) {
	
    const int GROUP_J = get_global_id(0);

    __global float* input = inputs + imageOffset;

    int i, j, x, y;
    
    int rows = GROUP_J / N; 
    int col = GROUP_J % N; 

    int dim = rows / N; 
    int row = rows - dim * N; 

    #pragma unroll
    for (i = 0; i < 3; i++) {

        #pragma unroll
        for (j = 0; j < 3; j++) {
            x = col + j - 1;
            y = row + i - 1;

            outputs[((dim * 3 * 3) + (3 * i + j)) * (N * N) + (row * N + col)] = (0 <= x && x < N && 0 <= y && y < N ? input[((dim * N) + y) * N + x] : 0);

        }

    }
}

__kernel void convolution_2 (__global float *inputs, __global float *outputs, __global float *networks, const int networkOffset, const int inDim, const int outDim, const int N) {
    
    const int ROW = get_local_id(0);
    const int COL = get_local_id(1);
    const int GROUP_ROW = get_group_id(0) * 16 + ROW;
    const int GROUP_COL = get_group_id(1) * 16 + COL;

    __global float* input = inputs;
    __global float* output = outputs;
    __global float * filters = networks + networkOffset;
    __global float * biases = networks + networkOffset + (inDim * outDim * 9);

    __local float filterSub[16][16];
    __local float inputSub[16][16];

    int i, j;

    int filterRow = outDim; 
    int filterCol = inDim * 3 * 3;

    int InputRow = inDim * 3 * 3;
    int InputCol = N * N;

    float sum = 0.0f;

    #pragma unroll
    for (i = 0; i < filterCol; i += 16) {

        const int TEMP_COL = i + COL;
        const int TEMP_ROW = i + ROW;

        filterSub[COL][ROW] = (GROUP_COL < outDim && TEMP_ROW < filterCol ? filters[GROUP_COL * filterCol + TEMP_ROW] : 0);
        inputSub[COL][ROW] = (TEMP_COL < InputRow && GROUP_ROW < InputCol ? input[TEMP_COL * InputCol + GROUP_ROW] : 0);
      
        barrier(CLK_LOCAL_MEM_FENCE);   

        #pragma unroll
        for (j = 0; j < 16; j++) {
            sum += inputSub[j][ROW] * filterSub[COL][j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    
    }

    if (GROUP_COL < filterRow && GROUP_ROW < InputCol) {
        sum += biases[GROUP_COL];
        output[GROUP_COL * InputCol + GROUP_ROW] = ReLU(sum);
    }

}


__kernel void pooling_max (__global float *inputs, __global float *outputs, const int inDim, const int N) {
    
    const int GROUP_I = get_global_id(0);
	const int GROUP_J = get_global_id(1);
    const int GROUP_NUM = GROUP_I / N;

    __global float *input, * output;
    
    int i, j;
    
	int frow = GROUP_J % N;
	int fcol = GROUP_I % N;

	float temp, max = 0.0f;
     
    input= inputs + (N * N) * 4 * GROUP_NUM;
	output= outputs + (N * N) * GROUP_NUM;

	#pragma unroll
	for(int i=0; i<2;i++) {

		#pragma unroll
		for(int j=0;j<2;j++) {
			temp = input[(N * 2) * (2 * frow + i) + (2 * fcol + j)];    
            if (max < temp) max = temp;
        }

	}

	output[(N * frow) + fcol] = max;

}


__kernel void fc_layer (__global float *inputs, __global float *outputs, __global float *networks, const int networkOffset, const int inDim, const int outDim) {
    
    const int TS = (outDim != 10? 16:2);
    const int LOCAL_ID = get_local_id(0);
    const int OUTPUT_GROUP = (get_group_id(0) * TS) + LOCAL_ID;

    __global float *weights = networks + networkOffset;
	__global float *biases = networks + networkOffset;
    
    float sum=0.0f;

    weights += inDim * OUTPUT_GROUP;
    biases += (inDim * outDim) + OUTPUT_GROUP;

	if (OUTPUT_GROUP >= outDim) return;
	
	#pragma unroll
	for(int i=0;i<inDim;i++)
        sum += inputs[i] * weights[i];
	
    sum += biases[0];
	outputs[OUTPUT_GROUP] = ReLU(sum);

}

__kernel void softmax(__global float* outputs, const int neuron_size)
{
    int i;
    float sum = 0;
    int begin_idx = get_local_id(0) * neuron_size;

    __global float* output = outputs + begin_idx;
    float max = output[0];
    
    for (i = 1; i < neuron_size; i++)
        max = (output[i] > max) ? output[i] : max;

    for (i = 0; i < neuron_size; i++)
        sum += exp(output[i] - max);

    for (i = 0; i < neuron_size; i++)
        output[i] = exp(output[i] - max) / sum;

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void find_max_and_conf(__global float* inputs, __global float* labels, __global float* confidences, const int classNum, const int batchOffset)
{
    int i = 0;
    int max_idx = 0;
    float max = 0;
    __global float* input = inputs + get_local_id(0) * classNum;

    for (i = 0; i < classNum; i++)
    {
        if (max < input[i])
        {
            max = input[i];
            max_idx = i;
        }
    }
    
    int loc = get_local_id(0) + batchOffset;
    max_idx += classNum * get_local_id(0);
    labels[loc] = max_idx;
    confidences[loc] = inputs[labels[loc]];

    barrier(CLK_LOCAL_MEM_FENCE);
}