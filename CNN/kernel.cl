#define ReLU(x) (((x)>0)?(x):0)

__kernel void convolution_1 
(__global float *inputs, const int imageOffset, __global float *outputs, const int outDim, const int N){
	
    const int GROUP_J = get_global_id(0);

    __global float* input = inputs + imageOffset;
    __global float* output = outputs;

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

            output[((dim * 3 * 3) + (3 * i + j)) * (N * N) + (row * N + col)] = (0 <= x && x < N && 0 <= y && y < N ? input[((dim * N) + y) * N + x] : 0);

        }

    }
}

__kernel void convolution_2 
(__global float *inputs, __global float *outputs, __global float *networks, const int networkOffset, const int inDim, const int outDim, const int N) {
    
    const int ROW = get_local_id(0);
    const int COL = get_local_id(1);
    const int GROUP_ROW = get_group_id(0) * 16 + ROW;
    const int GROUP_COL = get_group_id(1) * 16 + COL;

    __global float* input = inputs;
    __global float* output = outputs;
    __global float * filters = networks + networkOffset;
    __global float * biases = networks + networkOffset + (inDim * outDim * 9);

    __local float filter[16][16];
    __local float inputSub[16][16];

    int i, j;

    int rowA = outDim; 
    int colA = inDim * 3 * 3;

    int rowB = inDim * 3 * 3;
    int colB = N * N;

    float sum = 0.0f;

    #pragma unroll
    for (i = 0; i < colA; i += 16) {

        const int TEMP_ROW = i + ROW;
        const int TEMP_COL = i + COL;

        filter[ROW][COL] = (GROUP_ROW < outDim && TEMP_COL < colA ? filters[GROUP_ROW * colA + TEMP_COL] : 0);
        inputSub[ROW][COL] = (TEMP_ROW < rowB && GROUP_COL < colB ? input[TEMP_ROW * colB + GROUP_COL] : 0);
      
        barrier(CLK_LOCAL_MEM_FENCE);   

        #pragma unroll
        for (j = 0; j < 16; j++) {
            sum += inputSub[ROW][j] * filter[j][COL];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    
    }

    if (GROUP_ROW < rowA && GROUP_COL < colB) {
        sum += biases[GROUP_ROW];
        output[GROUP_ROW * colB + GROUP_COL] = ReLU(sum);
    }

}


__kernel void pooling_max 
(__global float *inputs, __global float *outputs, const int inDim, const int N) {
    
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
	for(int i=0; i<2;i++){

		#pragma unroll
		for(int j=0;j<2;j++){

			temp = input[(N * 2) * (2 * frow + i) + (2 * fcol + j)];    
            if (max < temp) max = temp;
		
        }

	}

	output[(N * frow) + fcol] = max;

}


__kernel void fc_layer 
(__global float *inputs, __global float *outputs, __global float *networks, const int networkOffset, const int inDim, const int outDim) {
    
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
	for(int i=0;i<inDim;i++) {
		
        sum += inputs[i] * weights[i];
	
    }
	
    sum += biases[0];
	outputs[OUTPUT_GROUP] = ReLU(sum);

}