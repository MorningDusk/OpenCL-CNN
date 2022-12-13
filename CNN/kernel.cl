#define ReLU(x) (((x)>0)?(x):0)
__kernel void convolution_1 (__global float *inputs, const int imageOffset, __global float *outputs, const int outDim, const int N){
	
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
            if ((0 <= x && x < N) && (0 <= y && y < N))
                output[((dim * 3 * 3) + (3 * i + j)) * (N * N) + (row * N + col)] = input[((dim * N) + y) * N + x];
            else
                output[((dim * 3 * 3) + (3 * i + j)) * (N * N) + (row * N + col)] = 0.0f;
        }
    }
}

__kernel void convolution_2 (__global float *inputs, __global float *outputs, __global float *networks, const int networkOffset, const int inDim, const int outDim, const int N) {
    
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int g_row = get_group_id(0) * 16 + row;
    const int g_col = get_group_id(1) * 16 + col;

    __global float* input = inputs;
    __global float* output = outputs;
    __global float * filter = networks + networkOffset;
    __global float * biases = networks + networkOffset + (inDim * outDim * 9);
    __local float filterSub[16][16];
    __local float inputSub[16][16];

    int i, j;
    int ROW_A = outDim; 
    int ROW_B = inDim * 3 * 3;
    int COL_A = inDim * 3 * 3;
    int COL_B = N * N;

    float sum = 0.0f;

    #pragma unroll
    for (i = 0; i < COL_A; i += 16) {
        const int temp_col = i + col;
        const int temp_row = i + row;

        if (g_col < outDim && temp_row < COL_A)
            filterSub[col][row] = filter[g_col * COL_A + temp_row];
        else
            filterSub[col][row] = 0;
        
        if (temp_col < ROW_B&& g_row < COL_B)
            inputSub[col][row] = input[temp_col * COL_B + g_row];
        else
            inputSub[col][row] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);   

        #pragma unroll
        for (j = 0; j < 16; j++) {
            sum += filterSub[col][j] * inputSub[j][row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_col < ROW_A && g_row < COL_B) {
        sum += biases[g_col];
        output[g_col * COL_B + g_row] = ReLU(sum);
    }
}


__kernel void pooling_max (__global float *inputs, __global float *outputs, const int inDim, const int N) {
    
    const int g_i=get_global_id(0);
	const int GROUP_J=get_global_id(1);

    __global float *input, * output;
    
    int i, j;
    int group_num=(g_i/N);
	int frow=GROUP_J%N;
	int fcol=g_i%N;
	float max=0.0f;

    input=inputs+(N*N)*4*group_num;
	output=outputs+(N * N)*group_num;


	#pragma unroll
	for(int i=0; i<2;i++){
		#pragma unroll
		for(int j=0;j<2;j++){
			float temp=input[(N * 2)*(2*frow+i)+(2*fcol+j)];
            if (max < temp)
                max = temp;
		}
	}

	output[N*frow+fcol]=max;

}


__kernel void fc_layer (__global float *inputs, __global float *outputs, __global float *networks, const int networkOffset, const int inDim, const int outDim) {
    
    __global float *weights = networks + networkOffset;
	__global float *biases = networks + networkOffset;

    int TS=(outDim!=10?16:2);
    int l_i=get_local_id(0);
    int output_group=get_group_id(0)*TS+l_i;
    
    float sum=0.0f;
    weights += inDim * output_group;
    biases += (inDim * outDim) + output_group;

	if(output_group>=outDim) return;
	
	#pragma unroll
	for(int i=0;i<inDim;i++) {
		sum += inputs[i] * weights[i];
	}
	sum += biases[0];
	outputs[output_group] = ReLU(sum);
}