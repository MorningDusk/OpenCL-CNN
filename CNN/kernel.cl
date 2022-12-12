#define ReLU(x) (((x)>0)?(x):0)
__kernel void convolution_layer (__global float* inputs, __global float* outputs, __constant float* networks,
								 int offset, int D2, int D1, int N, int Nsquare) {
    
    __global float *input, *output;
    __constant float* filters = networks + offset;
	__constant float* biases = filters + D2;

    __local float element[4608]; // 9 * 512
    __local float local_sum;

    // for parallel index
    int x, y;

    int channel = get_local_id(0);      // 채널 index
    int pixel = get_group_id(0);        // 픽셀 index

    // for variable devide input by matrix(3by3)
    const int matrixSize = 9;

    // insert element
	int row = pixel / N, col = pixel % N;
    input = inputs + (pixel + channel * N * N - N - 1);
	
    // (0, 0) ~ (0, 2)
	x = row - 1; y = col - 1;
	element[9 * channel] = (0 <= x && x < N && 0 <= y && y < N ? input[0] : 0);
	x = row - 1; y = col;
	element[9 * channel + 1] = (0 <= x && x < N && 0 <= y && y < N ? input[1] : 0);
	x = row - 1; y = col + 1;
	element[9 * channel + 2] = (0 <= x && x < N && 0 <= y && y < N ? input[2] : 0);
	
	x = row; y = col - 1;
	element[9 * channel + 3] = (0 <= x && x < N && 0 <= y && y < N ? input[N] : 0);
	x = row; y = col;
	element[9 * channel + 4] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 1] : 0);
	x = row; y = col + 1;
	element[9 * channel + 5] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 2] : 0);

	x = row + 1; y = col - 1;
	element[9 * channel + 6] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N] : 0);
	x = row + 1; y = col;
	element[9 * channel + 7] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 1] : 0);	
	x = row + 1; y = col + 1;
	element[9 * channel + 8] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 2] : 0);
	
	 
    if (256 < D1) {
        input = inputs + (pixel + (channel + 256) * N * N - N - 1);
        
        x = row - 1; y = col - 1;
		element[9 * (channel + 256)] = (0 <= x && x < N && 0 <= y && y < N ? input[0] : 0);
		
		x = row - 1; y = col;
		element[9 * (channel + 256) + 1] = (0 <= x && x < N && 0 <= y && y < N ? input[1] : 0);
		
		x = row - 1; y = col + 1;
		element[9 * (channel + 256) + 2] = (0 <= x && x < N && 0 <= y && y < N ? input[2] : 0);
		
		x = row; y = col - 1;
		element[9 * (channel + 256) + 3] = (0 <= x && x < N && 0 <= y && y < N ? input[N] : 0);
		
		x = row; y = col;
		element[9 * (channel + 256) + 4] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 1] : 0);

		x = row; y = col + 1;
		element[9 * (channel + 256) + 5] = (0 <= x && x < N && 0 <= y && y < N ? input[N + 2] : 0);
		
		x = row + 1; y = col - 1;
		element[9 * (channel + 256) + 6] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N] : 0);
		
		x = row + 1; y = col;
		element[9 * (channel + 256) + 7] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 1] : 0);
		
		x = row + 1; y = col + 1;
		element[9 * (channel + 256) + 8] = (0 <= x && x < N && 0 <= y && y < N ? input[2 * N + 2] : 0);
    }   

    float sum, result;
	__constant float * filter;

    for (int k = 0; k < D2; k++) {
        
        local_sum = sum = 0; 
        
        filter = filters + (9 * (k * D1 + channel));

        sum += (element[9 * channel] == 0 ? 0 : element[9 * channel] * filter[0]);
		sum += (element[9 * channel + 1] == 0 ? 0 : element[9 * channel + 1] * filter[1]);
		sum += (element[9 * channel + 2] == 0 ? 0 : element[9 * channel + 2] * filter[2]);
		sum += (element[9 * channel + 3] == 0 ? 0 : element[9 * channel + 3] * filter[3]);
		sum += (element[9 * channel + 4] == 0 ? 0 : element[9 * channel + 4] * filter[4]);
		sum += (element[9 * channel + 5] == 0 ? 0 : element[9 * channel + 5] * filter[5]);
		sum += (element[9 * channel + 6] == 0 ? 0 : element[9 * channel + 6] * filter[6]);
		sum += (element[9 * channel + 7] == 0 ? 0 : element[9 * channel + 7] * filter[7]);
		sum += (element[9 * channel + 8] == 0 ? 0 : element[9 * channel + 8] * filter[8]);

        if (256 < D1) {
            filter = filters + (9 * (k * D1 + channel + 256));

            sum += (element[9 * (channel + 256)] == 0 ? 0 : element[9 * (channel + 256)] * filter[0]);
			sum += (element[9 * (channel + 256) + 1] == 0 ? 0 : element[9 * (channel + 256) + 1] * filter[1]);
			sum += (element[9 * (channel + 256) + 2] == 0 ? 0 : element[9 * (channel + 256) + 2] * filter[2]);
			sum += (element[9 * (channel + 256) + 3] == 0 ? 0 : element[9 * (channel + 256) + 3] * filter[3]);
			sum += (element[9 * (channel + 256) + 4] == 0 ? 0 : element[9 * (channel + 256) + 4] * filter[4]);
			sum += (element[9 * (channel + 256) + 5] == 0 ? 0 : element[9 * (channel + 256) + 5] * filter[5]);
			sum += (element[9 * (channel + 256) + 6] == 0 ? 0 : element[9 * (channel + 256) + 6] * filter[6]);
			sum += (element[9 * (channel + 256) + 7] == 0 ? 0 : element[9 * (channel + 256) + 7] * filter[7]);
			sum += (element[9 * (channel + 256) + 8] == 0 ? 0 : element[9 * (channel + 256) + 8] * filter[8]);
        }

        local_sum += sum;
        result = local_sum + biases[k];

        output = outputs + (Nsquare * k);
        output[pixel] = ReLU(result);  
    }
	
}

__kernel void pooling_layer (__global float* input, __global float* output, const int N, const int Nsquare)
{
    int pos_z = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_x = get_global_id(2);

    float temp;
    float max = .0f;

    __global float* inpt = input + pos_z * Nsquare * 4;
    __global float* oupt = output + pos_z * Nsquare;

    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            temp = inpt[(pos_y * 2 + y) * 2 * N + pos_x * 2 + x];
            if (max < temp) max = temp;
        }
    }

    oupt[pos_y * N + pos_x] = max;
} 

__kernel void fc_layer 
	(__global float* input, __global float* output, __constant float* networks, const int offset, const int inDim, const int outDim) {
	
	__constant float* weights = networks + offset;
	__constant float* biases = weights + outDim;

    int i = get_global_id(0);
	int j = get_global_id(1);

	output[i] = input[i];
	
	int delta = (inDim * i);
	float sum = 0.0f;

	if(i >= outDim) return;

	__global float* inpt = input + delta;
	__constant float * weight = weights + delta;

	#pragma unroll
	for	(int size = 0 ; size < inDim; size++) {
		sum += inpt[size] * weight[size];
    }

	sum += biases[i];
	output[i] = sum; 
    output[i] = ReLU(sum);
	
}