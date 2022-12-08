#define ReLU(x) (((x)>0)?(x):0)
__kernel void convolution_layer(__global float* inputs, __global float* outputs, __global float* filters, __global float* biases, int D2, int D1, int N) {

	int i, j, k, l;

	for (j = 0; j < D2; j++) {
		for (i = 0; i < D1; i++) {
			__global float* input = inputs + N * N * i;
			__global float* output = outputs + N * N * j;
			__global float* filter = filters + 3 * 3 * (j * D1 + i);

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
	}

	for (i = 0; i < D2; i++) {

		__global float* output = outputs + N * N * i;
	
		for (j = 0; j < N * N; j++) {
			output[j] = ReLU(output[j] + biases[i]);
		}

	}

}

__kernel void fc_layer(__global float* input, __global float* output, __global float* weights, __global float* biases, const int inDim, const int outDim) {
    
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float l_sum = 0;

    l_sum += input[j] * weights[(i * inDim) + j];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(j == 0) 
        l_sum += biases[i];
        output[i] = (l_sum > 0) ? l_sum : 0;
    }
  
}

__kernel void pooling_layer(__global float* input, __global float* output, const int N, const int Nsquare)
{
    int pos_z = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_x = get_global_id(2);

    float temp;
    float max = .0f;

    __global float* inpt = input + pos_z * Nsquare * 4;
    __global float* oupt = output + pos_z * Nsquare;

    for (int y = 0; y < 2; y++)
    {
        for (int x = 0; x < 2; x++)
        {
            temp = inpt[(pos_y * 2 + y) * 2 * N + pos_x * 2 + x];
            if (max < temp) max = temp;
        }
    }

    oupt[pos_y * N + pos_x] = max;

} 
