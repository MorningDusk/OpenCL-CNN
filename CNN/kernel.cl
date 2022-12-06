__kernel void fc_layer(__global float* input, __global float* output, __global float* weights, 
__global float* biases, const int inDim, const int outDim) {
    
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