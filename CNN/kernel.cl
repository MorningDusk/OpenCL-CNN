__kernel void fclayer1(__global float* input, __global float* output, __global float* weights, 
    __global float* biases, const int outDim, const int inDim)
{
    int loc = get_group_id(0) * outDim + get_local_id(0);
    float sum = .0f;

    for (int in = 0; in < inDim; ++in)
        sum += input[in] * *(weights++);

    sum += biases[loc];
    if (sum > 0) output[loc] = sum;
    else output[loc] = 0;
}

__kernel void pooling_layer(__global float* input, __global float* output, const int N, const int Nsquare)
{
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_z = get_global_id(2);
    int temp;
    int max = 0;


    for (int y = 0; y < 2; y++)
    {
        for (int x = 0; x < 2; x++)
        {
            temp = input[Nsquare * pos_z + N * (pos_y + y) + pos_x + x];
            if (max < temp) max = temp;
        }
    }
}

/*
__kernel fclayer2(__global float* input, __global float* output, __global float* biases
        __global float* biases, const int outDim, const int inDim)
{
    int loc = get_group_id(0) * outDim + get_local_id(0);
    float sum = .0f;

    for (int in = 0; in < inDim; ++in)
    {
        sum += input[in] * *(weights++);
    }

    sum += biases[loc];
    if (sum > 0) output[loc] = sum;
    else output[loc] = 0;
}
*/