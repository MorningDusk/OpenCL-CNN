__kernel fclayer1(__global float* input, __global float* output, __global float* weights, 
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