__kernel void weights_mul(__global float *input_matrix,
                        __global float *weights,
                        __global float *calculated_matrix)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int N_IN = get_global_size(0);
    int N_OUT = get_global_size(1);

    calculated_matrix[j*N_IN + i] = input_matrix[i]*weights[j*N_IN + i];
    
}
