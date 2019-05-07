
__kernel void add_calculated(
                        __global float *calculated_matrix,
                        __global float *output_matrix,
                        __global int *N_IN
                        )
{
    int j = get_global_id(0);
    int N_OUT = get_global_size(0);
    
    __private float accumulator = 0;
    for(int i = 0; i < *N_IN; i++){
        accumulator +=  calculated_matrix[j*(int)(*N_IN) + i];
    }

    output_matrix[j] = accumulator;
   
}
