__kernel void dense_mul(__global float *input_matrix,
                        __global float *weights,
                        __global float *output_matrix){
    int j = get_global_id(0);
    int i = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    output_matrix[i] += input_matrix[j]+weights[i*width+j];
    
}

__kernel void dense_add(__global float *input_matrix,
                        __global float *weights,
                        __global float *output_matrix){
    int i = get_global_id(0);
    int j = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    output_matrix[j] += input_matrix[i]+weights[j*height+i];

    printf("%.2f+%.2f = %.2f \n", input_matrix[j],
            weights[i*height+j], output_matrix[i]);
    

    // printf("%d - %d = %f\n", i, j, input_matrix[j]*weights[i*width+j]);
    // printf("Bla %f - %f\n", input_matrix[i], weights[i*width+j]);
}

__kernel void float_array(__global float *input_matrix){
    int i = get_global_id(0);

    printf("%.2f*%.2f = %.2f \n", input_matrix[i],
            input_matrix[i], input_matrix[i]*input_matrix[i]);
    

}
