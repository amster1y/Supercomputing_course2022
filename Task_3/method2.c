#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

const int N = 20000;
const double EPS = 1e-10;
const double TAU = 0.00001;


double norm(double* v, int N) {
    double result = 0;
    for (int i = 0; i < N; i++)
        result += v[i]*v[i];
    return result;
}


double* multiplication(double* matrix, double* vector, double* result, int cols, int rows, int shift, int N){
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i] += matrix[i*N + (j + shift)]*vector[j];
    return result;
};


int main(int argc, char* argv[]) {
    double start_time = MPI_Wtime();
    MPI_Init (&argc, &argv);
    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    int shift_size = ((rank < N % size) ? 1 : 0);

    double* A = (double*)malloc(N*(N/size + shift_size)*sizeof(double));
    double* b = (double*)malloc((N/size + shift_size)*sizeof(double));
    double* x = (double*)malloc((N/size + 1)*sizeof(double));
    double* x_new = (double*)malloc((N/size + 1)*sizeof(double));

    memset(x, 0, (N/size + 1)*sizeof(double));
    memset(x_new, 0, (N/size + 1)*sizeof(double));

    int start_idx = 0;

    for(int i = 0; i < rank; i++)
        start_idx += N/size + (i < N % size ? 1 : 0);

    int* lengths = (int*)malloc(size*sizeof(int));
    memset(lengths, 0, size*sizeof(int));
    for (int i = 0; i < size; i++)
        lengths[i] = N/size + (i < N % size ? 1 : 0);

    int* positions = (int*)malloc(size*sizeof(int));
    memset(positions, 0, size*sizeof(int));
    int pos = 0;
    for (int i = 1; i < size; i++){
        pos += N/size + ((i - 1) < N % size ? 1 : 0);
        positions[i] = pos;
    }

    for (int i = 0; i < N/size + shift_size; i++) {
        for (int j = 0; j < N; j++) {
            if ((i*N + j - start_idx) % (N+1) == 0) 
                A[i*N + j] = 2;
            else
                A[i*N + j] = 1;
        }
    }

    for(int i = 0; i < N/size + shift_size; i++)
        b[i] = N + 1;

    double norm_b_thread = norm(b, N/size + shift_size);
    double norm_b;

    MPI_Allreduce(&norm_b_thread, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double norm_delta;
    double c;
    int shift_new = 0;
    int rank_new = 0;
    double* changer;
    double* delta = (double*)malloc((N/size + shift_size)*sizeof(double));

    for(;;) {
        memset(delta, 0, (N/size + shift_size)*sizeof(double));

        shift_new = shift_size;

        for(int i = 0; i < size; i++){
            multiplication(A, x, delta, N/size + shift_new, N/size + shift_size, start_idx, N);
            MPI_Sendrecv(x, N/size + 1, MPI_DOUBLE, (rank + 1) % size, 0, x_new, N/size + 1, MPI_DOUBLE, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, 0);
            changer = x;
            x = x_new;
            x_new = changer;
            shift_new = ((rank - 1 - i + size) % size < N % size) ? 1 : 0;
            rank_new = (rank - 1 - i + size) % size;
            start_idx = 0;
            for(int j = 0; j < rank_new; j++)
                start_idx += N/size + (j < N % size ? 1 : 0);
        }

        for (int i = 0; i < N/size + shift_size; i++) {
            delta[i] -= b[i];
            x[i] = x[i] - TAU*delta[i];
        }

        norm_delta = norm(delta, N/size + shift_size);
        MPI_Allreduce(&norm_delta, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        printf("%f\n", c);
        if(c/norm_b < EPS*EPS)
            break;
    }

    double* result = (double*)malloc(N*sizeof(double));
    MPI_Allgatherv(x, N/size + shift_size, MPI_DOUBLE, result, lengths, 
                   positions, MPI_DOUBLE, MPI_COMM_WORLD);

    // if (rank == 0) {
    //     for (int i = 0; i < N; i++) 
    //         printf("%f ", result[i]);
    //     printf("\n");
    // }

    double end_time_thread = MPI_Wtime() - start_time;
    double end_time;

    MPI_Reduce(&end_time_thread, &end_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank == 0){
        printf("Threads = %d\nExecution time = %f\n", size, end_time);
    }

    free(A);
    free(b);
    free(delta);
    free(x);
    free(x_new);
    free(lengths);
    free(positions);
    free(result);

    MPI_Finalize();

    return 0;
}