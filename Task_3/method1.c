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
    int i;
    for (i = 0; i < N; i++)
        result += v[i]*v[i];
    return result;
}


int main(int argc, char* argv[]) {

    double start = MPI_Wtime();
    int size,rank;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int shift_size = (rank < N%size)? 1:0;
    double* A = (double*)malloc(N*(N/size + shift_size)*sizeof(double));
    double* b = (double*)malloc((N)*sizeof(double));
    double* x = (double*)malloc(N*sizeof(double));

    int i, j;

    int start_idx = 0;
    for (i = 0; i < rank; i++)
        start_idx += N/size + (i < N % size? 1:0);

    for (i = 0; i < N/size + shift_size; i++) {
        for (j = 0; j < N; j++) {
            if ((i*N + j - start_idx) % (N+1) == 0) 
                A[i*N + j] = 2;
            else
                A[i*N + j] = 1;
        }
    }

    for (i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0;
    }

    int* lengths = (int*)malloc(size*sizeof(int));
    memset(lengths, 0, size*sizeof(int));
    for (i = 0; i < size; i++)
        lengths[i] = N/size + (i < N % size ? 1 : 0);

    int* positions = (int*)malloc(size*sizeof(int));
    memset(positions, 0, size*sizeof(int));
    int pos = 0;
    for (i = 1; i < size; i++){
        pos += N/size + ((i - 1) < N % size ? 1 : 0);
        positions[i] = pos;
    }
    
    double criterion = 1000;
    double c = 0;
    double norm_b = norm(b, N);
    double* delta = (double*)malloc((N/size + shift_size)*sizeof(double));

    while (1) {
        for (i = 0; i < N/size + shift_size; i++) {
            delta[i] = 0;
            for (j = 0; j < N; j++) {
                delta[i] += A[i*N + j] * x[j];
            }
        }    

        for (i = 0; i < N/size + shift_size; i++) {
            delta[i] -= b[i + start_idx];
            x[i + start_idx] = x[i + start_idx] - TAU*delta[i];
        }

        double norm_delta = norm(delta, N/size + shift_size);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&norm_delta, &c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        printf("%f\n", c);
        criterion = c/norm_b;
        if (criterion < EPS*EPS) {
            break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(x, N/size + shift_size, MPI_DOUBLE, x, lengths, 
                   positions, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_thread = MPI_Wtime() - start;
    double end;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&end_thread, &end, 1, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);
    if(rank == 0){
        for (i = 0; i < N; i++) 
            printf("%f ", x[i]);
        printf("\nThreads = %d\nExecution time = %f\n", size, end);
    }

    free(A);
    free(b);
    free(x);
    free(delta);
    free(lengths);
    free(positions);

    MPI_Finalize();

    return 0;
}