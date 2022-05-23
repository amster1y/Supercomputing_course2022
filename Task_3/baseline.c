#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const int N = 20000;
const double EPS = 1e-10;
const double TAU = 0.01;


double norm(double* v, int N) {
    double result = 0;
    for (int i = 0; i < N; i++)
        result += v[i]*v[i];
    return result;
}


int main(int argc, char* argv[]) {
    clock_t tStart = clock();
    int size,rank;

    double* A = (double*)malloc(N*N*sizeof(double));
    double* x = (double*)malloc(N*sizeof(double));
    double* b = (double*)malloc(N*sizeof(double));

    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j)
                A[i*N + j] = 2;
            else
                A[i*N + j] = 1;
        }
    }

    double criterion = EPS + 1;

    while (criterion >= EPS*EPS) {
        double* delta = (double*)malloc(N*N*sizeof(double));
        for (int i = 0; i < N; i++) {
            delta[i] = 0;
            for (int j = 0; j < N; j++) {
                delta[i] += A[i*N + j] * x[j];
            }
            delta[i] -= b[i];
            x[i] = x[i] - TAU*delta[i];
        }
        criterion = norm(delta, N)/norm(b, N);

        free(delta);
    }

    for (int i = 0; i < N; i++) 
        printf("%f ", x[i]);

    free(A);
    free(b);
    free(x);

    printf("\nExecution time: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    return 0;
}