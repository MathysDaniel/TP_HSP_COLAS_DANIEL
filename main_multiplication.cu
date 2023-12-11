#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define DEBUG 0

// --------------------- Initialization of Matrix ----------------------

void MatrixInit(float *M, int n, int p){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < p; j++){
            *(M + i * p + j) = (float)(rand() % 200) / 100 - 1;
        }
    }
}

// --------------------- Printing Matrix ------------------------------

void MatrixPrint(float *M, int n, int p){
    for(int x = 0 ; x < n ; x++) {
        printf(" (");
        for(int y = 0 ; y < p ; y++){
            printf("%f ", *(M + x * p + y));
        }
        printf(")\n");
    }
    printf("\n");
}

// --------------------- Multiplication in C ------------------------------

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
    }
}

// --------------------- Multiplication with CUDA ------------------------------

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, int p) {
    int row = blockIdx.x;    
    int col = threadIdx.x;

    if (row < n && col < p) {
        float sum = 0.0;
        for (int i = 0; i < p; i++) {
            sum += M1[row * n + i] * M2[i * p + col];
        }
        Mout[row * n + col] = sum;
    }
}

int main(void){
// ----------------------- Variable Definition -----------------------------------
    
    // define cpu
    float *M1, *M2, *Mout, *Mout_gpu_cpu;
    int n = 2;
    int p = 2;
    int N = n*p;
    clock_t start_cpu, end_cpu;
    double time_used; 

    // define gpu
    float *M1_d, *M2_d, *Mout_d;


    // malloc
    M1 = (float *)malloc(N * sizeof(float));
    M2 = (float *)malloc(N * sizeof(float));
    Mout = (float *)malloc(N * sizeof(float));
    Mout_gpu_cpu = (float *)malloc(N * sizeof(float));

    // init sur cpu
    MatrixInit(M1, n, p);
    if (DEBUG == 1){
        MatrixPrint(M1, n, p);
    }
    

    MatrixInit(M2, n, p);
    if (DEBUG == 1){
        MatrixPrint(M2, n, p);
    }
    
    // calcul sur cpu
    start_cpu = clock();
    MatrixMult(M1, M2, Mout, n, p);
    end_cpu = clock();
    time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("Time taken for CPU multiplication: %f seconds\n", time_used);
    if (DEBUG == 1){
        MatrixPrint(Mout, n, p);
    }
    

    // malloc gpu
    cudaMalloc((void **)&M1_d, N * sizeof(float));
    cudaMalloc((void **)&M2_d, N * sizeof(float));
    cudaMalloc((void **)&Mout_d, N * sizeof(float));

    // cpu -> gpu
    cudaMemcpy(M1_d, M1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(M2_d, M2, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Mout_d, Mout, N * sizeof(float), cudaMemcpyHostToDevice);


    // kernel
    dim3 dimGridAdd(n, 1, 1);   
    dim3 dimBlockAdd(p, 1, 1);
    cudaMatrixMult<<<dimGridAdd,dimBlockAdd>>>(M1_d, M2_d, Mout_d, n, p);

    // gpu -> cpu
    cudaMemcpy(Mout_gpu_cpu, Mout_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (DEBUG == 1){
        MatrixPrint(Mout_gpu_cpu, n, p);
    }

    // verif
    for (int i=0; i<N; i++){
        if (abs(Mout[i] - Mout_gpu_cpu[i]) > 0.01){
            printf("error at %d\n", i);
            exit(1);
        }
    }
    printf("passed !\n");

    // free gpu
    cudaFree(M1_d);
    cudaFree(M2_d);
    cudaFree(Mout_d);

    // free 
    free(M1);
    free(M2);
    free(Mout);

}