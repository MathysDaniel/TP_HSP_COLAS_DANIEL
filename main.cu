#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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
            printf("%f     ", *(M + x * p + y));
        }
        printf(")\n");
    }
}

// --------------------- Addition in C ------------------------------

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            *(Mout + i * p + j) = *(M1 + i * p + j) + *(M2 + i * p + j);
        }
    }
}


// --------------------- Addition with CUDA ------------------------------

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int row = threadIdx.x;    // Fetch the block index as row
    int col = threadIdx.y;   // Fetch the thread index as column

    if (row < n && col < p) {
        int index = row * p + col;  // Calculate the index in the flattened array

        // Perform addition
        Mout[index] = M1[index] + M2[index];
    }
}


// --------------------- Multiplication in C ------------------------------

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
    }
}

// --------------------- Multiplication with CUDA ------------------------------

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.x;    
    int col = threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += M1[row * n + i] * M2[i * n + col];
        }
        Mout[row * n + col] = sum;
    }
}

int main(void){
// ----------------------- Variable Definition -----------------------------------
    int n = 5;
    int p = 5;
    float *M = (float *)malloc(n * p * sizeof(float));

    float *CPU_M1 = (float *)malloc(n * p * sizeof(float));
    float *CPU_M2 = (float *)malloc(n * p * sizeof(float));
    float *CPU_Mout = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M, n, p);

    MatrixPrint(M, n, p);

    clock_t start_cpu, end_cpu;
    double cpu_time_used;

// --------------------- Addition with CPU ----------------------------------------
    start_cpu = clock();
    MatrixAdd(CPU_M1, CPU_M2, CPU_Mout, n, p);
    end_cpu = clock();
    cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    printf("Time taken for CPU addition: %f seconds\n", cpu_time_used);

    free(CPU_M1);
    free(CPU_M2);
    free(CPU_Mout);

// ---------------------- Addition with GPU ----------------------------------------
    float *GPU_M1, *GPU_M2, *GPU_Mout;
    cudaMalloc((void **)&GPU_M1, n * n * sizeof(float));
    cudaMalloc((void **)&GPU_M2, n * n * sizeof(float));
    cudaMalloc((void **)&GPU_Mout, n * n * sizeof(float));

    clock_t start_cpu2, end_cpu2;
    double gpu_time_used;

    dim3 dimGridAdd(n,1,1);   
    dim3 dimBlockAdd(p, 1, 1);

    start_cpu2 = clock();
    cudaMatrixAdd<<<dimGridAdd,dimBlockAdd>>>(GPU_M1, GPU_M2, GPU_Mout, n, p);
    cudaDeviceSynchronize();
    end_cpu2 = clock();
    gpu_time_used = ((double)(end_cpu2 - start_cpu2)) / CLOCKS_PER_SEC;

    printf("Time taken for GPU addition: %f seconds\n", gpu_time_used);


    cudaFree(GPU_M1);
    cudaFree(GPU_M2);
    cudaFree(GPU_Mout);

    float *CPU_M3 = (float *)malloc(n * p * sizeof(float));
    float *CPU_M4 = (float *)malloc(n * p * sizeof(float));
    float *CPU_Mout2 = (float *)malloc(n * p * sizeof(float));

// -------------------------- Multiplication with CPU -----------------------------------

    clock_t start_cpu3, end_cpu3;
    double cpu_time_used2;

    start_cpu3 = clock();
    MatrixMult(CPU_M3, CPU_M4, CPU_Mout2, n);
    end_cpu3 = clock();
    cpu_time_used2 = ((double)(end_cpu3 - start_cpu3)) / CLOCKS_PER_SEC;

    printf("Time taken for CPU multiplication: %f seconds\n", cpu_time_used2);
    free(CPU_M3);
    free(CPU_M4);
    free(CPU_Mout2);

// -------------------------- Multiplication with GPU ------------------------------------
    float *GPU_M3, *GPU_M4, *GPU_Mout2;
    cudaMalloc((void **)&GPU_M3, n * n * sizeof(float));
    cudaMalloc((void **)&GPU_M4, n * n * sizeof(float));
    cudaMalloc((void **)&GPU_Mout2, n * n * sizeof(float));

    dim3 dimGridMult(1);   
    dim3 dimBlockMult(n, p);

    clock_t start_cpu4, end_cpu4;
    double gpu_time_used2;

    start_cpu4 = clock();
    cudaMatrixMult<<<dimGridMult, dimBlockMult>>>(GPU_M3, GPU_M4, GPU_Mout2, n);
    cudaDeviceSynchronize();
    end_cpu4 = clock();
    gpu_time_used2 = ((double)(end_cpu4 - start_cpu4)) / CLOCKS_PER_SEC;

    printf("Time taken for GPU multiplication: %f seconds\n", gpu_time_used2);

    cudaFree(GPU_M3);
    cudaFree(GPU_M4);
    cudaFree(GPU_Mout2);

    free(M);
    return 0;
}