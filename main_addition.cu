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
    MatrixAdd(M1, M2, Mout, n, p);
    end_cpu = clock();
    time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("Time taken for CPU addition: %f seconds\n", time_used);
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
    cudaMatrixAdd<<<dimGridAdd,dimBlockAdd>>>(M1_d, M2_d, Mout_d, n, p);

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