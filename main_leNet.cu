#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 32
// Size of input matrix (raw_data)
#define WA 32   
#define HA 32     

// Size of our convolution kernels
#define HC 5     
#define WC 5

// Size of our output matrixes
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)


// --------------------- Initialization of Matrix ----------------------

void MatrixInit(float *M, int n, int p){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < p; j++){
            *(M + i * p + j) = (float)(rand() % 200) / 100 - 1;
        }
    }
}

void MatrixInit1D(float *M, int n, int p, int q){
    for (int i = 0; i < n*p*q; i++){
        *(M + i) = (float)(rand() % 100) / 100;
    }
}

void MatrixInitZeros(float *M, int n, int p, int q){
    for (int i = 0; i < n*p*q; i++){
        *(M + i) = (float) 0;
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

__global__ void Convolution(float* A, float* B, float* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
	int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
	int row_i = row - WC + 1;
	int col_i = col - WC + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0)
	{
		shm[threadIdx.y][threadIdx.x] = A[col_i * WA + row_i];
	}
	else
	{
		shm[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) && col < (WB - WC + 1))
	{
		for (int i = 0; i< WC;i++)
			for (int j = 0;j<WC;j++)
				tmp += shm[threadIdx.y + i][threadIdx.x + j] * C[j*WC + i];
		B[col*WB + row] = tmp;
	}
}

// ----------------------------- Extract kernel ----------------------------------

void extractMatrix(float *inputMatrix, int index, int n, int p, int q, float **outputMatrix) {
    *outputMatrix = (float *)malloc(p * q * sizeof(float));

    int start = (index - 1) * p * q; // Adjusting the index for 1-based indexing

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            int matrixIndex = start + i * q + j;
            (*outputMatrix)[i * q + j] = inputMatrix[matrixIndex];
        }
    }
}

// --------------------------- Implémentation d'un CNN --------------------------------

int main(void) {
    
    // Define CPU
    float *raw_data;
    cudaMalloc((void **)&raw_data, WA * HA * sizeof(float));
    MatrixInit1D(raw_data, WA, HA, 1);

    float *C1_data = (float *)malloc(6 * WB * HB * sizeof(float));
    MatrixInitZeros(C1_data, 6, WB, HB);

    float *S1_data = (float *)malloc(6 * WB / 2 * HB / 2 * sizeof(float));
    MatrixInitZeros(S1_data, 6, WB / 2, HA / 2);

    float *C1_kernel = (float *)malloc(6 * WC * HC * sizeof(float));
    MatrixInit1D(C1_kernel, 6, WC, HC);

    // Define GPU
    float *d_raw_data, *d_C1_data, *d_C1_kernel;

    // Malloc GPU memory
    cudaMalloc((void **)&d_raw_data, WA * HA * sizeof(float));
    cudaMalloc((void **)&d_C1_data, 6 * WB * HB * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, 6 * WC * HC * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_raw_data, raw_data, WA * HA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, 6 * WB * HB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, 6 * WC * HC * sizeof(float), cudaMemcpyHostToDevice);

    // CPU operations
    for (int i = 0; i < 6; i++) {
        float *C1_kernel_i;

        // CPU calculations
        extractMatrix(C1_kernel, i + 1, 6, WC, HC, &C1_kernel_i);
        MatrixPrint(C1_kernel_i, WC, HC);

        // Kernel launch
        dim3 dimGridConv(1, 1, 1);
        dim3 dimBlockConv(WA, HA, 1);

        Convolution<<<dimGridConv, dimBlockConv>>>(d_raw_data, &d_C1_data[i * WB * HB], d_C1_kernel_i, WA, HA, WB, HB, WC, HC);

        free(C1_kernel_i);
    }

    // Copy data from GPU to CPU
    cudaMemcpy(C1_data, d_C1_data, 6 * WB * HB * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, 6 * WB / 2 * HB / 2 * sizeof(float), cudaMemcpyDeviceToHost);


    // Free GPU memory
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_C1_kernel);

    // Free CPU memory
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;
}
