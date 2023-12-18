#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

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

/*__global__ void Convolution(float* A, float* B, float* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
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
} */

#define NUM_KERNELS 6
#define KERNEL_SIZE 5

__global__ void Convolution(float* A, float* B, float* kernels, int numARows, int numACols, int numCRows, int numCCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows - KERNEL_SIZE + 1 && col < numACols - KERNEL_SIZE + 1) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            float tmp = 0.0f;
            for (int i = 0; i < KERNEL_SIZE; ++i) {
                for (int j = 0; j < KERNEL_SIZE; ++j) {
                    tmp += A[(row + i) * numACols + (col + j)] * kernels[k * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j];
                }
            }
            B[k * numCCols * numCRows + row * numCCols + col] = tmp;
        }
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


__global__ void Downsampling(float* input, float* output, int inputWidth, int inputHeight, int inputDepth) {
    int outputWidth = inputWidth / 2;
    int outputHeight = inputHeight / 2;
    int outputDepth = inputDepth;

    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    if (tx < outputWidth && ty < outputHeight && tz < outputDepth) {
        int startX = tx * 2;
        int startY = ty * 2;
        int endX = startX + 2;
        int endY = startY + 2;

        float maxVal = -100000;

        for (int y = startY; y < endY; ++y) {
            for (int x = startX; x < endX; ++x) {
                float val = input[(tz * inputWidth * inputHeight) + (y * inputWidth) + x];
                if (val > maxVal) {
                    maxVal = val;
                }
            }
        }

        output[(tz * outputWidth * outputHeight) + (ty * outputWidth) + tx] = maxVal;
    }
}



// --------------------------- Implémentation d'un CNN --------------------------------

int main(void) {
   
    // Define CPU
    float *raw_data, *C1_data, *C1_kernel, *S1_data;
    raw_data = (float *) malloc(WA * HA * sizeof(float));
    C1_data = (float *)malloc(6 * WB * HB * sizeof(float));
    C1_kernel = (float *)malloc(6 * WC * HC * sizeof(float));
    S1_data = (float *)malloc(6 * WB / 2 * HB / 2 * sizeof(float));

    MatrixInit1D(raw_data, WA, HA, 1);
    MatrixInitZeros(C1_data, 6, WB, HB);
    MatrixInit1D(C1_kernel, 6, WC, HC);
    MatrixInitZeros(S1_data, 6, WB / 2, HB / 2);

    // Define GPU
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    // Malloc GPU memory
    cudaMalloc((void **)&d_raw_data, WA * HA * sizeof(float));
    cudaMalloc((void **)&d_C1_data, 6 * WB * HB * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, 6 * WC * HC * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_raw_data, raw_data, WA * HA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, 6 * WB * HB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, 6 * WC * HC * sizeof(float), cudaMemcpyHostToDevice);

    // CPU operations
    
    /*for (int i = 0; i < 1; i++) {
        float *C1_kernel_i;
        cudaMalloc((void **)&C1_kernel_i, WC * HC * sizeof(float));

         // Allocate memory for d_C1_kernel_i on the GPU
        float *d_C1_kernel_i;

        cudaMalloc((void **)&d_C1_kernel_i, WC * HC * sizeof(float));

        cudaMemcpy(d_C1_kernel_i, C1_kernel_i, WC * HC * sizeof(float), cudaMemcpyHostToDevice);

        // CPU calculations
        extractMatrix(C1_kernel, i + 1, 6, WC, HC, &C1_kernel_i);
        //MatrixPrint(C1_kernel_i, WC, HC);
        
        // Kernel launch
        dim3 dimGridConv(6, 1, 1);
        dim3 dimBlockConv(WA, HA, 1);

        Convolution<<<dimGridConv, dimBlockConv>>>(d_raw_data, &d_C1_data[i * WB * HB], d_C1_kernel_i, WA, HA, WB, HB, WC, HC);

        free(C1_kernel_i);
        cudaFree(d_C1_kernel_i);
    }*/

    
    dim3 dimBlockConv(HA, WA,1); // À ajuster selon les performances et les dimensions de la matrice
    dim3 dimGridConv(6,1,1);

    Convolution<<<dimGridConv, dimBlockConv>>>(d_raw_data, d_C1_data, d_C1_kernel, WA, HA, WC, HC);

    Downsampling<<<dimGridConv, dimBlockConv>>>(d_C1_data, d_S1_data, HB, WB, 6);

    // Appel du kernel
    
    
    // Copy data from GPU to CPU
    cudaMemcpy(C1_data, d_C1_data, 6 * WB * HB * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, 6 * WB / 2 * HB / 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    MatrixPrint(C1_data, WB, HB);
    MatrixPrint(S1_data,14,14);
    // Free GPU memory
    //cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_C1_kernel);
 
    // Free CPU memory
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;

}