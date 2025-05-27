#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// Задача 1

// CPU-версия сложения векторов
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// GPU-ядро для сложения векторов
global void vectorAddGPU(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Задача 2

// CPU-версия увеличения яркости
void brightenCPU(const unsigned char* input, unsigned char* output, int size, int delta) {
    for (int i = 0; i < size; i++) {
        int value = input[i] + delta;
        output[i] = (value > 255) ? 255 : value;
    }
}

// GPU-ядро для увеличения яркости
global void brightenGPU(const unsigned char* input, unsigned char* output, int size, int delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int value = input[i] + delta;
        output[i] = (value > 255) ? 255 : value;
    }
}


int main() {
    srand(time(NULL));

// Задача 1
    printf("Задача 1: Сложение векторов\n");
    
    const int N = 1000000;
    size_t vectorSize = N * sizeof(float);
    
    float *h_A = (float*)malloc(vectorSize);
    float *h_B = (float*)malloc(vectorSize);
    float *h_C_cpu = (float*)malloc(vectorSize);
    float *h_C_gpu = (float*)malloc(vectorSize);
    
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    clock_t start = clock();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    clock_t end = clock();
    printf("CPU время: %.3f мс\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, vectorSize);
    cudaMalloc(&d_B, vectorSize);
    cudaMalloc(&d_C, vectorSize);
    
    cudaMemcpy(d_A, h_A, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vectorSize, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    start = clock();
    vectorAddGPU<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    end = clock();
    printf("GPU время: %.3f мс\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);
    
    cudaMemcpy(h_C_gpu, d_C, vectorSize, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 0.0001f) {
            errors++;
        }
    }
    printf("Ошибок: %d\n\n", errors);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

// Задача 2
    printf("Задача 2: Увеличение яркости\n");
    
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int IMG_SIZE = WIDTH * HEIGHT;
    const int BRIGHTNESS_DELTA = 50;
    size_t imgSizeBytes = IMG_SIZE * sizeof(unsigned char);
    
    unsigned char *h_img = (unsigned char*)malloc(imgSizeBytes);
    unsigned char *h_res_cpu = (unsigned char*)malloc(imgSizeBytes);
    unsigned char *h_res_gpu = (unsigned char*)malloc(imgSizeBytes);
    
    for (int i = 0; i < IMG_SIZE; i++) {
        h_img[i] = rand() % 256;
    }
    
    start = clock();
    brightenCPU(h_img, h_res_cpu, IMG_SIZE, BRIGHTNESS_DELTA);
    end = clock();
    printf("CPU время: %.3f мс\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);
    
    unsigned char *d_img, *d_res;
    cudaMalloc(&d_img, imgSizeBytes);
    cudaMalloc(&d_res, imgSizeBytes);
    
    cudaMemcpy(d_img, h_img, imgSizeBytes, cudaMemcpyHostToDevice);
    
    blockSize = 256;
    numBlocks = (IMG_SIZE + blockSize - 1) / blockSize;
    
    start = clock();
    brightenGPU<<<numBlocks, blockSize>>>(d_img, d_res, IMG_SIZE, BRIGHTNESS_DELTA);
    cudaDeviceSynchronize();
    end = clock();
    printf("GPU время: %.3f мс\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);
    
    cudaMemcpy(h_res_gpu, d_res, imgSizeBytes, cudaMemcpyDeviceToHost);
    
    errors = 0;
    for (int i = 0; i < IMG_SIZE; i++) {
        if (h_res_cpu[i] != h_res_gpu[i]) {
            errors++;
        }
    }
    printf("Ошибок: %d\n", errors);
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_img);
    free(h_res_cpu);
    free(h_res_gpu);
    cudaFree(d_img);
    cudaFree(d_res);
    
    return 0;
}
