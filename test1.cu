#include <iostream>
#include <chrono>

// CUDA 커널 함수: 행렬 곱셈
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) {
        sum += a[row * width + i] * b[i * width + col];
    }
    c[row * width + col] = sum;
}

int main() {
    const int width = 1000;
    const int size = width * width * sizeof(float);
    
    // 호스트 메모리 할당
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 초기화 데이터
    for (int i = 0; i < width * width; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 디바이스 메모리 할당
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // CUDA 커널 함수 실행 구성 정의
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(width / threadsPerBlock.x, width / threadsPerBlock.y);

    auto start = std::chrono::steady_clock::now();

    // 행렬 곱셈 수행
    do {
        matrixMul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        if (elapsed_seconds.count() >= 60) {
            break;
        }
    } while (true);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    // 메모리 해제
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

