#include <iostream>
#include <chrono>

const int N = 1024;

// CUDA 커널 함수: 행렬 추가 함수
__global__ void matrixAddition(float* A, float* B, float* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // 매트릭스 초기화
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // 호스트에서 장치로 데이터 복사
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA 커널 함수 실행 구성 정의
    int threadsPerBlock = 256;
    int numBlocks = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::steady_clock::now();

    do {
        matrixAddition<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        if (elapsed_seconds.count() >= 30) {
            break;
        }
    } while (true);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    // 장치의 결과를 다시 디바이스에 복사합니다.
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 장치 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

