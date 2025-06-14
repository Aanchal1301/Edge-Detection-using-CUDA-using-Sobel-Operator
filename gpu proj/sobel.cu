#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define SOBEL_SIZE 3

__constant__ int Gx[SOBEL_SIZE * SOBEL_SIZE] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

__constant__ int Gy[SOBEL_SIZE * SOBEL_SIZE] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

// CUDA kernel for Sobel edge detection
__global__ void sobelEdgeDetection(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int gx = 0;
    int gy = 0;

    // Only process valid pixels
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + j) * width + (x + i)];
                gx += Gx[(j + 1) * SOBEL_SIZE + (i + 1)] * pixel;
                gy += Gy[(j + 1) * SOBEL_SIZE + (i + 1)] * pixel;
            }
        }
        
        int magnitude = abs(gx) + abs(gy);
        output[y * width + x] = (magnitude > 255) ? 255 : magnitude;
    }
}

int main() {
    // Load image using OpenCV
    cv::Mat image = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate memory for output image
    cv::Mat outputImage(height, width, CV_8UC1);

    // Device memory pointers
    unsigned char* d_input;
    unsigned char* d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Copy data from host to device
    cudaMemcpy(d_input, image.ptr(), imageSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the Sobel kernel
    sobelEdgeDetection<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(outputImage.ptr(), d_output, imageSize, cudaMemcpyDeviceToHost);

    // Save or display the output image
    cv::imwrite("output.jpg", outputImage);
    cv::imshow("Edge Detected Image", outputImage);
    cv::waitKey(0);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
