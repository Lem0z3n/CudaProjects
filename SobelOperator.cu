// This program implements 2D convolution using Constant memory in CUDA
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>


// 3 convolutional mask
#define MASK_DIM 3

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)

// Allocate mask in constant memory
struct mask{
    int x1 [MASK_DIM] = {1,2,1};
    int x2 [MASK_DIM] = {-1,0,1};
    int y1 [MASK_DIM] = {1,0,-1};
    int y2 [MASK_DIM] = {1,2,1};
};


__constant__ struct mask gpuMask;
// 2D Convolution Kernel
// Takes:
//  matrix: Input matrix
//  result: Convolution result
//  N:      Dimensions of the matrices
__global__ void convolution_2d(int *matrix, int *resultX, int *resultY,
                                 float *resultFinal, int N) {
    // Calculate the global thread positions
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index for calculation
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    // Temp value for accumulating the result
    int temp = 0;

    // Sum(X1b,𝐴 [i,𝑗−𝑏])
    for (int i = 0; i < MASK_DIM; i++) {
        // Range check for cols
        if ((start_c + i) >= 0 && (start_c + i) < N) {
            // Accumulate result
            temp += matrix[(start_r) * N + (start_c + i)] *
                    gpuMask.x1[i];
        }
    }
    // Write back the result
    resultX[row * N + col] = temp;

    //wait for all the threads to write their result
    __syncthreads(); 

    //Sum( X2𝑎, 𝐻 [𝑖−𝑎,𝑗])
    //NOT COALESCED READING!!! this is baaad
    for (int i = 0; i < MASK_DIM; i++) {
        // Range check for cols
        if ((start_r + i) >= 0 && (start_r + i) < N) {
            // Accumulate result
            temp += resultX[(start_r+i) * N + (start_c)] *
                    gpuMask.x2[i];
        }
    }

    // Write back the result
    resultX[row * N + col] = temp;
    //wait for all the threads to write their result
    __syncthreads(); 

    // Sum(Y1b,𝐴 [i,𝑗−𝑏])
    for (int i = 0; i < MASK_DIM; i++) {
        // Range check for cols
        if ((start_c + i) >= 0 && (start_c + i) < N) {
            // Accumulate result
            temp += matrix[(start_r) * N + (start_c + i)] *
                    gpuMask.y1[i];
        }
    }

    resultY[row*N + col] = temp;
    //wait for all the threads to write their result
    __syncthreads(); 

    //Sum( X2𝑎, 𝐻 [𝑖−𝑎,𝑗])
    //NOT COALESCED READING!!! this is baaad
    for (int i = 0; i < MASK_DIM; i++) {
        // Range check for cols
        if ((start_r + i) >= 0 && (start_r + i) < N) {
            // Accumulate result
            temp += resultY[(start_r+i) * N + (start_c)] *
                    gpuMask.y2[i];
        }
    }

    resultY[row*N + col] = temp;
    //wait for all the threads to write their result
    __syncthreads(); 

    //√(𝐻 𝑖𝑗)² + (𝑉 𝑖𝑗)²

    resultFinal[row*N+col] =  sqrt( pow(resultX[row*N+col],2) + pow(resultY[row*N+col],2)) ;
}
    




// Verifies the 2D convolution result on the CPU
// Takes:
//  m:      Original matrix
//  mask:   Convolutional mask
//  result: Result from the GPU
//  N:      Dimensions of the matrix
void verify_result(int *m, int *mask, int *result, int N) {
    // Temp value for accumulating results
    int temp;

    // Intermediate value for more readable code
    int offset_r;
    int offset_c;

    // Go over each row
    for (int i = 0; i < N; i++) {
        // Go over each column
        for (int j = 0; j < N; j++) {
            // Reset the temp variable
            temp = 0;

            // Go over each mask row
            for (int k = 0; k < MASK_DIM; k++) {
            // Update offset value for row
                offset_r = i - MASK_OFFSET + k;

                // Go over each mask column
                for (int l = 0; l < MASK_DIM; l++) {
                    // Update offset value for column
                    offset_c = j - MASK_OFFSET + l;

                    // Range checks if we are hanging off the matrix
                    if (offset_r >= 0 && offset_r < N) {
                        if (offset_c >= 0 && offset_c < N) {
                            // Accumulate partial results
                            temp += m[offset_r * N + offset_c] * mask[k * MASK_DIM + l];
                        }
                    }
                }
            }
            // Fail if the results don't match
            assert(result[i * N + j] == temp);
        }
    }
}




int main(int argc, char * args[]) {

    //load image into cpu memory
    cv :: Mat image = cv :: imread(args[1],cv::IMREAD_GRAYSCALE);
    //error check
    if (image.empty()) {
        std::cout << "Error: Unable to read the image." << std::endl;
        return -1;
    }
    printf("image read.\n");   
    // Dimensions of the image
    int N = image.cols * image.rows;

    // Size of the matrix (in bytes)
    size_t bytes_n = N  * sizeof(int);
    size_t bytes_res = N * sizeof(float);

    // Allocate the matrix and initialize it
    int *matrix = new int[N];
    int *resultX = new int[N];
    int *resultY = new int[N];
    float *resultFinal = new float[N];


    //convertin from cv datatype to int[]
    for(int i = 0; i < image.cols; i++){
        for(int j = 0; j < image.rows; j++){
            matrix[i+j] = static_cast<int>(image.at<uchar>(i,j));
        }
    }
    

    // Allocate device memory
    int *d_matrix;
    int *d_resultX;
    int *d_resultY;
    float *d_resultFinal;

    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_resultX, bytes_n);
    cudaMalloc(&d_resultY, bytes_n);
    cudaMalloc(&d_resultFinal, bytes_n);
    //allocate memory in gpu for mask
    mask hostMask;
    cudaMemcpyToSymbol(gpuMask,&hostMask,sizeof(mask));

    // Copy data to the device
    cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
    printf("Image copied to GPU\n");
    // Calculate grid dimensions
    int THREADS = 16;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    // Perform 2D Convolution
    printf("calling gpu\n");
    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_resultX, d_resultY, d_resultFinal, N);
    printf("returning from gpu\n");
    // Copy the result back to the CPU

    //cudaMemcpy(resultFinal, d_resultFinal, bytes_res, cudaMemcpyDeviceToHost);


    printf("COMPLETED SUCCESSFULLY!\n");

    // Free the memory we allocated
    delete[] matrix;
    delete[] resultX;
    delete[] resultY;
    delete[] resultFinal;

    cudaFree(d_matrix);
    cudaFree(d_resultX);
    cudaFree(d_resultY);
    cudaFree(d_resultFinal);

    return 0;
}