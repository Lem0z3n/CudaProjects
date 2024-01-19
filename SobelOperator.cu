// This program implements 2D convolution using Constant memory in CUDA
// By: Nick from CoffeeBeforeArch

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// 3 convolutional mask
#define MASK_DIM 3

// Amount the the matrix will hang over the matrix
#define MASK_OFFSET (MASK_DIM / 2)



// 2D Convolution Kernel
// Takes:
//  matrix: Input matrix
//  result: Convolution result
//  N:      Dimensions of the matrices
__global__ void sobelOperator(int *matrix, int *gpuMaskX, int *gpuMaskY,
                                 int *resultFinal, int cols, int rows) { 
   
    // Calculate the global thread positions
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    int tCol = tid % cols;
    int tRow = tid / cols;

    // Temp value for accumulating the result
    int tempX = 0;
    int tempY = 0;

    // Starting index for calculation
    int start_r = tRow - MASK_OFFSET;
    int start_c = tCol - MASK_OFFSET;

    // Iterate over all the rows
    for (int i = 0; i < MASK_DIM; i++) {
        // Go over each column
        for (int j = 0; j < MASK_DIM; j++) {
            // Range check for rows
            if ((start_r + i) >= 0 && (start_r + i) < rows) {
            // Range check for columns
                if ((start_c + j) >= 0 && (start_c + j) < cols) {
                    // Accumulate result
                    tempX += matrix[(start_r + i) * cols + (start_c + j)] *
                            gpuMaskX[i*3+j];
                }
            }
        }
    }

    for (int i = 0; i < MASK_DIM; i++) {
        // Go over each column
        for (int j = 0; j < MASK_DIM; j++) {
            // Range check for rows
            if ((start_r + i) >= 0 && (start_r + i) < rows) {
            // Range check for columns
                if ((start_c + j) >= 0 && (start_c + j) < cols) {
                    // Accumulate result
                    tempY += matrix[(start_r + i) * cols + (start_c + j)] *
                            gpuMaskY[i*3+j];
                }
            }
        }
    }
    //√(𝐻 𝑖𝑗)² + (𝑉 𝑖𝑗)²
    float accResult =  sqrt( pow(tempX,2) + pow(tempY,2));

    //if the result is bigger than the threshold write white if not black.
    (accResult>15) ? resultFinal[tRow*cols+tCol] = 255 : resultFinal[tRow*cols+tCol] =0;

    }

    bool check_result(int * endRes, char* filename, int columns, int N){

    char name[512];
    sprintf(name,"%s.txt",filename);
    FILE * file = fopen(name,"w");

    int i = 0;
    printf("writing image\n");
    while(fprintf(file," %i ",endRes[i])>0 && i < N){
        i++;
        if(i%columns == 0)
            fprintf(file,"\n");
    }
    fclose(file);

    return true;
}
    
int main(int argc, char * args[]) {

    if(argc < 2){
        printf("please provide image filename\n");
        exit(1);
    }
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

    // Allocate the matrix and initialize it
    int *matrix = new int[N];
    int *resultFinal = new int[N];
    
    const int maskX[9] =    {-1,0,1,
                            -2,0,2,
                            -1,0,1};

    const int maskY[9] =    {1,2,1
                            ,0,0,0,
                            -1,-2,-1};


    //convertin from cv datatype to int[]
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            matrix[i*image.rows+j] = static_cast<int>(image.at<uchar>(i,j));
        }
    }
    

    // Allocate device memory
    int *d_matrix;
    int *d_maskX;
    int *d_maskY;
    int *d_resultFinal;

    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_resultFinal, bytes_n);
    cudaMalloc(&d_maskX, sizeof(maskX));
    cudaMalloc(&d_maskY, sizeof(maskX));
    
    printf("checking image\n");
    check_result(matrix,"matrix",image.cols, N);
    // Copy data to the device
    cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskX, maskX, sizeof(maskX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskY, maskY, sizeof(maskX), cudaMemcpyHostToDevice);

    printf("Image copied to GPU\n");
    
    // Threads per TB
    int THREADS = 256;

    // Number of TBs
    int GRID = (N + THREADS - 1) / THREADS;

    // Perform 2D Convolution

    printf("calling gpu\n");
    sobelOperator<<<GRID, THREADS>>>(d_matrix, d_maskX, d_maskY, d_resultFinal, image.cols, image.rows);
    printf("returning from gpu\n");
    // Copy the result back to the CPU

    cudaMemcpy(resultFinal, d_resultFinal, bytes_n, cudaMemcpyDeviceToHost);

    printf("COMPLETED SUCCESSFULLY!\n");

    check_result(resultFinal,"result",image.cols, N);

    cv :: Mat imageResult(image.cols, image.rows, CV_8UC1, resultFinal);

    char resultName [1024];
    sprintf(resultName, "Completed%s", args[1]);

    if (cv::imwrite(resultName, imageResult))
        std::cout << "Image saved successfully!" << std::endl;
    else
        std::cerr << "Error saving image" << std::endl;
    


    // Free the memory we allocated

    //delete[] matrix;
    delete[] resultFinal;
    delete[] matrix;


    cudaFree(d_matrix);
    cudaFree(d_resultFinal);
    cudaFree(d_maskX);
    cudaFree(d_maskY);

    return 0;
}