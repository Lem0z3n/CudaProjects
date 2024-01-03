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
__global__ void sobelOperator(int *matrix, int *resultX, int *resultY,
                                 float *resultFinal, int cols, int rows) { //missing rows and cols i cant use N
   
    // Calculate the global thread positions
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int tCol = tid % cols;
    int tRow = (tid-tCol) % cols;

    // Starting index for calculation
    int start_c = tCol - MASK_OFFSET;
    int start_r = tRow - MASK_OFFSET;

    // Temp value for accumulating the result
    int tempX = 0;
    int tempY = 0;

    // Sum(X1b,𝐴 [i,𝑗−𝑏])
    for (int i = 0; i < MASK_DIM; i++) {
        // Range check for cols
        if ((start_c + i) >= 0 && (start_c + i) < cols) {
            // Accumulate result
            tempX += matrix[(tRow) * cols + (start_c + i)] *
                    gpuMask.x2[i];
            tempY +=matrix[(tRow) * cols + (start_c + i)] *
                    gpuMask.y2[i];
        }
    }
    // Write back the result
    resultX[tRow * cols + tCol] = tempX;
    resultY[tRow * cols + tCol] = tempY;
    //wait for all the threads to write their result
    __syncthreads(); 
    
    //√(𝐻 𝑖𝑗)² + (𝑉 𝑖𝑗)²
    resultFinal[tRow*cols+tCol] =  sqrt( pow(resultX[tRow*cols+tCol],2) + pow(resultY[tRow*cols+tCol],2));
   
   //if the result is bigger than the threshold write white if not black.
    (resultFinal[tRow*cols+tCol]>600) ? resultFinal[tRow*cols+tCol] = 255 
    : resultFinal[tRow*cols+tCol] =0;

    }

bool check_result(float * endRes, char* filename){

    char *name;
    sprintf(name,"%s.txt",filename);
    int fd = open(name,O_CREAT);

    char buf[sizeof(float)+4];
    int i = 0;
    sprintf(buf,"%f", endRes[i]);
    printf("writing image\n");
    while(write(fd,buf,sizeof(buf)) != -1){
        i++;
        sprintf(buf,"%f", endRes[i]);
    }

    close(fd);
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
    size_t bytes_res = N * sizeof(float);

    // Allocate the matrix and initialize it
    int *matrix = new int[N];
    int *resultX = new int[N];
    int *resultY = new int[N];
    float *resultFinal = new float[N];


    //convertin from cv datatype to int[]
    for(int i = 0; i < image.cols; i++){
        for(int j = 0; j < image.rows; j++){
            matrix[i+j] = static_cast<float>(image.at<uchar>(i,j));
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
    cudaMalloc(&d_resultFinal, bytes_res);


    //allocate memory in gpu for mask
    mask hostMask;
    cudaMemcpyToSymbol(gpuMask,&hostMask,sizeof(mask));
    
    
    for(int i = 0; i < N ; i++){
        resultFinal[i] = matrix[i];
    }
    printf("checking image\n");
    check_result(resultFinal,"matrix");
    // Copy data to the device
    cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
    printf("Image copied to GPU\n");
    
    // Threads per TB
    int THREADS = 256;

    // Number of TBs
    int GRID = (N + THREADS - 1) / THREADS;

    // Perform 2D Convolution
    printf("calling gpu\n");
    sobelOperator<<<GRID, THREADS>>>(d_matrix, d_resultX, d_resultY, d_resultFinal, image.cols, image.rows);
    printf("returning from gpu\n");
    // Copy the result back to the CPU

    cudaMemcpy(resultFinal, d_resultFinal, bytes_res, cudaMemcpyDeviceToHost);

    printf("COMPLETED SUCCESSFULLY!\n");

    check_result(resultFinal,"result");

    cv :: Mat imageResult(image.cols, image.rows, CV_32F, resultFinal);

    imageResult.convertTo(imageResult, CV_8U);

    char resultName [1024];
    sprintf(resultName, "Completed%s", args[1]);

    if (cv::imwrite(resultName, imageResult))
        std::cout << "Image saved successfully!" << std::endl;
    else
        std::cerr << "Error saving image" << std::endl;
    


    // Free the memory we allocated

    //delete[] matrix;
    delete[] resultX;
    delete[] resultY;
    delete[] resultFinal;

    cudaFree(d_matrix);
    cudaFree(d_resultX);
    cudaFree(d_resultY);
    cudaFree(d_resultFinal);

    return 0;
}