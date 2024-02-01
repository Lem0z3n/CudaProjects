#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#define MASK_DIM 3

__global__ void sobelEdgeDetector(const unsigned char* inputImage, unsigned char* outputImage, int *gpuMaskX, int *gpuMaskY,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int gx=0;
    int gy=0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {

        int start_c = x-1;
        int start_r = y-1;
    
        for (int i = 0; i < MASK_DIM; i++) {
            // Go over each column
            for (int j = 0; j < MASK_DIM; j++) {
                // Range check for rows
                // Accumulate result
                gx += inputImage[(start_r + i) * width + (start_c + j)] *
                        gpuMaskX[i*3+j];
                gy += inputImage[(start_r + i) * width + (start_c + j)] *
                        gpuMaskY[i*3+j];
                
            }
        }
    
        // Calculate gradient magnitude
        float magnitude = sqrt(static_cast<float>(gx * gx + gy * gy));

        int threshold = 75;

        //magnitude = (magnitude > threshold) ? 255 : 0;

        magnitude = fminf(255.0f, fmaxf(0.0f, magnitude * 1.0f));

        outputImage[y * width + x] = magnitude;
    } else {
        // Border pixels - just copy the input to output
        outputImage[y * width + x] = inputImage[y * width + x];
    }
}

void CpuSobelOperator(const unsigned char* inputImage, unsigned char* outputImage, 
                      const int *maskX, const int *maskY,
                      const int width, const int height)
{

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int gx=0;
                int gy=0;

                int start_c = x-1;
                int start_r = y-1;

                for (int i = 0; i < MASK_DIM; i++) {
                        // Go over each column
                        for (int j = 0; j < MASK_DIM; j++) {
                            // Range check for rows
                            // Accumulate result
                            gx += inputImage[(start_r + i) * width + (start_c + j)] *
                                    maskX[i*3+j];
                            gy += inputImage[(start_r + i) * width + (start_c + j)] *
                                    maskY[i*3+j];
                            
                        }
                }
                float magnitude = sqrt(static_cast<float>(gx * gx + gy * gy));
                magnitude = fminf(255.0f, fmaxf(0.0f, magnitude * 1.0f));
                outputImage[y * width + x] = magnitude;
            } else {
                // Border pixels - just copy the input to output
                outputImage[y * width + x] = inputImage[y * width + x];
            }
        }
    }
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

    int height = image.rows;
    int width = image.cols;

    const int maskX[9] =    {-1,0,1,
                            -2,0,2,
                            -1,0,1};

    const int maskY[9] =    {1,2,1
                            ,0,0,0,
                            -1,-2,-1};

    // Allocate memory for the input and output images
    unsigned char* h_inputImage = new unsigned char[width * height];
    unsigned char* h_outputImage = new unsigned char[width * height];
    int *d_maskX;
    int *d_maskY;

    // Initialize input image (populate it with some values)
    int index=0;
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            h_inputImage[index++] =(image.at<uchar>(i,j));
        }
    }
    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));
    cudaMalloc(&d_maskX, sizeof(maskX));
    cudaMalloc(&d_maskY, sizeof(maskX));

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskX, maskX, sizeof(maskX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskY, maskY, sizeof(maskX), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start);

    // Launch the Sobel edge detector kernel
    sobelEdgeDetector<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_maskX, d_maskY, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv :: Mat imageResult(image.rows, image.cols, CV_8U, h_outputImage);

    char resultName [1024];
    sprintf(resultName, "Completed%s", args[1]);

    cv::imwrite(resultName, imageResult);

    cudaEventRecord(start);

    CpuSobelOperator(h_inputImage, h_outputImage, maskX, maskY, width, height);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cpuTime = 0;
    cudaEventElapsedTime(&cpuTime, start, stop);

    sprintf(resultName, "CompletedCPU%s", args[1]);

    cv::imwrite(resultName, imageResult);

    printf("Cpu time = %f ms\n", cpuTime);
    printf("Gpu time = %f ms\n", kernelTime);

    // Cleanup
    delete[] h_inputImage;
    delete[] h_outputImage;

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_maskX);
    cudaFree(d_maskY);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}
