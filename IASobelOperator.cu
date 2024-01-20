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

        // Clamp magnitude to the range [0, 255]
        magnitude = fminf(255.0f, fmaxf(0.0f, magnitude* 1.0f));

        // Set the output pixel value
        outputImage[y * width + x] = static_cast<unsigned char>(magnitude);
    } else {
        // Border pixels - just copy the input to output
        outputImage[y * width + x] = inputImage[y * width + x];
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

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));
    cudaMalloc(&d_maskX, sizeof(maskX));
    cudaMalloc(&d_maskY, sizeof(maskX));

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskX, maskX, sizeof(maskX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maskY, maskY, sizeof(maskX), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the Sobel edge detector kernel
    sobelEdgeDetector<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_maskX, d_maskY, width, height);

    // Copy the result back to the host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv :: Mat imageResult(image.rows, image.cols, CV_8U, h_outputImage);

    char resultName [1024];
    sprintf(resultName, "Completed%s", args[1]);

    if (cv::imwrite(resultName, imageResult))
        std::cout << "Image saved successfully!" << std::endl;
    else
        std::cerr << "Error saving image" << std::endl;

    // Cleanup
    delete[] h_inputImage;
    delete[] h_outputImage;

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_maskX);
    cudaFree(d_maskY);


    return 0;
}
