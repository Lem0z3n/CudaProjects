#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#define MASK_DIM 3

__global__ void sobelEdgeDetector(const unsigned char* inputImage, 
                                  unsigned char* alpha, unsigned char* red, unsigned char* blue,
                                  int *gpuMaskX, int *gpuMaskY,
                                  int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int gx=0;
    int gy=0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {

        int start_c = x-1;
        int start_r = y-1;
        /*
        int sum = 0;
        //Apply a box blur
        for (int i = 0; i < MASK_DIM; i++) {
            // Go over each column
            for (int j = 0; j < MASK_DIM; j++) {
                // Range check for rows
                // Accumulate result
                sum += inputImage[(start_r + i) * width + (start_c + j)];
                             
            }
        }

        sum = sum/9; //we do the mean
        alpha[y*width+x] = sum; //we stablish the new value
        //we might as well use alpha to store temporary information. for memory efficiency 
        __syncthreads();    //we wait for all threads
        */
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

        //normalizing
        magnitude = fminf(255.0f, fmaxf(0.0f, magnitude * 1.0f));
        
        float redV = 0;
        float blueV = 0;

        if(magnitude > 0){
            redV = fminf(255.0f, fmaxf(0.0f, gx * 1.0f));
            blueV = fminf(255.0f, fmaxf(0.0f, gy * 1.0f));
        }

        alpha[y * width + x] = magnitude;
        red[y * width + x] = redV;
        blue[y * width + x] = blueV;

    } else {
        // Border pixels - just copy the input to output
        alpha[y * width + x] = inputImage[y * width + x];
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
    unsigned char* h_alpha = new unsigned char[width * height];
    unsigned char* h_red = new unsigned char[width * height];
    unsigned char* h_blue = new unsigned char[width * height];
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
    unsigned char *d_inputImage, *d_alpha, *d_red, *d_blue;

    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_alpha, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_red, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_blue, width * height * sizeof(unsigned char));
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
    sobelEdgeDetector<<<gridSize, blockSize>>>(d_inputImage, d_alpha, d_blue, d_red, d_maskX, d_maskY, width, height);

    // Copy the result back to the host
    cudaMemcpy(h_alpha, d_alpha, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_red, d_red, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue, d_blue, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Create cv::Mat objects for each channel
    cv::Mat blueMat(height, width, CV_8UC1, h_blue);
    cv::Mat greenMat(height, width, CV_8UC1, cv::Scalar(0)); // Set green channel to all zeros
    cv::Mat redMat(height, width, CV_8UC1, h_red);
    cv::Mat alphaMat(height, width, CV_8UC1, h_alpha);

    // Merge the channels into a single ARGB image
    std::vector<cv::Mat> channels = {blueMat, greenMat, redMat, alphaMat};
    cv::Mat imageResult;
    cv::merge(channels, imageResult);

    char resultName [1024];
    sprintf(resultName, "Completed%s", args[1]);

    if (cv::imwrite(resultName, imageResult))
        std::cout << "Image saved successfully!" << std::endl;
    else
        std::cerr << "Error saving image" << std::endl;

    // Cleanup
    delete[] h_inputImage;
    delete[] h_alpha;
    delete[] h_blue;
    delete[] h_red;

    cudaFree(d_inputImage);
    cudaFree(d_alpha);
    cudaFree(d_red);
    cudaFree(d_blue);
    cudaFree(d_maskX);
    cudaFree(d_maskY);


    return 0;
}
