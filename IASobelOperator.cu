#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

__global__ void sobelEdgeDetector(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Apply the Sobel filter
        int gx = -1 * inputImage[(y - 1) * width + (x - 1)] + inputImage[(y - 1) * width + (x + 1)] +
                 -2 * inputImage[y * width + (x - 1)] + 2 * inputImage[y * width + (x + 1)] +
                 -1 * inputImage[(y + 1) * width + (x - 1)] + inputImage[(y + 1) * width + (x + 1)];

        int gy = -1 * inputImage[(y - 1) * width + (x - 1)] - 2 * inputImage[(y - 1) * width + x] - inputImage[(y - 1) * width + (x + 1)] +
                  inputImage[(y + 1) * width + (x - 1)] + 2 * inputImage[(y + 1) * width + x] + inputImage[(y + 1) * width + (x + 1)];

        // Calculate gradient magnitude
        float magnitude = sqrt(static_cast<float>(gx * gx + gy * gy));

        // Clamp magnitude to the range [0, 255]
        magnitude = fminf(255.0f, fmaxf(0.0f, magnitude));

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

    // Allocate memory for the input and output images
    unsigned char* h_inputImage = new unsigned char[width * height];
    unsigned char* h_outputImage = new unsigned char[width * height];

    // Initialize input image (populate it with some values)

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the Sobel edge detector kernel
    sobelEdgeDetector<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    // Copy the result back to the host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv :: Mat imageResult(image.rows, image.cols, CV_8UC1, h_outputImage);

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

    return 0;
}
