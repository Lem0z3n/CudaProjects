// This program implements a 1D convolution using CUDA
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

// 1-D convolution kernel
//  Arguments:
//      array   = padded array
//      mask    = convolution mask
//      result  = result array
//      n       = number of elements in array
//      m       = number of elements in the mask
__global__ void convolution_1d(int *array, int *mask, int *result, int n,
                               int m) {
  // Global thread ID calculation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate radius of the mask
  int r = m / 2;

  // Calculate the starting point for the element
  int start = tid - r;

  // Temp value for calculation
  int temp = 0;

  // Go over each element of the mask
  for (int j = 0; j < m; j++) {
    // Ignore elements that hang off (0s don't contribute)
    if (((start + j) >= 0) && (start + j < n)) {
      // accumulate partial results
      temp += array[start + j] * mask[j];
    }
  }

  // Write-back the results
  result[tid] = temp;
}

// Verify the result on the CPU
bool check_result(int * endRes, int N, const char* filename){


    FILE * file = fopen( filename,"w");

    int i = 0;
    printf("cheking result\n");
    while(fprintf(file," %i ",endRes[i])>0 && i < N){
        i++;
    }
    fclose(file);

    return true;
}

int main() {
  // Number of elements in result array
  int n = 1 << 20;

  // Size of the array in bytes
  int bytes_n = n * sizeof(int);

  // Number of elements in the convolution mask
  int m = 7;

  // Size of mask in bytes
  int bytes_m = m * sizeof(int);

  // Allocate the array (include edge elements)...
  std::vector<int> h_array(n);

  // ... and initialize it
  std::generate(begin(h_array), end(h_array), [](){ return rand() % 100; });

  // Allocate the mask and initialize it
  std::vector<int> h_mask(m);
  std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

  // Allocate space for the result
  std::vector<int> h_result(n);

  // Allocate space on the device
  int *d_array, *d_mask, *d_result;
  cudaMalloc(&d_array, bytes_n);
  cudaMalloc(&d_mask, bytes_m);
  cudaMalloc(&d_result, bytes_n);

  // Copy the data to the device
  cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

  // Threads per TB
  int THREADS = 256;

  // Number of TBs
  int GRID = (n + THREADS - 1) / THREADS;

  check_result(h_array.data(), h_array.size(), "beforeConvolution.txt");

  // Call the kernel
  convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

  // Copy back the result
  cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

  // Verify the result
  check_result(h_array.data(), h_array.size(), "afterConvolution.txt");

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free allocated memory on the device and host
  cudaFree(d_result);
  cudaFree(d_mask);
  cudaFree(d_array);

  return 0;
}