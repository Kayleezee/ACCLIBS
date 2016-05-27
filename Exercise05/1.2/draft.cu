#include <stdio.h>

//include CUDA Runtime
#include <cuda_runtime.h>

#include <thrust/memory.h>
#include <thrust/system/cuda/memory.h>

//thrust includes
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <iostream>
#include <iomanip>
#include <chrono>



int main(int argc, char *argv[]) {
  
  int N = 1e6;
  
  thrust::default_random_engine rng(123456);
  thrust::uniform_int_distribution<int> dist(0, 1e9);
  
  while(N <= 1e9)
  {
    // raw pointer to device memory
    int * raw_ptr;
    
    //alloc memory
    cudaError_t ref = cudaMallocHost(&raw_ptr, N * sizeof(int));

    if(ref != cudaSuccess)
      std::cout << "Malloc failed at N = " << N << std::endl;
    
    
    cudaDeviceProp deviceProp;
          cudaGetDeviceProperties(&deviceProp, 0);

    if(deviceProp.totalGlobalMem > N*sizeof(int))
    {
      //Data fits into GPU, create device_ptr
      
      std::cout << "Data fits on GPU" << std::endl;
      std::cout << "Data size: " << N << std::endl;
      
      // wrap raw pointer with a device_ptr 
      thrust::device_ptr<int> dev_ptr(raw_ptr);
      
      for(size_t i = 0; i < N; i++)
      {
        dev_ptr[i] = dist(rng);
      }
      
      std::cout << "Vector is filled with random numbers" << std::endl;
      
      cudaDeviceSynchronize();
      auto startTime = std::chrono::high_resolution_clock::now();
      
      // Do the sorting thing
      thrust::sort(dev_ptr, dev_ptr + N);
      
      cudaDeviceSynchronize();
      auto endTime= std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = endTime-startTime;
      std::cout << "Elapsed time for sorting in ms: "<< diff.count()*1000 << std::endl;
    }
    else
    {
      //Data is too big to be moved completely to GPU work on host_ptr
      
      int * h_ptr;
      int * d_ptr;
      if(cudaMallocHost(&h_ptr, N * sizeof(int)) == cudaSuccess) 
      {
        
        if(cudaHostGetDevicePointer(&d_ptr, h_ptr, 0) == cudaSuccess) 
        {
          thrust::cuda::pointer<int> begin  = thrust::cuda::pointer<int>(d_ptr);
          thrust::cuda::pointer<int> end    = begin + N;
          
          thrust::tabulate(begin, end, thrust::placeholders::_1 % 1024);
          
          thrust::sort(thrust::cuda::par(&d_ptr), begin, end);
        }
      }
      
      
      
      /*
      // wrap raw pointer with a device_ptr 
      thrust::host_ptr<int> host_ptr(raw_ptr);
      
      for(size_t i = 0; i < N; i++)
      {
        host_ptr[i] = dist(rng);
      }
      
      // Do the sorting thingy
      thrust::sort(host_ptr.begin(), host_ptr.end());
      */
    }

    // Free the allocated memory
    cudaFreeHost(raw_ptr);

    // Increment N
    N = N*10;
  }
}


