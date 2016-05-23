#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <chrono>
#include <iostream>
#include <iomanip>

// This example computes a summed area table using segmented scan
// http://en.wikipedia.org/wiki/Summed_area_table



// convert a linear index to a linear index in the transpose 
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
  size_t m, n;

  __host__ __device__
  transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

  __host__ __device__
  size_t operator()(size_t linear_index)
  {
      size_t i = linear_index / n;
      size_t j = linear_index % n;

      return m * j + i;
  }
};

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t,size_t>
{
  size_t n;
  
  __host__ __device__
  row_index(size_t _n) : n(_n) {}

  __host__ __device__
  size_t operator()(size_t i)
  {
      return i / n;
  }
};

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
  thrust::counting_iterator<size_t> indices(0);
  
  thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
     thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
     src.begin(),
     dst.begin());
}


// scan the rows of an M-by-N array
template <typename T>
void scan_horizontally(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::counting_iterator<size_t> indices(0);

  thrust::inclusive_scan_by_key
    (thrust::make_transform_iterator(indices, row_index(n)),
     thrust::make_transform_iterator(indices, row_index(n)) + d_data.size(),
     d_data.begin(),
     d_data.begin());
}

// scan the rows of an M-by-N array
template <typename T>
void transpose_scan_horizontally(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  
  auto pi_ = thrust::make_permutation_iterator(
      d_data.begin(), 
      thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0),
      transpose_index(n, m))
    );
    
  thrust::inclusive_scan_by_key(
    thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_index(m)),
    thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_index(m)) + d_data.size(),
    
    pi_,
    
    pi_
    
  );
  
  /*
  1. Approach:
  
  thrust::counting_iterator<size_t> indices(0);
  
  thrust::inclusive_scan_by_key
    (thrust::make_transform_iterator(indices, row_index(m)),
     thrust::make_transform_iterator(indices, row_index(m)) + d_data.size(),
     thrust::make_permutation_iterator(d_data.begin(),thrust::make_transform_iterator(indices,transpose_index(n,m))),
     d_data.begin());
  */
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::host_vector<T> h_data = d_data;

  for(size_t i = 0; i < m; i++)
  {
    for(size_t j = 0; j < n; j++)
      std::cout << std::setw(8) << h_data[i * n + j] << " ";
    std::cout << "\n";
  }
}

int main(void)
{
  size_t m = 1e5; // number of rows
  size_t n = 10; // number of columns

  std::cout << "Nr. of Rows: " << m << std::endl;
  std::cout << "Nr. of Columns: " << n << std::endl;
  
  // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
  //thrust::device_vector<int> data(m * n, 1);
  //thrust::device_vector<int> data_modified(m * n, 1);
  
  std::cout << "THRUST EXAMPLE: " << std::endl;
  std::cout << "======================== " << std::endl;
  {
    // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
    thrust::device_vector<int> data(m * n, 1);
    
    // Synchronise cuda device and start measurement.
    cudaDeviceSynchronize();
    auto startTime = std::chrono::high_resolution_clock::now();
    
    //std::cout << "[step 1] scan horizontally" << std::endl;
    scan_horizontally(m, n, data);
    
    //std::cout << "[step 2] transpose and scan with temp array" << std::endl
    thrust::device_vector<int> temp(m * n);
    transpose(m, n, data, temp);
    
    //std::cout << "[step 3] scan transpose horizontally" << std::endl;
    scan_horizontally(n, m, temp);
    
    //std::cout << "[step 4] transpose the transpose" << std::endl;
    transpose(n, m, temp, data);
   // print(m, n, data);
    
    cudaDeviceSynchronize();
    auto endTime= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = endTime-startTime;
    std::cout << "Temp approach in ms: "<< diff.count()*1000 << std::endl;
  }
  
  std::cout << std::endl;
  
  std::cout << "THRUST EXAMPLE MODIFIED: " << std::endl;
  std::cout << "======================== " << std::endl;
  {
    // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
    thrust::device_vector<int> data(m * n, 1);    
    
    // Synchronise cuda device and start measurement.
    cudaDeviceSynchronize();
    auto startTime = std::chrono::high_resolution_clock::now();
    
    //std::cout << "[step 1] scan horizontally" << std::endl;
    scan_horizontally(m, n, data);
    
    transpose_scan_horizontally(m, n, data);
   // print(m, n, data);
    
    cudaDeviceSynchronize();
    auto endTime= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = endTime-startTime;
    std::cout << "Perm_iter approach in ms: "<< diff.count()*1000 << std::endl;
  }

  return 0;
}