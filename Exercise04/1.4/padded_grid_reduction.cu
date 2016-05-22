#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/extrema.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <float.h>

// This example computes the minimum and maximum values
// over a padded grid.  The padded values are not considered
// during the reduction operation.


// transform a tuple (int,value) into a tuple (bool,value,value)
// where the bool is true for valid grid values and false for 
// values in the padded region of the grid
template <typename IndexType, typename ValueType>
struct transform_tuple : 
    public thrust::unary_function< thrust::tuple<IndexType,ValueType>, 
                                   thrust::tuple<bool,ValueType,ValueType> >
{
  typedef typename thrust::tuple<IndexType,ValueType>      InputTuple;
  typedef typename thrust::tuple<bool,ValueType,ValueType> OutputTuple;

  IndexType n, N;

  transform_tuple(IndexType n, IndexType N) : n(n), N(N) {}

  __host__ __device__
    OutputTuple operator()(const InputTuple& t) const
    { 
      bool is_valid = (thrust::get<0>(t) % N) < n;
      return OutputTuple(is_valid, thrust::get<1>(t), thrust::get<1>(t));
    }
};


// reduce two tuples (bool,value,value) into a single tuple such that output
// contains the smallest and largest *valid* values.
template <typename IndexType, typename ValueType>
struct reduce_tuple :
    public thrust::binary_function< thrust::tuple<bool,ValueType,ValueType>,
                                    thrust::tuple<bool,ValueType,ValueType>,
                                    thrust::tuple<bool,ValueType,ValueType> >
{
  typedef typename thrust::tuple<bool,ValueType,ValueType> Tuple;

  __host__ __device__
    Tuple operator()(const Tuple& t0, const Tuple& t1) const
    { 
      if(thrust::get<0>(t0) && thrust::get<0>(t1)) // both valid
        return Tuple(true, 
            thrust::min(thrust::get<1>(t0), thrust::get<1>(t1)),
            thrust::max(thrust::get<2>(t0), thrust::get<2>(t1)));
      else if (thrust::get<0>(t0))
        return t0;
      else if (thrust::get<0>(t1))
        return t1;
      else
        return t1; // if neither is valid then it doesn't matter what we return
    }
};

int main(void)
{
  int M = 1e3;  // number of rows
  int n = 1011;  // number of columns excluding padding
  int N = 1024;  // number of columns including padding

  thrust::default_random_engine rng(12345);
  thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

  thrust::device_vector<float> data(M * N, -1);

  // initialize valid values in grid
  for(int i = 0; i < M; i++)
    for(int j = 0; j < n; j++)
      data[i * N + j] = dist(rng);


  // compute min & max over valid region of the 2d grid
  typedef thrust::tuple<bool, float, float> result_type;

  result_type                 init(true, FLT_MAX, -FLT_MAX); // initial value
  transform_tuple<int,float>  unary_op(n, N);                // transformation operator
  reduce_tuple<int,float>     binary_op;                     // reduction operator

  
  cudaDeviceSynchronize();
  auto startTime = std::chrono::high_resolution_clock::now();
  
  result_type result = 
    thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), data.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), data.begin())) + data.size(),
        unary_op,
        init,
        binary_op);
		
		
  cudaDeviceSynchronize();
  auto endTime= std::chrono::high_resolution_clock::now();
  std::cout << "seconds: "<< (endTime-startTime).count() << std::endl;


  std::cout << "minimum value: " << thrust::get<1>(result) << std::endl;
  std::cout << "maximum value: " << thrust::get<2>(result) << std::endl;

  return 0;
}

