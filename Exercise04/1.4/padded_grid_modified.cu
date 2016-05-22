#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/random.h>
#include <thrust/extrema.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <float.h>

// This example computes the minimum and maximum values
// over a padded grid.  The padded values are not considered
// during the reduction operation.

//PermutationIterator that skips the padded values.
template <typename Iterator>
class pitched_range
{
  public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct pitch_functor : public thrust::unary_function<difference_type,difference_type>
    {
        const difference_type _width;
        const difference_type _padded_width;
        const bool _no_padding;

        pitch_functor(difference_type width, difference_type padded_width)
            : _width(width), _padded_width(padded_width), _no_padding(width == padded_width) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            if (_no_padding) return i;

            // we need to convert from the unpadded index to the padded
            difference_type row = i / _width;
            // save a division versus i % width
            difference_type col = i - (row * _width);
            return row * _padded_width + col;
        }
    };

    typedef typename thrust::counting_iterator<difference_type> CountingIterator;
    typedef typename thrust::transform_iterator<pitch_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator> PermutationIterator;

    // type of the pitched_range iterator
    typedef PermutationIterator iterator;

    // construct our iterator for the range [first,last)
    pitched_range(Iterator first, Iterator last, difference_type width, difference_type padded_width)
        : _first(first), _last(last), _width(width), _padded_width(padded_width) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(_first, TransformIterator(CountingIterator(0), pitch_functor(_width, _padded_width)));
    }

    iterator end(void) const
    {
        /// convert from padded to unpadded size
        return begin() + ((_last - _first)/_padded_width)*_width;
    }
    
  protected:
    difference_type _width;
    difference_type _padded_width;
    Iterator _first;
    Iterator _last;
};


// transform value into a tuple(value,value)
template <typename IndexType, typename ValueType>
struct transform_tuple : 
    public thrust::unary_function< ValueType, 
                                   thrust::tuple<ValueType,ValueType> >
{
  typedef typename thrust::tuple<ValueType,ValueType> OutputTuple;


  __host__ __device__
    OutputTuple operator()(const ValueType& t) const
    { 
		//Create tuple of format (ValueType, ValueType)	
		return OutputTuple(t, t);
    }
};


// reduce two tuples (value,value) into a single tuple such that output
// contains the smallest and largest values.
template <typename IndexType, typename ValueType>
struct reduce_tuple :
    public thrust::binary_function< thrust::tuple<ValueType,ValueType>,
                                    thrust::tuple<ValueType,ValueType>,
                                    thrust::tuple<ValueType,ValueType> >
{
  typedef typename thrust::tuple<ValueType,ValueType> Tuple;

  __host__ __device__
    Tuple operator()(const Tuple& t0, const Tuple& t1) const
    { 
		//Save the additional checks as all incoming values are valid		
		return Tuple(thrust::min(thrust::get<0>(t0), thrust::get<0>(t1)),
			thrust::max(thrust::get<1>(t0), thrust::get<1>(t1)));		
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
  typedef thrust::tuple<float, float> result_type;

  result_type                 init(FLT_MAX, -FLT_MAX); // initial value
  transform_tuple<int,float>  unary_op;                		// transformation operator
  reduce_tuple<int,float>     binary_op;                     // reduction operator

  // Create the permutation iterator based on number of columns w/o and with padding.
  typedef thrust::device_vector<float>::iterator Iterator;
  pitched_range<Iterator> padIt(data.begin(), data.end(), n, N);
  
  // Synchronise cuda device and start measurement.
  cudaDeviceSynchronize();
  auto startTime = std::chrono::high_resolution_clock::now();
  
  result_type result = 
    thrust::transform_reduce(
        padIt.begin(),
        padIt.end(),
        unary_op,
        init,
        binary_op);
		
  cudaDeviceSynchronize();
  auto endTime= std::chrono::high_resolution_clock::now();
  std::cout << "seconds: "<< (endTime-startTime).count() << std::endl;

  std::cout << "minimum value: " << thrust::get<0>(result) << std::endl;
  std::cout << "maximum value: " << thrust::get<1>(result) << std::endl;

  return 0;
}