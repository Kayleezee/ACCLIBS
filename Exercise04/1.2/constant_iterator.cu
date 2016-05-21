#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h> 
#include <iterator>
#include <iostream>

int main(void)
{
    thrust::device_vector<int> data(4);
    data[0] = 3;
    data[1] = 7;
    data[2] = 2;
    data[3] = 5;

    thrust::constant_iterator<int> first(10);
    //thrust::constant_iterator<int> last = first + data.size();
  
    // add 10 to all values in data    
    thrust::transform(first, first + data.size(),
                      data.begin(),
                      data.begin(),
                      thrust::plus<int>());

    // data is now [13, 17, 12, 15]

    // print result
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}
