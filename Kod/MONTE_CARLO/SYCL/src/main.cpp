#include <CL/sycl.hpp>
#include <algorithm>
#include <device_selector.hpp>
#include <range.hpp>
#include <oneapi/dpl/random>
#include <boost/multiprecision/cpp_bin_float.hpp>

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(), [](double &val) {
    val = std::pow(val, 20);
  });
  return std::sin(std::reduce(x.begin(), x.end()));
}

int main() {
  int num_arguments = 100, count_under = 0, num_iterations = 12*10000000, a=-1, b=1;


  sycl::queue queue{sycl::default_selector_v};
  sycl::buffer buffer{&count_under, sycl::range(1)}; 

  queue.submit([&](sycl::handler &cgh) {
    auto acc = buffer.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for(sycl::range<1>(num_iterations), [=](sycl::id<1> idx) {
      oneapi::dpl::minstd_rand gen(83734727, idx);
      oneapi::dpl::uniform_real_distribution<double> dis(a, b);
      std::vector<double> x(num_arguments-1);
      std::for_each(x.begin(), x.end(), [&](double &val) {
        val = dis(gen);
      });
      acc[0] += f(x) > dis(gen);
    }); 
  });

  queue.wait();

  boost::multiprecision::cpp_bin_float_quad V = std::pow(b-a, num_arguments);
  boost::multiprecision::cpp_bin_float_quad result = V * count_under/num_iterations;

  std::cout << result << " " << count_under << '\n';


}
