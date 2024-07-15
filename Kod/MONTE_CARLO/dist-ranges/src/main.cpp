#include <dr/mhp.hpp>
#include <fmt/core.h>
#include <oneapi/dpl/random>
#include <boost/multiprecision/cpp_bin_float.hpp>

namespace mhp = dr::mhp;

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(),
                [](double &val) { val = std::pow(val, 20); });
  return std::sin(std::reduce(x.begin(), x.end()));
}

int main(int argc, char **argv) {

  mhp::init(sycl::cpu_selector_v);

  int num_arguments = 100, count_under = 0, num_iterations = 12 * 10000000,
      a = -1, b = 1;

  mhp::distributed_vector<double> A(num_iterations);

  mhp::iota(A, 0);

  mhp::for_each(A, [=](auto &elem) {
    oneapi::dpl::minstd_rand gen(83734727, elem);
    oneapi::dpl::uniform_real_distribution<double> dis(a, b);
    std::vector<double> x(num_arguments - 1);
    std::for_each(x.begin(), x.end(), [&](double &value) { value = dis(gen); });
    elem = f(x) > dis(gen);
  });

  mhp::barrier();

  count_under = mhp::reduce(A);

  
  if (mhp::rank() == 0) {
    boost::multiprecision::cpp_bin_float_quad V = std::pow(b-a, num_arguments);
    boost::multiprecision::cpp_bin_float_quad result = V * count_under/num_iterations;
    std::cout << result << " " << count_under << '\n';
  }

  mhp::finalize();

  return 0;

}
