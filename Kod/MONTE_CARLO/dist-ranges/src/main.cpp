#include <dr/mhp.hpp>
#include <fmt/core.h>
#include <oneapi/dpl/random>

namespace mhp = dr::mhp;

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(),
                [](double &val) { val *= std::pow(val, 5); });
  return std::exp(std::reduce(x.begin(), x.end()));
}

int main(int argc, char **argv) {

  mhp::init(sycl::cpu_selector_v);

  int num_arguments = 120, count_under = 0, num_iterations = 12 * 10000000,
      a = -1, b = 1;

  mhp::distributed_vector<double> A(num_iterations);

  mhp::iota(A, 0);

  mhp::for_each(A, [=](auto &elem) {
    oneapi::dpl::minstd_rand gen(83734727, elem);
    oneapi::dpl::uniform_real_distribution<double> dis(a, b);
    oneapi::dpl::uniform_real_distribution<double> dis_2(-1.0, 100000000.0);
    std::vector<double> x(num_arguments - 1);
    std::for_each(x.begin(), x.end(), [&](double &value) { value = dis(gen); });
    elem = f(x) > dis_2(gen);
  });

  mhp::barrier();

  count_under = mhp::reduce(A);

  double V = std::pow(b-a, num_arguments);
  double result = V * count_under/num_iterations;
  
  if (mhp::rank() == 0) {
    std::cout << result << " " << count_under << '\n';
  }

  mhp::finalize();

  return 0;

}
