#include <dr/mhp.hpp>
#include <complex>
#include <random>

namespace mhp = dr::mhp;

int main() {
  
  mhp::init(sycl::cpu_selector_v);

  mhp::distributed_vector<std::complex<double>> dv(128);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  mhp::for_each(dv, [&dis, &gen](std::complex<double> &val) { 
    val = std::complex<double>(dis(gen), 0);
    std::cout << val << '\n';
  });

}
