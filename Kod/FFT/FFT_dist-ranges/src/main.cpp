#include <complex>
#include <dr/mhp.hpp>
#include <fmt/core.h>
#include <oneapi/dpl/random>

#define J std::complex<double>(0.0, 1.0)

namespace mhp = dr::mhp;

void gen_data(mhp::distributed_vector<std::complex<double>> &vec) {
  int size = vec.size();
  mhp::distributed_vector<int> idx(size);
  mhp::iota(idx, 0);

  mhp::for_each(mhp::views::zip(vec, idx), [](auto &&elem) {
    auto &&[val, i] = elem;
    oneapi::dpl::minstd_rand engine(247246, i);
    oneapi::dpl::uniform_real_distribution<double> distr(0.0, 1.0);
    val = std::complex<double>(distr(engine), 0.);
  });
}

void fft(mhp::distributed_vector<std::complex<double>> &vec) {
  int size = vec.size();
  int num_bits = std::log2(size);

  for (int step = 1; step <= num_bits; ++step) {
    int step_size = 1 << step;
    std::complex<double> omega =
        std::exp((2 * M_PI * J) / static_cast<double>(step_size));
    for (int start = 0; start < size; start += step_size) {
      auto even = vec | mhp::views::drop(start) | mhp::views::take(step_size/2);
      auto odd = vec | mhp::views::drop(start + step_size/2) | mhp::views::take(step_size/2);
      auto zipped = mhp::views::zip(even, odd);
      mhp::for_each(zipped, [=](auto &&elem) {
        auto &&[e, o] = elem;
        std::complex<double> pow = omega;
        std::complex<double> temp = e;
        e = e + pow * o;
        o = temp - pow * o;
      });
    }
  }
}

int main(int argc, char **argv) {
  mhp::init(sycl::cpu_selector_v);

  int size = 128;

  mhp::distributed_vector<std::complex<double>> vec(size);

  gen_data(vec);

  fft(vec);

  for (int i = 0; i < size; ++i) {
    std::complex<double> a = vec[i];
    fmt::print("{}\t", a.real());
  }
  fmt::print("\n");

  mhp::finalize();
}
