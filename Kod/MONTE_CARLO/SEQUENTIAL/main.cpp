#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <quadmath.h>

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(),
                [](double &val) { val = std::pow(val, 20); });
  return std::sin(std::reduce(x.begin(), x.end()));
}

int main() {

  int num_dimensions = 100, count_under = 0, num_iterations = 12 * 10000000,
      a = -1, b = 1;

  std::vector<double> x(num_dimensions - 1);
  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<double> dis(a, b);

  for (int i = 0; i < num_iterations; ++i) {
    std::for_each(x.begin(), x.end(), [&](double &val) { val = dis(gen); });
    count_under += f(x) > dis(gen);
  }

  __float128 V = std::pow(b - a, num_dimensions);
  __float128 result = V * count_under / num_iterations;
  char buffer[128];
  quadmath_snprintf(buffer, sizeof(buffer), "%.5Qe", result);

  std::cout << buffer << " " << count_under << " ";

  return 0;
}
