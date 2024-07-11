#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

double f(std::vector<double> &x) {
  std::for_each(x.begin(), x.end(),
                [](double &val) { val *= std::pow(val, 5); });
  return std::exp(std::reduce(x.begin(), x.end()));
}

int main() {

  int num_arguments = 120, count_under = 0, num_iterations = 12 * 10000000,
      a = -1, b = 1;

  std::vector<double> x(num_arguments);
  std::random_device rd;
  std::minstd_rand gen(rd());
  std::uniform_real_distribution<double> dis(a, b);
  std::uniform_real_distribution<double> dis_2(-1.0, 100000000.0);

  double V = std::pow(b - a, num_arguments);

  for (int i = 0; i < num_iterations; ++i) {
    std::for_each(x.begin(), x.end(), [&](double &val) { val = dis(gen); });
    count_under += f(x) > dis_2(gen);
  }

  double result = V * count_under / num_iterations;
  std::cout << result << " " << count_under << " ";

  return 0;
}
