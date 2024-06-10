#include <dr/mhp.hpp>
#include <fmt/core.h>
#include <vector>

namespace mhp = dr::mhp;

int read_data_from_files(std::vector<double> &vector) {
  int size;

  std::ifstream vectorFile("../../../TEST_DATA/CG_test_vector",
                           std::ios::binary);
  if (!vectorFile.is_open()) {
    std::cerr << "Error: Failed to open vector file for reading.\n";
    return -1;
  }

  vectorFile.seekg(0, std::ios::end);
  std::streampos fileSize = vectorFile.tellg();
  vectorFile.seekg(0, std::ios::beg);

  size = fileSize / sizeof(double);

  vector.resize(size);

  vectorFile.read(reinterpret_cast<char *>(vector.data()), fileSize);
  vectorFile.close();

  return size;
}

template <dr::distributed_range X, dr::distributed_range Y>
void vector_combination(X &&vec, Y &&vec2, double mult) {
  auto zipped = dr::mhp::views::zip(vec, vec2);
  mhp::for_each(zipped, [=](auto &&elem) {
    auto &&[a, b] = elem;
    a = a + b * mult;
  });
}

template <dr::distributed_range X, dr::distributed_range Y>
double inner_product(X &&x, Y &&y) {
  auto zipped =
      dr::mhp::views::zip(x, y) | dr::mhp::views::transform([](auto &&elem) {
        auto &&[a, b] = elem;
        return a * b;
      });

  return mhp::reduce(zipped);
}

template <dr::distributed_range X> double norm(X &&vec) {
  double inner_prod = inner_product(vec, vec);
  return std::sqrt(inner_prod);
}

template <dr::distributed_range X, dr::distributed_range Y,
          dr::distributed_range Z>
void matrix_vector_multiply(X &&mat, Y &&vec, Z &&result, int size) {
  //Tutaj dzieje się dziwna rzecz kiedy w pętli dam i=0 dlatego 0 wydzielone przed pętle.
  result[0] = inner_product(mat, vec);
  for (int i = 1; i < size; ++i) {
    auto row_view =
        rng::subrange(mat.begin() + i * size, mat.begin() + (i + 1) * size);
    result[i] = inner_product(row_view, vec);
  }
}

int main(int argc, char **argv) {

  mhp::init(sycl::cpu_selector_v);

  int size = 12;

  mhp::distributed_vector<double> mat1(size * size);
  mhp::distributed_vector<double> res(size);
  mhp::distributed_vector<double> dv1(size);
  mhp::distributed_vector<double> dv2(size);

  mhp::iota(dv1, 1);
  mhp::iota(dv2, 49);
  mhp::iota(mat1, 1);

  if (mhp::rank() == 0) {
    fmt::print("mat1:\n");
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        fmt::print("{}\t", mat1[i * size + j]);
      }
      fmt::print("\n");
    }
    fmt::print("dv1:\n");
    for (int i = 0; i < size; ++i) {
      fmt::print("{}\t", dv1[i]);
    }
    fmt::print("\n");

    fmt::print("dv2:\n");
    for (int i = 0; i < size; ++i) {
      fmt::print("{}\t", dv2[i]);
    }
    fmt::print("\n\n");
  }

  matrix_vector_multiply(mat1, dv1, res, size);
  double inner_prod = inner_product(dv1, dv2);
  double vec_norm = norm(dv1);
  vector_combination(dv1, dv2, 0.5);
  if (mhp::rank() == 0) {
    fmt::print("inner_prod(dv1, dv2): {}\n", inner_prod);
    std::cout << "vec_norm(dv1) before combination: " << vec_norm << "\n";
    fmt::print("dv1 after combination:\n");
    for (int i = 0; i < size; ++i) {
      fmt::print("{}\t", dv1[i]);
    }
    fmt::print("\n");
  }
  vec_norm = norm(dv1);
  if (mhp::rank() == 0)
    std::cout << "vec_norm(dv1) after combination: " << vec_norm << "\n";
  if (mhp::rank() == 0) {
    fmt::print("res:\n");
    for (int i = 0; i < size; ++i) {
      fmt::print("{}\t", res[i]);
    }
    std::cout << "\n";
  }

  mhp::finalize();

  /*   mhp::init(sycl::cpu_selector_v); */

  /*   mhp::distributed_vector<char> dv(81); */
  /*   std::string decoded_string(80, 0); */

  /*   mhp::copy( */
  /*       0, */
  /*       std::string("Mjqqt%|twqi&%Ymnx%nx%ywfsxrnxnts%kwtr%ymj%tsj%fsi%tsq~%"
   */
  /*                   "Inxywngzyji%Wfsljx%wjfqr&"), */
  /*       dv.begin()); */

  /*   mhp::for_each(dv, [](char &val) { val -= 5; }); */
  /*   mhp::copy(0, dv, decoded_string.begin()); */

  /*   if (mhp::rank() == 0) */
  /*     fmt::print("{}\n", decoded_string); */

  /*   mhp::finalize(); */

  return 0;
}
