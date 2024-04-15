#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

typedef std::complex<double> Complex;

int main() {
  std::ifstream inputFile_1("output_1.dat", std::ios::binary);
  if (!inputFile_1.is_open()) {
    std::cerr << "Error: couldn't open output_1.dat.\n";
    return -1;
  }

  inputFile_1.seekg(0, std::ios::end);
  std::streampos fileSize = inputFile_1.tellg();
  inputFile_1.seekg(0, std::ios::beg);

  size_t size = fileSize / sizeof(Complex);

  std::vector<Complex> array_1(size);
  inputFile_1.read(reinterpret_cast<char *>(array_1.data()), fileSize);

  inputFile_1.close();

  std::ifstream inputFile_2("output_2.dat", std::ios::binary);
  if (!inputFile_2.is_open()) {
    std::cerr << "Error: couldn't open output_2.dat.\n";
    return -1;
  }

  inputFile_2.seekg(0, std::ios::end);
  std::streampos fileSize_2 = inputFile_2.tellg();
  inputFile_2.seekg(0, std::ios::beg);

  size_t size_2 = fileSize_2 / sizeof(Complex);

  std::vector<Complex> array_2(size_2);
  inputFile_2.read(reinterpret_cast<char *>(array_2.data()), fileSize);

  inputFile_2.close();


  for (size_t i=0; i<size; i++) {
    if (array_1[i] != array_2[i]) {
      std::cout << std::setprecision(8) << std::fixed << array_1[i] << "\t\t" << array_2[i] << '\n';
    }
  }

}
