#include <omp.h>

#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "jacobi.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cout << "Wrong arguments passed. Exiting." << std::endl;
    return 1;
  }

  size_t rowsCount = static_cast<size_t>(std::atoi(argv[1]));
  float accuracyTarget = std::atof(argv[2]);
  int iterationsLimit = std::atoi(argv[3]);
  std::string_view deviceType = argv[4];

  sycl::queue queue = utils::createDeviceQueueByType(deviceType);
  std::cout << "Device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  auto [A, b] = utils::generateEquationSystem(rowsCount);

  {
    auto result = jacobi::calcAccessor(A, b, iterationsLimit,
                                                accuracyTarget, queue);
  }

  {
    auto result = jacobi::calcAccessor(A, b, iterationsLimit,
                                                accuracyTarget, queue);
    float deviation = utils::deviation(A, b, result.x);
    std::cout << " - Accessor\n";
    std::cout << "   Time all    : " << result.elapsed_all << " ms \n";
    std::cout << "   Time kernel : " << result.elapsed_kernel << " ms\n";
    std::cout << "   Error       : " << result.accuracy << "\n\n";
  }

  {
    auto result = jacobi::calcShared(A, b, iterationsLimit,
                                                    accuracyTarget, queue);
  }

  {
    auto result = jacobi::calcShared(A, b, iterationsLimit,
                                                    accuracyTarget, queue);
    float deviation = utils::deviation(A, b, result.x);
    std::cout << " - Shared\n";
    std::cout << "   Time all    : " << result.elapsed_all << " ms \n";
    std::cout << "   Time kernel : " << result.elapsed_kernel << " ms\n";
    std::cout << "   Error       : " << result.accuracy << "\n\n";
  }

  {
    auto result = jacobi::calcDevice(A, b, iterationsLimit,
                                                    accuracyTarget, queue);
  }
  {
    auto result = jacobi::calcDevice(A, b, iterationsLimit,
                                                    accuracyTarget, queue);
    float deviation = utils::deviation(A, b, result.x);
    std::cout << " - Device\n";
    std::cout << "   Time all    : " << result.elapsed_all << " ms \n";
    std::cout << "   Time kernel : " << result.elapsed_kernel << " ms\n";
    std::cout << "   Error       : " << result.accuracy << "\n\n";
  }
}
