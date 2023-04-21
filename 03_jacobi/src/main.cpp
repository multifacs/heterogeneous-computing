#include <cstdlib>
#include <iostream>
#include <string_view>

#include <CL/sycl.hpp>
#include <omp.h>

#include "jacobi.hpp"

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Wrong arguments passed. Exiting." << std::endl;
        return 1;
    }

    std::cout << " --- Jacoby method ---\n";

    size_t rowsCount = static_cast<size_t>(std::atoi(argv[1]));
    float accuracyTarget = std::atof(argv[2]);
    int iterationsLimit = std::atoi(argv[3]);
    std::string_view deviceType = argv[4];

    sycl::queue queue = utils::createDeviceQueueByType(deviceType);
    std::cout << " * Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl << std::endl;

    auto [A, b] = utils::generateEquationSystem(rowsCount);

    {
        auto result = jacobi::AccessorJacoby(A, b, iterationsLimit, accuracyTarget, queue);
    } 

    {
        auto result = jacobi::AccessorJacoby(A, b, iterationsLimit, accuracyTarget, queue);
        float deviation = utils::deviation(A, b, result.x);
        std::cout << " -- Using accessor\n\tTime: " << result.elapsed_all << " ms\n\tDeviation: " << deviation << std::endl;
    }

    {
        auto result = jacobi::SharedMemoryJacoby(A, b, iterationsLimit, accuracyTarget, queue);
    }

    {
        auto result = jacobi::SharedMemoryJacoby(A, b, iterationsLimit, accuracyTarget, queue);
        float deviation = utils::deviation(A, b, result.x);
        std::cout << " -- Using shared\n\tTime: " << result.elapsed_all << " ms\n\tDeviation: " << deviation << std::endl;
    }

    {
        auto result = jacobi::DeviceMemoryJacoby(A, b, iterationsLimit, accuracyTarget, queue);
    }
    {
        auto result = jacobi::DeviceMemoryJacoby(A, b, iterationsLimit, accuracyTarget, queue);
        float deviation = utils::deviation(A, b, result.x);
        std::cout << " -- Using device\n\tTime: " << result.elapsed_all << " ms\n\tDeviation: " << deviation << std::endl;
    }
}
