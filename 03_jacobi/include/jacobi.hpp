#pragma once

#include <vector>

#include <CL/sycl.hpp>

namespace jacobi {

struct Result {
    std::vector<float> x;
    double elapsed_all;
    double elapsed_kernel = 0;
    int iter;
    float accuracy;
};

Result withAccessor(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                 float accuracyTarget, sycl::queue &queue);
Result withSharedMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);
Result withDeviceMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);

} // namespace jacobi
