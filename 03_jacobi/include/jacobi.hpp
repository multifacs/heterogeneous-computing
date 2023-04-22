#pragma once

#include <vector>

#include <CL/sycl.hpp>

namespace jacobi {

struct ResultStruct {
    std::vector<float> x;
    double elapsed_all;
    double elapsed_kernel = 0;
    int iter;
    float accuracy;
};

ResultStruct slauAccessor(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                 float accuracyTarget, sycl::queue &queue);
ResultStruct slauSharedMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);
ResultStruct slauDeviceMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);

} // namespace jacobi
