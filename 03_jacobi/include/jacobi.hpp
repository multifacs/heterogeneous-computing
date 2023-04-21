#pragma once

#include <vector>

#include <CL/sycl.hpp>

namespace jacobi {

struct CompResult {
    std::vector<float> x;
    double elapsed_all;
    double elapsed_kernel = 0;
    int iter;
    float accuracy;
};

CompResult jacoby_accessors(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                 float accuracyTarget, sycl::queue &queue);
CompResult jacoby_shared(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);
CompResult jacoby_device(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue);

} // namespace jacobi
