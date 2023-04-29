#include "../include/jacobi.hpp"

#include <omp.h>

#include "../include/utils.hpp"

namespace jacobi {

CompResult jacoby_accessors(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                            float accuracyTarget, sycl::queue &queue) {

    using sycl::buffer;

    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    std::vector<float> x_k(globalSize, 0.0);
    std::vector<float> x_k_1(b);

    buffer<float> a_buffer(A.data(), A.size());
    buffer<float> b_buffer(b.data(), b.size());
    buffer<float> x_k_buffer(x_k.data(), b.size());
    buffer<float> x_k_1_buffer(x_k_1.data(), b.size());

    auto p_x_k_buffer = &x_k_buffer;
    auto p_x_k_1_buffer = &x_k_1_buffer;

    double begin = omp_get_wtime();

    {
        do {
            sycl::event event = queue.submit([&](sycl::handler &h) {
                auto a_handle = a_buffer.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(h);
                auto b_handle = b_buffer.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(h);
                auto x_k_handle = p_x_k_buffer->get_access<sycl::access::mode::read>(h);
                auto x_k_1_Handle = p_x_k_1_buffer->get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                    int i = item.get_id(0);
                    int n = item.get_range(0);

                    float s = 0;

                    for (int j = 0; j < n; j++)
                        s += i != j ? a_handle[j * n + i] * x_k_handle[j] : 0;
                    x_k_1_Handle[i] = (b_handle[i] - s) / a_handle[i * n + i];
                });
            });
            queue.wait();

            std::swap(p_x_k_buffer, p_x_k_1_buffer);
            result.accuracy = utils::norm(x_k, x_k_1);
            //result.accuracy = 1;
            result.iter++;
        } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    }

    double end = omp_get_wtime();

    result.elapsed_all = (end - begin) * 1000.;
    result.x = x_k_1;
    return result;
}

CompResult jacoby_shared(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                         float accuracyTarget, sycl::queue &queue) {

    using sycl::malloc_shared;

    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    size_t bSize = globalSize * sizeof(float);

    float *aShared = malloc_shared<float>(A.size(), queue);
    float *bShared = malloc_shared<float>(b.size(), queue);
    float *x0Shared = malloc_shared<float>(b.size(), queue);
    float *x1Shared = malloc_shared<float>(b.size(), queue);

    queue.memcpy(aShared, A.data(), A.size() * sizeof(float)).wait();
    queue.memcpy(bShared, b.data(), bSize).wait();
    queue.memset(x0Shared, 0, bSize).wait();
    queue.memcpy(x1Shared, b.data(), bSize).wait();

    double begin = omp_get_wtime();
    do {
        sycl::event event = queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                int i = item.get_id(0);
                int n = item.get_range(0);
                float s = 0;
                for (int j = 0; j < n; j++)
                    s += i != j ? aShared[j * n + i] * x0Shared[j] : 0;
                x1Shared[i] = (bShared[i] - s) / aShared[i * n + i];
            });
        });
        queue.wait();

        std::swap(x0Shared, x1Shared);
        result.accuracy = utils::norm(x0Shared, x1Shared, globalSize);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    sycl::free(aShared, queue);
    sycl::free(bShared, queue);
    sycl::free(x0Shared, queue);
    sycl::free(x1Shared, queue);

    result.elapsed_all = (end - begin) * 1000.;
    result.x = std::vector<float>(globalSize, 0);
    memcpy(result.x.data(), x1Shared, bSize);

    return result;
}

CompResult jacoby_device(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                         float accuracyTarget, sycl::queue &queue) {

    using sycl::malloc_device;
    
    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    size_t bSize = globalSize * sizeof(float);

    float *aDevice = malloc_device<float>(A.size(), queue);
    float *bDevice = malloc_device<float>(globalSize, queue);
    float *x0Device = malloc_device<float>(globalSize, queue);
    float *x1Device = malloc_device<float>(globalSize, queue);

    queue.memcpy(aDevice, A.data(), A.size() * sizeof(float)).wait();
    queue.memcpy(bDevice, b.data(), bSize).wait();
    queue.memset(x0Device, 0, bSize).wait();
    queue.memcpy(x1Device, b.data(), bSize).wait();

    double begin = omp_get_wtime();
    do {
        sycl::event event = queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                int i = item.get_id(0);
                int n = item.get_range(0);
                float s = 0;
                for (int j = 0; j < n; j++)
                    s += i != j ? aDevice[j * n + i] * x1Device[j] : 0;
                x1Device[i] = (bDevice[i] - s) / aDevice[i * n + i];
            });
        });
        queue.wait();

        std::swap(x0Device, x1Device);
        result.accuracy = utils::norm(x0Device, x1Device, globalSize);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    sycl::free(aDevice, queue);
    sycl::free(bDevice, queue);
    sycl::free(x0Device, queue);
    sycl::free(x1Device, queue);

    result.elapsed_all = (end - begin) * 1000.;
    result.x = std::vector<float>(globalSize, 0);
    memcpy(result.x.data(), x1Device, bSize);

    return result;
}

} // namespace jacobi
