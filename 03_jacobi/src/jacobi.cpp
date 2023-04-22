#include "../include/jacobi.hpp"

#include <omp.h>

#include "../include/utils.hpp"

namespace jacobi {

CompResult calculateWithAccessor(const std::vector<float> &matrixA, const std::vector<float> &vectorB, int iterLimit,
                                 float accuracy, sycl::queue &deviceQueue) {
    CompResult compResult;
    compResult.iter = 0;
    compResult.accuracy = 0;

    std::vector<float> x0Vector;
    std::vector<float> x1Vector = vectorB;

    sycl::buffer<float> bufferA(matrixA.data(), matrixA.size());
    sycl::buffer<float> bufferB(vectorB.data(), vectorB.size());
    sycl::buffer<float> bufferX0(x0Vector.data(), vectorB.size());
    sycl::buffer<float> bufferX1(x1Vector.data(), vectorB.size());

    size_t globalSize = vectorB.size();

    double startTime = omp_get_wtime();

    {
        do {
            x0Vector = x1Vector;

            sycl::event deviceEvent = deviceQueue.submit([&](sycl::handler &handler) {
                auto handleA =
                    bufferA.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(handler);
                auto handleB =
                    bufferB.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(handler);
                auto handleX0 = bufferX0.get_access<sycl::access::mode::read_write>(handler);
                auto handleX1 = bufferX1.get_access<sycl::access::mode::read_write>(handler);

                handler.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
                    int indexI = item.get_id(0);
                    int rangeN = item.get_range(0);
                    float sum = 0;
                    for (int indexJ = 0; indexJ < rangeN; indexJ++)
                        sum += indexI != indexJ ? handleA[indexJ * rangeN + indexI] * handleX0[indexJ] : 0;
                    handleX1[indexI] = (handleB[indexI] - sum) / handleA[indexI * rangeN + indexI];
                    handleX0[indexI] = handleX1[indexI];
                });
            });
            deviceQueue.wait();

            auto startTimestamp = deviceEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto endTimestamp = deviceEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
            compResult.elapsed_kernel += (endTimestamp - startTimestamp) / 1e+6;

            compResult.accuracy = utils::norm(x0Vector, x1Vector);
            compResult.iter++;
        } while (compResult.iter < iterLimit && compResult.accuracy > accuracy);
    }

    double endTime = omp_get_wtime();

    compResult.elapsed_all = (endTime - startTime) * 1000.;
    compResult.x = x1Vector;

    return compResult;
}

CompResult calculateWithSharedMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue) {
    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    size_t bSize = globalSize * sizeof(float);

    float *aShared = sycl::malloc_shared<float>(A.size(), queue);
    float *bShared = sycl::malloc_shared<float>(b.size(), queue);
    float *x0Shared = sycl::malloc_shared<float>(b.size(), queue);
    float *x1Shared = sycl::malloc_shared<float>(b.size(), queue);

    queue.memcpy(aShared, A.data(), A.size() * sizeof(float)).wait();
    queue.memcpy(bShared, b.data(), bSize).wait();
    queue.memcpy(x1Shared, b.data(), bSize).wait();

    std::vector<float> x0(globalSize);
    std::vector<float> x1(globalSize);

    double begin = omp_get_wtime();
    do {
        queue.memcpy(x0Shared, x1Shared, bSize).wait();
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
        queue.memcpy(x0.data(), x0Shared, bSize).wait();
        queue.memcpy(x1.data(), x1Shared, bSize).wait();

        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        result.elapsed_kernel += (end - start) / 1e+6;

        result.accuracy = utils::norm(x0, x1);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    sycl::free(aShared, queue);
    sycl::free(bShared, queue);
    sycl::free(x0Shared, queue);
    sycl::free(x1Shared, queue);

    result.elapsed_all = (end - begin) * 1000.;
    result.x = x1;

    return result;
}

CompResult calculateWithDeviceMemory(const std::vector<float> &A, const std::vector<float> &b, int iterationsLimit,
                                     float accuracyTarget, sycl::queue &queue) {
    CompResult result;
    result.iter = 0;
    result.accuracy = 0;

    size_t globalSize = b.size();
    size_t bSize = globalSize * sizeof(float);

    float *aDevice = sycl::malloc_device<float>(A.size(), queue);
    float *bDevice = sycl::malloc_device<float>(b.size(), queue);
    float *x0Device = sycl::malloc_device<float>(b.size(), queue);
    float *x1Device = sycl::malloc_device<float>(b.size(), queue);

    queue.memcpy(aDevice, A.data(), A.size() * sizeof(float)).wait();
    queue.memcpy(bDevice, b.data(), bSize).wait();
    queue.memcpy(x1Device, b.data(), bSize).wait();

    std::vector<float> x0(globalSize);
    std::vector<float> x1(globalSize);

    double begin = omp_get_wtime();
    do {
        queue.memcpy(x0Device, x1Device, bSize).wait();
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
        queue.memcpy(x0.data(), x0Device, bSize).wait();
        queue.memcpy(x1.data(), x1Device, bSize).wait();

        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        result.elapsed_kernel += (end - start) / 1e+6;

        result.accuracy = utils::norm(x0, x1);
        result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
    double end = omp_get_wtime();

    sycl::free(aDevice, queue);
    sycl::free(bDevice, queue);
    sycl::free(x0Device, queue);
    sycl::free(x1Device, queue);

    result.elapsed_all = (end - begin) * 1000.;
    result.x = x1;

    return result;
}

} // namespace jacobi
