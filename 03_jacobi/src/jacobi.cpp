#include "../include/jacobi.hpp"

#include <omp.h>

#include "../include/utils.hpp"

namespace jacobi {

Result calcAccessor(const std::vector<float> &A, const std::vector<float> &b,
                    int iterationsLimit, float accuracyTarget,
                    sycl::queue &queue) {
  Result result;
  result.iter = 0;
  result.accuracy = 0;
  std::vector<float> vec_x0;
  std::vector<float> vec_x1 = b;
  sycl::buffer<float> buffer_a(A.data(), A.size());
  sycl::buffer<float> buffer_b(b.data(), b.size());
  sycl::buffer<float> buffer_x0(vec_x0.data(), b.size());
  sycl::buffer<float> buffer_x1(vec_x1.data(), b.size());

  size_t globalSize = b.size();

  double begin = omp_get_wtime();

  {
    do {
      vec_x0 = vec_x1;
      sycl::event event = queue.submit([&](sycl::handler &h) {
        auto handler_a =
            buffer_a.get_access<sycl::access::mode::read,
                                sycl::access::target::constant_buffer>(h);
        auto handler_b =
            buffer_b.get_access<sycl::access::mode::read,
                                sycl::access::target::constant_buffer>(h);
        auto handler_x0 =
            buffer_x0.get_access<sycl::access::mode::read_write>(h);
        auto handler_x1 =
            buffer_x1.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(globalSize), [=](sycl::item<1> item) {
          int i = item.get_id(0);
          int n = item.get_range(0);
          float s = 0;
          for (int j = 0; j < n; j++)
            s += i != j ? handler_a[j * n + i] * handler_x0[j] : 0;
          handler_x1[i] = (handler_b[i] - s) / handler_a[i * n + i];
          handler_x0[i] = handler_x1[i];
        });
      });
      queue.wait();

      auto start =
          event
              .get_profiling_info<sycl::info::event_profiling::command_start>();
      auto end =
          event.get_profiling_info<sycl::info::event_profiling::command_end>();
      result.elapsed_kernel += (end - start) / 1e+6;

      result.accuracy = utils::norm(vec_x0, vec_x1);
      result.iter++;
    } while (result.iter < iterationsLimit && result.accuracy > accuracyTarget);
  }

  double end = omp_get_wtime();

  result.elapsed_all = (end - begin) * 1000.;
  result.x = vec_x1;

  return result;
}

Result calcShared(const std::vector<float> &A, const std::vector<float> &b,
                  int iterationsLimit, float accuracyTarget,
                  sycl::queue &queue) {
  Result result;
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

    auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
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

Result calcDevice(const std::vector<float> &A, const std::vector<float> &b,
                  int iterationsLimit, float accuracyTarget,
                  sycl::queue &queue) {
  Result result;
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

    auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
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

}  // namespace jacobi
