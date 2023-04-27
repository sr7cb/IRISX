#include <iris/iris.hpp>
#include <iris/iris_openmp.h>
#include <include/cpubackend.hpp>
#include <stdio.h>

char a[12] = "hello world";
char b[12];
size_t size = 12;

static void uppercase(char* b, char* a, IRIS_OPENMP_KERNEL_ARGS) {
  int i = 0;
#pragma omp parallel for shared(b, a) private(i)
  IRIS_OPENMP_KERNEL_BEGIN(i)
  if (a[i] >= 'a' && a[i] <= 'z') b[i] = a[i] + 'A' - 'a';
  else b[i] = a[i];
  IRIS_OPENMP_KERNEL_END
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, true);

  iris::Mem mem_a(size);
  iris::Mem mem_b(size);

  iris::Task task;
  task.h2d(&mem_a, 0, size, a);
  void* params[2] = { &mem_b, &mem_a };
  int params_info[2] = { iris_w, iris_r };
  task.kernel("uppercase", 1, NULL, &size, NULL, 2, params, params_info);
  task.d2h(&mem_b, 0, size, b);
  task.submit(iris_roundrobin, NULL, true);

  printf("%s\n", b);

  platform.finalize();

  return 0;
}