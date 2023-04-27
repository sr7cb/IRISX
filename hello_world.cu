#include <iris/iris.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

char a[12] = "hello world";
char b[12];
size_t size = 12;

extern "C" __global__ void uppercase(char* b, char* a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (a[i] >= 'a' && a[i] <= 'z') b[i] = a[i] + 'A' - 'a';
  else b[i] = a[i];
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