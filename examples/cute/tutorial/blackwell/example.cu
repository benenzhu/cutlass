#include <stdio.h>
#include <cuda.h>
__global__ void report_latency(int *dst, int *src)
{
auto stt = clock64();
int val = *src;
__threadfence();
// __syncwarp();
auto mid = clock64();
*dst = val;
__threadfence();
// __syncwarp();
auto end = clock64();
__threadfence();
// __syncwarp();
printf("read took %lld cycles, write took %lld cycles\n", mid -stt, end - mid);
}
void latency_test()
{
int *dst, *src;
cudaMalloc(&dst, sizeof(int));
cudaMalloc(&src, sizeof(int));
report_latency<<<1, 1>>>(dst, src);
cudaDeviceSynchronize();
cudaFree(dst);
dst = nullptr;
cudaFree(src);
src = nullptr;
}
__global__ void embrassingly_parallel(int4 *a, int4 *b, int4 *c, size_t n_fields)
{
for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < n_fields; i += gridDim.x * blockDim.x)
{
int4 a_field = a[i];
int4 b_field = b[i];
c[i] = make_int4(a_field.x + b_field.x, a_field.y + b_field.y, a_field.z + b_field.z, a_field.w + b_field.w);
}
}
void throughput_test()
{
constexpr size_t n_bytes = 512 * 1024 * 1024;
auto n_fields = n_bytes / sizeof(uint4);
int4 *a, *b, *c;
cudaMalloc(&a, n_bytes);
cudaMalloc(&b, n_bytes);
cudaMalloc(&c, n_bytes);
cudaEvent_t es, ee;
cudaEventCreate(&es);
cudaEventCreate(&ee);
cudaEventRecord(es);
embrassingly_parallel<<<48 * 3, 512>>>(a, b, c, n_fields);
cudaEventRecord(ee);
cudaEventSynchronize(ee);
float t_ms;
cudaEventElapsedTime(&t_ms, es, ee);
printf("effective memory bandwidth: %f GB/s\n", n_bytes * 3 / 1e9 / (t_ms / 1e3));
cudaEventDestroy(es);
cudaEventDestroy(ee);
cudaFree(a);
a = nullptr;
cudaFree(b);
b = nullptr;
cudaFree(c);
c = nullptr;
}
int main()
{
latency_test();
throughput_test();
return 0;
}
