#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/helper_cuda.hpp"
#define L(a)                                                                   \
  printf(__FILE__ ":%d"                                                        \
                  "] " #a ":",                                                 \
         __LINE__);                                                            \
  print(a);                                                                    \
  printf("\n");
#define L0(a)                                                                  \
  if (thread0()) {                                                             \
    L(a);                                                                      \
  }

const int M = 512, N = 256, K = 128;
const int BM = 128, BN = 128, BK = 8;

__global__ void gemm_device(float* A, float* B, float* C){
  using namespace cute; 
  auto prob_shape = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(N, Int<1>{});
  auto dC = make_stride(N, Int<1>{});

  auto bM = Int<BM>{};
  auto bN = Int<BN>{};
  auto bK = Int<BK>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
  auto sB = make_layout(make_shape(bK, bN), LayoutRight{});
  auto sC = make_layout(make_shape(bM, bN), LayoutRight{});
  
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(prob_shape), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(prob_shape), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(prob_shape), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  


  
  
  
}

void my_gemm(float *A, float *B, float *C) { 
  using namespace cute;
  dim3 dimBlock(256);
  dim3 dimGrid(ceil_div(M, BM), ceil_div(N, BN));
  gemm_device<<<dimGrid, dimBlock, 0>>>(A, B, C);
    






  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(N, Int<1>{});
  auto dC = make_stride(N, Int<1>{});
  


  
  
  auto bM = Int<128>{};
  auto bN = Int<128>{}; 
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  

  auto sA = make_layout(make_shape())

}
int main() {
  using namespace cute;

  // cute::device_init(0);

  thrust::host_vector<float> h_A(M * K);
  thrust::host_vector<float> h_B(K * N);
  thrust::host_vector<float> h_C(M * N);
  for (int j = 0; j < M * K; ++j)
    h_A[j] = j; // static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < N * K; ++j)
    h_B[j] = j; // static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < M * N; ++j)
    h_C[j] = static_cast<float>(-1);

  thrust::device_vector<float> d_A = h_A;
  thrust::device_vector<float> d_B = h_B;
  thrust::device_vector<float> d_C = h_C;
  my_gemm(d_A.data().get(), d_B.data().get(), d_C.data().get());

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(Int<1>{}, 1000);
  L(prob_shape);
  L(dA);
  // CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(prob_shape), dA));
  // auto dA = make_stride(Int<1>{}, M);
  // auto dA = make_stride(Int<1>{}, M);
  L(prob_shape);
}
