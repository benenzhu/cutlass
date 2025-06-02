#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>

#include <cute/tensor.hpp>

// #include "cutlass/util/helper_cuda.hpp"
// ... existing code ...

// Modified L macro to be a function that returns the value it prints
template <typename T>
T ZTY_PRINT(T a, std::string s, std::string file, int line) {
  std::cout << file <<":"<< line <<
                  "] \e[34m" << s<< ": \e[36m";
  cute::print(a);
  printf("\e[0m\n");
  return a;
}



// ... existing code ...
#define L(a)                                                                   \
  ZTY_PRINT(a, #a, __FILE__, __LINE__);
  // printf(__FILE__ ":%d"                                                        \
  //                 "] " #a ": ",                                                 \
  //        __LINE__);                                                            \
  // print(a);                                                                    \
  // printf("\n");
#define L0(a)                                                                  \
  if (thread0()) {                                                             \
    L(a);                                                                      \
  }

#define LP(a)                                                                  \
  printf(__FILE__ ":%d"                                                        \
                  "] \e[34m" #a ": \e[36m" ,                                               \
         __LINE__);                                                            \
  a;                                                                           \
  printf("\e[0m\n");


#define LA(a, b)                                                                  \
  printf(__FILE__ ":%d"                                                        \
                  "] " #a " = \e[34m" #b ": \e[36m" ,                                               \
         __LINE__);                                                            \
  auto a = b;                                                                           \
  cute::print(a);                                                               \
  printf("\e[0m\n");
const int M = 512, N = 256, K = 128;
const int BM = 128, BN = 128, BK = 8; 

// __global__ void gemm_device(float* A, float* B, float* C){
//   using namespace cute; 
//   auto prob_shape = make_shape(M, N, K);
//   auto dA = make_stride(K, Int<1>{});
//   auto dB = make_stride(N, Int<1>{});
//   auto dC = make_stride(N, Int<1>{});

//   auto bM = Int<BM>{};
//   auto bN = Int<BN>{};
//   auto bK = Int<BK>{};
//   auto cta_tiler = make_shape(bM, bN, bK);
//   auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
//   auto sB = make_layout(make_shape(bK, bN), LayoutRight{});
//   auto sC = make_layout(make_shape(bM, bN), LayoutRight{});
  
//   auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
//   auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});
//   auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  
//   Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(prob_shape), dA);
//   Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(prob_shape), dB);
//   Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(prob_shape), dC);

//   auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
//   Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
//   Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  


  
  
  
// }

// void my_gemm(float *A, float *B, float *C) { 
//   using namespace cute;
//   dim3 dimBlock(256);
//   dim3 dimGrid(ceil_div(M, BM), ceil_div(N, BN));
//   gemm_device<<<dimGrid, dimBlock, 0>>>(A, B, C);
    






//   auto dA = make_stride(K, Int<1>{});
//   auto dB = make_stride(N, Int<1>{});
//   auto dC = make_stride(N, Int<1>{});
//   auto bM = Int<128>{};
//   auto bN = Int<128>{}; 
//   auto bK = Int<8>{};
//   auto cta_tiler = make_shape(bM, bN, bK);
  

//   auto sA = make_layout(make_shape())

// }
// 
// 

using namespace cute;
template<class Shape, class Stride>
void print2D(Layout<Shape, Stride> const& layout){
  for(int m = 0; m < size<0>(layout); ++m){
    for(int n = 0; n < size<1>(layout); ++n){
      printf("%3d   ", layout(m, n));
    }
    printf("\n");
  }
}
using namespace cute;
template<class Shape, class Stride>
void print1D(Layout<Shape, Stride> const& layout){
  for(int m = 0; m < size<0>(layout); ++m){
    printf("%3d   ", layout(m));
  }
  printf("\n");
}
int main() {

  L(int{2}); // c++里面的用法
  L(Int<2>{});
  L(make_shape(Int<1>{}, Int<2>{}));
  
  /************* tuple ***************/
  auto t = L(make_tuple(uint16_t{42}, make_tuple(Int<1>{}, int32_t{3}), Int<17>{}));
  L(t);
  L(rank(t));
  L(get<1>(t)); 
  L(depth(t)); // _2
  L(size(t)); // 2412 = 42 * 3 * 17
  auto s8 = L(make_layout(Int<8>{}));
  auto d8 = L(make_layout(8));
  Layout s2xs4 = L(make_layout(make_shape(Int<2>{},Int<4>{})));
  Layout s2xd4 = L(make_layout(make_shape(Int<2>{},4)));

  Layout s2xd4_a = L(make_layout(make_shape (Int< 2>{},4), make_stride(Int<12>{},Int<1>{})));
  Layout s2xd4_col = L(make_layout(make_shape(Int<2>{},4), LayoutLeft{}));
  Layout s2xd4_row = L(make_layout(make_shape(Int<2>{},4),
                                 LayoutRight{}))

  Layout s2xh4 = L(make_layout(make_shape (2,make_shape (2,2)), make_stride(4,make_stride(2,1))))
  Layout s2xh4_row = L(make_layout(shape(s2xh4), LayoutRight{}));
  Layout s2xh4_col = L(make_layout(shape(s2xh4), LayoutLeft{}));
  static_assert(congruent(shape(s2xh4), stride(s2xh4)));
  LP(print2D(s2xs4));
  LP(print2D(s2xh4_row));
  LP(print2D(s2xh4_col));
  L(s2xh4_col(1,2));
  LP(print_layout(s2xh4));
  auto vec = L(make_layout(make_shape(8), make_stride(4)));
  LP(print1D(vec)); // 1D vector
  
  // Creating the ((4,2)):((2,1)) layout
  auto special_layout = L(make_layout(make_shape(make_shape(4, 2)), make_stride(make_stride(2, 1))));
  LP(print1D(special_layout)); // ((2,4):(1,2))
  auto special_layout2 = L(make_layout(make_shape(make_shape(2, 4)), make_stride(make_stride(4, 1))));
  LP(print1D(special_layout2)); // ((4,2):(2,1))
  auto special_layout3 = L(make_layout(make_shape(make_shape(4, 2)), make_stride(make_stride(1, 4))));
  LP(print1D(special_layout3)); // ((4,2):(2,1))
  // printf("Layout ((4,2)):((2,1)) mapping:\n"); // !!!!!! 这个很难理解..........
  // 

  auto matrix_col = L(make_layout(make_shape(2, 4), make_stride(1, 2)));
  LP(print_layout(matrix_col));
  
  LA(matrix_row, make_layout(make_shape(2,4), make_stride(4, 1)));
  LP(print_layout(matrix_row));
  
  LA(matrix_specital, make_layout(make_shape(make_shape(2,2), 2), make_stride(make_stride(4, 1),2)))
  LP(print_layout(matrix_specital));
  // auto matrix_row = L(make_layout(make_shape(2,4), make_stride(2, 1))); 
  // print_layout(matrix_row);
  LA(la, make_layout(make_shape(3, make_shape(3, 3))));
 LP(print_layout(la));
  LA(shape, la.shape());
  L(idx2crd(25, shape));
  L(idx2crd(make_coord(1, 5), shape));
  L(idx2crd(make_coord(1, make_coord(1,2)), shape)); // 默认是按照 column major来进行的
  
  LA(la_right, (make_layout(make_shape(3, make_shape(3, 3)), LayoutRight{}))); 
  LP(print_layout(la_right ));
  LA(shape_r, la_right.shape());
  L(idx2crd(25, la_right.shape(), la_right.stride()));
  L(idx2crd(make_coord(1, 5), la_right.shape(), la_right.stride()));
  L(idx2crd(make_coord(1, make_coord(1,2)), la_right.shape(), la_right.stride())); // 默认是按照 column major来进行的
  L(crd2idx(25, la_right.shape(), la_right.stride()));
  L(crd2idx(make_coord(1, 5), la_right.shape(), la_right.stride()));
  L(crd2idx(make_coord(1, make_coord(1,2)), la_right.shape(), la_right.stride())); // 默认是按照 column major来进行的

  {
    LA(a, (Layout<Shape<_4,Shape<_3,_6>>>{}));
    LA(a0, layout<0>(a) );
    LA(a1, layout<1>(a));
    LA(a10, (layout<1, 0>(a)) );
    LA(a11, (layout<1,1>(a)));
  }
  
  {
    LA(a, (Layout<Shape<_2,_3,_5,_7>>{}));
    LA(a13, (select<1,3>(a)));
    LA(a01, (select<0,1,3>(a)));
    LA(a2, select<2>(a));
  }

  {
    LA(a , (Layout<Shape<_2,_3,_5,_7>>{}));     // (2,3,5,7):(1,2,6,30)
    LA(a13, (take<1,3>(a)));                     // (3,5):(2,6)
    LA(a14, (take<1,4>(a)));                     // (3,5,7):(2,6,30)
  }
  
  {
  Layout a = Layout<_3,_1>{};                     // 3:1
  Layout b = Layout<_4,_3>{};                     // 4:3
  Layout row = make_layout(a, b);                 // (3,4):(1,3)
  LP(print_layout(row));
  Layout col = make_layout(b, a);                 // (4,3):(3,1)
  LP(print_layout(make_layout(Layout<_4, _3>{}, Layout<_3, _1>{})));

  LP(print_layout(make_layout(row, col)));             // ((3,4),(4,3)):((1,3),(3,1))
  Layout aa  = make_layout(a);                    // (3):(1)
  Layout aaa = make_layout(aa);                   // ((3)):((1))
  Layout d   = make_layout(a, make_layout(a), a); // (3,(3),3):(1,(1),1)
  }
  // Concat Layout
  // Group Layout
   

  /* layout manipulation ************ */ 
  
  {
    /* layout coalesce*/
    // LA(layout, Lay)
    LA(layout, (Layout<Shape <_2,Shape <_1,_6>>, Stride<_1,Stride<_6,_2>>>{}));
    LP(print_layout(layout));
    L(layout(1, 5));
    LA(result, coalesce(layout));    // _12:_1
    LA(la2, (Layout<Shape<_3,_4>, Stride<_1,_3>>{}));
    LP(print_layout(la2));
    L(rank(la2));
    LP(print_layout(Layout<Shape<Shape<_2,_2>, _3>, Stride<Stride<_24,_2>, _8>>{}));
    // LP(print_layout(coalesce(la2)));
  }

  

}
