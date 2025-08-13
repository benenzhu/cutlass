# %%
import os
for i in sorted(os.environ):
    print(i, os.environ[i])
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import os

@cute.kernel
def native_elementwise_add_kernel(
    gA: cute.Tensor, 
    gB: cute.Tensor,
    gC: cute.Tensor
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    
    thread_idx = bidx * bdim + tidx
    
    _, n = gA.shape
    ni = thread_idx % n 
    mi = thread_idx // n # 所以刚才是内存的读取没有做 合并访存, 如果这里的 ni & mi swap 的话.
    
    a_val = gA[mi, ni]  # or this kernel is calculated according to the row index
    b_val = gB[mi, ni]
    

    gC[mi, ni] = a_val + b_val

@cute.jit
def native_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor, # why use m prefix in this tenosr at host? 
    mC: cute.Tensor
): 
    num_thread_per_block = 256
    m, n = mA.shape 
    kernel = native_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(grid=((m*n) // num_thread_per_block, 1, 1), 
                  block=(num_thread_per_block, 1, 1))

M, N = 4096 * 4, 4096 * 4

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=32)
b_ = from_dlpack(b, assumed_align=32)
c_ = from_dlpack(c, assumed_align=32)



native_elementwise_add_ = cute.compile(native_elementwise_add, a_, b_, c_)
import time
tic = time.time()
native_elementwise_add_(a_, b_, c_)
print(time.time() - tic)


should_skip = False
if os.environ.get("NV_CUDA_START_SUSPENDED") or os.environ.get("NV_NSIGHT_INJECTION_PORT_BASE"):
    print("Skipping benchmark due to profiling environment detected")
    should_skip = True

from functools import partial
def benchmark(func, elems,  num_warmups = 5, num_iterations = 100): 
    if should_skip:
        print("Skip benchmark here...")
        return
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    
    for _ in range(num_warmups):
        func()
    start_event.record(stream=torch.cuda.current_stream())
    
    for _ in range(num_iterations):
        func()
    end_event.record(stream=torch.cuda.current_stream())
    
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iterations
    
    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Throughput: { elems / (avg_time / 1000) / 1e9:.2f} GB/s")
    
benchmark(partial(native_elementwise_add_, a_, b_, c_), elems = a.numel() * 2 * 3)



#%%
@cute.kernel
def vectorized_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor, 
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    
    thread_idx = bidx * bdim + tidx
    
    _, n = gA.shape[1] ## 为什么有个 1 呢?
    ni = thread_idx % n 
    mi = thread_idx // n
    
    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    print(f"[DSL], sliced gA = {gA[(None, (mi, ni))]}") 
    print(f"[DSL], sliced gB = {gB[(None, (mi, ni))]}") 
    print(f"[DSL], sliced gC = {gC[(None, (mi, ni))]}") 
    gC[(None, (mi, ni))] = a_val + b_val
    
@cute.jit
def vectorized_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor, 
    mC: cute.Tensor
):
    threads_per_block = 256
    gA = cute.zipped_divide(mA, (1,4))
    gB = cute.zipped_divide(mB, (1,4))
    gC = cute.zipped_divide(mC, (1,4))
    
    print(f"[DSL INFO] Tiled Tensors")
    print(f"[DSL INFO] gA = {gA}")
    print(f"[DSL INFO] gB = {gB}")
    print(f"[DSL INFO] gC = {gC}")
    

    kernel = vectorized_elementwise_add_kernel(gA, gB, gC)
    kernel.launch(grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1), 
                  block=(threads_per_block, 1, 1))

compiled_func = cute.compile(vectorized_elementwise_add, a_,b_,c_)

compiled_func(a_, b_, c_)

if not should_skip:
    torch.testing.assert_close(c, a + b)


benchmark(partial(compiled_func, a_, b_, c_), elems = a.numel() * 2 * 3)

# %%
