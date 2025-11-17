import os
os.system("pwd")
os.system("mkdir profile_data")
os.system("ls -lh")
os.system("git clone https://github.com/benenzhu/cutlass.git --depth=1 cutlass2")
os.system("cd cutlass2/examples/cute/tutorial/blackwell && nvcc 01_mma_sm100.cu -o 01_mma_sm100")
