import os
if not os.path.exists("cutlass3"):
    os.system("mkdir cutlass3")
else:
    print('running for 2nd time here...')
    with open("a.sh", "w") as f:
        f.write(r"""
pwd
git clone https://github.com/benenzhu/cutlass.git --depth=1 cutlass2
cp submission.py profile_data/
cp cutlass2/examples/cute/tutorial/blackwell/01_mma_sm100.cu profile_data/ && cp cutlass2/examples/cute/tutorial/blackwell/example_utils.hpp profile_data/&& cd profile_data && nvcc 01_mma_sm100.cu 2>&1 >> build.log && ./a.out 2>&1 >> run.log && rm -rf a.out *.cu *.hpp
""")
    os.system("chmod +x a.sh")
    os.system("mkdir -p profile_data")
    os.system("bash a.sh > profile_data/a.log 2>&1")