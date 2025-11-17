submit: 
	@https_proxy=http://127.0.0.1:8890 popcorn-cli submit --gpu NVIDIA --mode test --leaderboard nvfp4_gemv print.py


build_local: 
	@nvcc --generate-code=arch=compute_100a,code=[sm_100a] --fdevice-time-trace=- --generate-code=arch=compute_100a,code=[compute_100a] -O3 -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -I/A/cutlass/include -I/A/cutlass/examples/common -I/A/cutlass/build/include -I/A/cutlass/tools/util/include 01_mma_sm100.cu
	

download:
	@if [ -z "$(R)" ]; then \
		echo "Error: RUN_ID is required. Usage: make download RUN_ID=<run_id>"; \
		exit 1; \
	fi
	@gh run download -R gpu-mode/discord-cluster-manager $(R) -D $(R)