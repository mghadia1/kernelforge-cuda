# KernelForge build. One shared library holds every version so the benchmark can
# switch between them by symbol name through ctypes.
#
#   make            build the CUDA library and the host-side selection simulation
#   make test       build (if nvcc exists) and run pytest
#   make bench      run the sweep into bench/results.csv
#   make ARCH=sm_80 build for a different GPU (default targets the T4)
#
# ARCH defaults to sm_75, the Turing arch of the Colab/Kaggle T4 this project is
# benchmarked on. sm_80 = A100, sm_89 = L4/L40S.

NVCC    ?= nvcc
ARCH    ?= sm_75
PYTHON  ?= python3
BUILD   := build
LIB     := $(BUILD)/libkernelforge.so
SIM     := $(BUILD)/libkfsim.so
SRC     := src/v0_naive.cu src/v1_shared.cu src/v2_warp.cu src/v3_topk.cu \
           src/v4_batch.cu src/v5_regblock.cu src/cublas_ref.cu
NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH) -Xcompiler -fPIC -lineinfo -Isrc
LDLIBS    := -lcublas

CXX     ?= c++

.PHONY: all sim test bench clean profile

all: $(LIB) $(SIM)

$(LIB): $(SRC) src/kernelforge.h src/common.cuh src/topk_rule.h \
        src/v3_config.h src/v4_config.h src/v5_config.h \
        src/merge_topk.cuh src/select_warp.cuh
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) --shared $(SRC) -o $@ $(LDLIBS)

# -lineinfo above keeps source line numbers in the cubin so Nsight Compute can
# attribute stalls to lines; it does not affect optimization.

# v3's selection logic, compiled for the host so it can be tested with no GPU.
# No CUDA involved: see the header comment in src/selection_sim.cpp for what
# this does and does not prove.
sim: $(SIM)

$(SIM): src/selection_sim.cpp src/topk_rule.h src/v3_config.h src/v4_config.h
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 -shared -fPIC -Isrc src/selection_sim.cpp -o $@

test: $(SIM)
	@command -v $(NVCC) >/dev/null 2>&1 && $(MAKE) $(LIB) || \
		echo "nvcc not found - running CPU-only tests; the GPU tests will skip"
	$(PYTHON) -m pytest

bench: $(LIB)
	$(PYTHON) bench/run.py --out bench/results.csv

# Nsight Compute on the best kernel. Needs a GPU with profiling permitted
# (Colab allows the basic sections; add --set full off-Colab).
profile: $(LIB)
	ncu --set basic --target-processes all \
		$(PYTHON) bench/run.py --profile-once --n 100000 --b 32

clean:
	rm -rf $(BUILD) bench/__pycache__ .pytest_cache
