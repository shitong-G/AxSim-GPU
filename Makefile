# AxSim-GPU Makefile
# Usage: make [DEBUG=1] [CUDA_ARCH=sm_70]

CXX       ?= g++
NVCC      ?= nvcc
# CUDA runtime: libcudart/header paths under CUDA toolkit.
# Prefer nvcc's toolkit root when nvcc is on PATH.
NVCC_CUDA_HOME := $(shell nvcc --show-cuda-home 2>/dev/null)
CUDA_HOME      ?= $(if $(strip $(NVCC_CUDA_HOME)),$(NVCC_CUDA_HOME),/usr/local/cuda)
CUDA_LIB       ?= $(CUDA_HOME)/lib64
# ABC headers use paths relative to abc/src (e.g. misc/vec/vec.h).
# ABC_USE_STDINT_H: platform types via stdint (otherwise abc_global.h needs LIN/LIN64/WIN32).
INC       := -I include -I abc/src
CXXFLAGS  := -std=c++17 $(INC) -O2 -Wall -DABC_USE_STDINT_H
# Older nvcc (CUDA 10.x and below) rejects -std=c++17; .cu code is C++14-compatible.
# Override on CUDA 11+:  make NVCC_STD=c++17
NVCC_STD  ?= c++14
NVCCFLAGS := -std=$(NVCC_STD) $(INC) -I$(CUDA_HOME)/include -O2

# CUDA architecture (override: make CUDA_ARCH=sm_80)
CUDA_ARCH ?= sm_70

# ABC: Io_*, Abc_* symbols come from a static lib built in-tree (see abc/Makefile: libabc.a).
ABC_DIR        := abc
ABC_LIB        := $(ABC_DIR)/libabc.a
# Match src/interface.cpp (-DABC_USE_STDINT_H); skip readline so we do not need -lreadline.
ABC_MAKE_ARGS  := ABC_USE_STDINT_H=1 ABC_USE_NO_READLINE=1

# Final link uses g++ (not nvcc): nvlink cannot link pure host .o from g++.
# After pulling on another OS or toolchain, run: make clean && make
# Order: project .o, then libabc.a, then ABC/CUDA system libs.
LDFLAGS := $(ABC_LIB) -lm -pthread -ldl -lrt -L$(CUDA_LIB) -lcudart

# Debug build
ifdef DEBUG
  CXXFLAGS  := -std=c++17 $(INC) -O0 -g -Wall -DABC_USE_STDINT_H
  NVCCFLAGS := -std=$(NVCC_STD) $(INC) -O0 -g
endif

NVCCFLAGS += -arch=$(CUDA_ARCH)

# Output and objects (all .o in build/)
BUILD     := build
EXE       := $(BUILD)/axsim_main
SRC_CPP   := src/main_axsim.cpp src/circuit_soa.cpp src/interface.cpp src/pattern_file.cpp src/cpu_sim_metrics.cpp
SRC_CU    := cuda/sim_kernels.cu
OBJ_CPP   := $(BUILD)/main_axsim.o $(BUILD)/circuit_soa.o $(BUILD)/interface.o $(BUILD)/pattern_file.o $(BUILD)/cpu_sim_metrics.o
OBJ_CU    := $(BUILD)/sim_kernels.o
OBJS      := $(OBJ_CPP) $(OBJ_CU)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/main_axsim.o: src/main_axsim.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/circuit_soa.o: src/circuit_soa.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/interface.o: src/interface.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/pattern_file.o: src/pattern_file.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/cpu_sim_metrics.o: src/cpu_sim_metrics.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/sim_kernels.o: cuda/sim_kernels.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(ABC_LIB):
	$(MAKE) -C $(ABC_DIR) $(ABC_MAKE_ARGS) libabc.a

$(EXE): $(OBJS) $(ABC_LIB)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

all: $(EXE)

run: $(EXE)
	$(EXE)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
.DEFAULT_GOAL := all
