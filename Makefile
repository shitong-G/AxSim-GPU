# AxSim-GPU Makefile
# Usage: make [DEBUG=1] [CUDA_ARCH=sm_70]

CXX       ?= g++
NVCC      ?= nvcc
INC       := -I include
CXXFLAGS  := -std=c++17 $(INC) -O2 -Wall
# Older nvcc (CUDA 10.x and below) rejects -std=c++17; .cu code is C++14-compatible.
# Override on CUDA 11+:  make NVCC_STD=c++17
NVCC_STD  ?= c++14
NVCCFLAGS := -std=$(NVCC_STD) $(INC) -O2
LDFLAGS   :=

# CUDA architecture (override: make CUDA_ARCH=sm_80)
CUDA_ARCH ?= sm_70

# Debug build
ifdef DEBUG
  CXXFLAGS  := -std=c++17 $(INC) -O0 -g -Wall
  NVCCFLAGS := -std=$(NVCC_STD) $(INC) -O0 -g
endif

NVCCFLAGS += -arch=$(CUDA_ARCH)

# Output and objects (all .o in build/)
BUILD     := build
EXE       := $(BUILD)/axsim_main
SRC_CPP   := src/main_axsim.cpp src/circuit_soa.cpp src/interface.cpp
SRC_CU    := cuda/sim_kernels.cu
OBJ_CPP   := $(BUILD)/main_axsim.o $(BUILD)/circuit_soa.o $(BUILD)/interface.o
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

$(BUILD)/sim_kernels.o: cuda/sim_kernels.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXE): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

all: $(EXE)

run: $(EXE)
	$(EXE)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
.DEFAULT_GOAL := all
