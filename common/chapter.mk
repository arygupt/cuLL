# Shared build rules for each chapter folder. In a chapter folder:
#   book make            build every .cu here
#   book make foo        build foo.cu → ./foo
#   book make run-foo    build and run foo
#   book make clean
# Override flags: book make foo NVCC_FLAGS="-O0 -G"   (debug build)
COMMON     := $(dir $(lastword $(MAKEFILE_LIST)))
ARCH       ?= sm_75
NVCC_FLAGS ?= -O3 -lineinfo
NVCC       := nvcc -std=c++17 -arch=$(ARCH) -I$(COMMON) $(NVCC_FLAGS)
LIBS       ?= -lcublas

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu $(COMMON)util.cuh
	$(NVCC) $< -o $@ $(LIBS)

run-%: %
	./$<

clean:
	rm -f $(BINS) *.ncu-rep *.o

.PHONY: all clean
