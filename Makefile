# Compiler

.SILENT:  

CC = gcc

# Compiler flags
CFLAGS = -Ofast -g -Wall -Wextra -Wpedantic -Iinc -I/usr/include/hdf5/serial -march=native

# Base Libraries
BASE_LIBS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lopenblas -lm

# Directories
SRC_DIR = src
INCLUDE_DIR = inc
BUILD_DIR = build

# Common source files
COMMON_SRC = $(SRC_DIR)/main.c $(SRC_DIR)/utilities.c

# Variants
VARIANTS = OpenCilk OpenMP Pthreads BruteForce

# Output executable
OUT = $(BUILD_DIR)/$(VARIANT)

# Default target
all: $(VARIANTS)

# Build specific variant
$(VARIANTS):
	mkdir -p $(BUILD_DIR)
	$(if $(findstring OpenCilk,$@),clang,gcc) $(CFLAGS) $(COMMON_SRC) $(SRC_DIR)/knn_$@.c -o $(BUILD_DIR)/$@ $(BASE_LIBS) $(if $(findstring OpenCilk,$@),-fopencilk, $(if $(findstring OpenMP,$@),-fopenmp, $(if $(findstring Pthreads,$@),-lpthread)))

# Clean up build
clean:
	rm -rf $(BUILD_DIR)

# Run the program with a specific variant and optional thread count (eg make run TYPE=OpenMP THREADS=18)
run:
	@$(if $(TYPE),,echo "Usage: make run TYPE=type THREADS=num. Options for type: $(VARIANTS)"; exit 1)
	@$(if $(THREADS), \
		echo "Running $(TYPE) with THREADS=$(THREADS)"; \
		$(if $(findstring Cilk,$(TYPE)),CILK_NWORKERS=$(THREADS) ./$(BUILD_DIR)/$(TYPE), \
		$(if $(findstring OpenMP,$(TYPE)),OMP_NUM_THREADS=$(THREADS) ./$(BUILD_DIR)/$(TYPE))), \
		echo "Running $(TYPE) with default thread settings"; \
		./$(BUILD_DIR)/$(TYPE))

.PHONY: all clean run $(VARIANTS)