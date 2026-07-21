# SENG 440 - Embedded Jacobi Matrix Diagonalization
#
# Split build: three source files link into one binary (./svd).
#   svd_angles_final.c  [Amir]   - trig / rotation-angle half
#   svd_rotate.c  		[Param]  - rotation MAC-kernel half
#   svd_main.c    		[shared] - driver + golden-reference tests
#
#   make          build ./svd
#   make run      build and run the tests
#   make bench    build + run the apply_rotations microbenchmark
#   make asm      disassemble svd_rotate.c and svd_angles_final.c for instruction counting
#   make ref      build the frozen single-file reference (svdreference.c)
#   make clean    remove binaries

SHELL   = /bin/bash
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -I.
LDLIBS  = -lm

# Target-specific flags for the ARM VM (Cortex-A7). Uncomment on the VM:
CFLAGS += -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard

# Fixed-point build: driver + Q11 Taylor-series trig + Param's kernel.
# (Taylor sin/cos replaced the piecewise-linear ones, which contracted the unit
#  circle ~0.38%/rotation and lost ~25% of the singular values. Now validates PASS.)
SPLIT_SRC = svd_main.c svd_angles/svd_angles_final.c svd_rotate.c
SPLIT_HDR = svd_common.h

all: svd

svd: $(SPLIT_SRC) $(SPLIT_HDR)
	$(CC) $(CFLAGS) -o $@ $(SPLIT_SRC) $(LDLIBS)

run: svd
	./svd

# Regression check: the fixed-point SVD self-validates its singular values
# against the double golden values (printed by ./svd). Run after every edit.
check: svd
	@./svd | grep -E "Max singular|Reconstruction|VALIDATION"

# Microbenchmark for apply_rotations() (Param's kernel). Builds AND runs.
bench: bench.c svd_rotate.c $(SPLIT_HDR)
	$(CC) $(CFLAGS) -o bench bench.c svd_rotate.c $(LDLIBS)
	./bench

# NEON int16x8 rotation prototype (standalone). Builds AND runs.
# Needs -mfpu=neon (the Cortex-A7 CFLAGS line above, uncommented on the VM).
neon: bench_neon.c
	$(CC) $(CFLAGS) -o bench_neon bench_neon.c $(LDLIBS)
	./bench_neon

# Static instruction count for the rotation and trig kernels: disassemble the objects.
asm: svd_rotate.c svd_angles/svd_angles_final.c $(SPLIT_HDR)
	$(CC) $(CFLAGS) -c svd_rotate.c -o svd_rotate.o
	objdump -d svd_rotate.o > svd_rotate.s
	$(CC) $(CFLAGS) -c svd_angles/svd_angles_final.c -o svd_angles_final.o
	objdump -d svd_angles_final.o > svd_angles_final.s
	@echo "Disassembly written to svd_rotate.s and svd_angles_final.s -- count the respective blocks."

# Frozen single-file golden reference, for diffing against the split build.
ref: svdreference.c
	$(CC) $(CFLAGS) -o svd_reference svdreference.c $(LDLIBS)

clean:
	rm -f svd svd_reference bench bench_neon svd_rotate.o svd_rotate.s svd_angles_final.o svd_angles_final.s

.PHONY: all run check bench asm neon ref clean
