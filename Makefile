# SENG 440 - Embedded Jacobi Matrix Diagonalization
#
# Split build: three source files link into one binary (./svd).
#   svd_angles.c  [Amir]   - trig / rotation-angle half
#   svd_rotate.c  [Param]  - rotation MAC-kernel half
#   svd_main.c    [shared] - driver + golden-reference tests
#
#   make          build ./svd
#   make run      build and run the tests
#   make ref      build the frozen single-file reference (svdreference.c)
#   make clean    remove binaries

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra
LDLIBS  = -lm

# Target-specific flags for the ARM VM (Cortex-A7). Uncomment on the VM:
# CFLAGS += -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard

SPLIT_SRC = svd_main.c svd_angles.c svd_rotate.c
SPLIT_HDR = svd_common.h

all: svd

svd: $(SPLIT_SRC) $(SPLIT_HDR)
	$(CC) $(CFLAGS) -o $@ $(SPLIT_SRC) $(LDLIBS)

run: svd
	./svd

# Frozen single-file golden reference, for diffing against the split build.
ref: svdreference.c
	$(CC) $(CFLAGS) -o svd_reference svdreference.c $(LDLIBS)

clean:
	rm -f svd svd_reference

.PHONY: all run ref clean
