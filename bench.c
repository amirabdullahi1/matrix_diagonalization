/*
 * bench.c  --  microbenchmark for apply_rotations()  [Param's kernel, FIXED-POINT].
 *
 * Times many back-to-back rotations on fixed int16 data with Q11 factors.
 * Each rotation is (near) orthogonal, so values stay bounded over millions of
 * calls. Reports ns/call; compare before/after each optimization.
 *
 * Build:  make bench      Run:  ./bench
 * (Run on the Cortex-A7 VM for meaningful ARM numbers; QEMU is not
 * cycle-accurate, so treat timings as relative and back them with objdump.)
 */
#include <stdio.h>
#include <time.h>
#include "svd_common.h"

#define ITERS 5000000L

int main(void)
{
    mat_t M, U, V;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            M[i][j] = (i == j) ? 1000 : 10;
            U[i][j] = (i == j) ? (1 << TRIG_SHIFT) : 0;   // Q11 identity
            V[i][j] = (i == j) ? (1 << TRIG_SHIFT) : 0;
        }

    // Q11 rotation factors for small angles (orthogonal -> bounded).
    const int16_t cl = 2047, sl = 20;    // ~cos(0.01), sin(0.01) in Q11
    const int16_t cr = 2047, sr = 27;    // ~cos(0.013), sin(0.013)

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long it = 0; it < ITERS; it++)
        apply_rotations(M, U, V, 0, 1, cl, sl, cr, sr);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

    long sum = 0;   // checksum so the loop is not optimized away
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += M[i][j] + U[i][j] + V[i][j];

    printf("apply_rotations: %ld calls in %.1f ms = %.2f ns/call\n",
           ITERS, ns / 1e6, ns / ITERS);
    printf("checksum (ignore the value): %ld\n", sum);
    return 0;
}
