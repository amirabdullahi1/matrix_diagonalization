/*
 * svd_main.c  --  Two-sided Jacobi SVD, cyclic-by-rows (FIXED-POINT driver).
 *
 * Algorithm scaffolding (sweep loop, post-processing, verification, main).
 * The two optimization halves meet at the seam in jacobi_rotate():
 *
 *   svd_angles/svd_slopes_approx_int.c  [Amir]  -> compute_rotation_factors()
 *   svd_rotate.c                        [Param] -> apply_rotations()
 *
 * Fixed-point layout:  M = raw int (Q0),  U,V = Q11,  factors = Q11.
 * Because fixed-point is approximate, the output is NOT bit-identical to the
 * double golden model; instead we validate the singular values against the
 * double reference within a tolerance and report a float reconstruction error.
 *
 * Build: make        (links svd_main.c + the int trig + svd_rotate.c into ./svd)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "svd_common.h"

#define Q11_ONE (1 << TRIG_SHIFT)          // 1.0 in Q11 == 2048

// Singular values of the reference 6x6 from the double golden model.
static const double GOLDEN_SV[N] = {
    3985.84, 3571.62, 2438.81, 1399.95, 884.87, 166.82
};
#define SV_TOL 20.0                         // acceptable |fixed-point - golden|

// U, V start as the identity in Q11 (diagonal = 1.0).
static void identity_q11(mat_t A)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i == j) ? Q11_ONE : 0;
}

static void mat_copy(mat_t dst, const mat_t src)
{
    memcpy(dst, src, sizeof(mat_t));
}

static void mat_print_int(const char *label, const mat_t A)
{
    printf("%s =\n", label);
    for (int i = 0; i < N; i++) {
        printf("  ");
        for (int j = 0; j < N; j++)
            printf("%8d ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

// Largest |off-diagonal| element; drives the convergence test.
static int max_offdiagonal(const mat_t A)
{
    int m = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j) {
                int v = A[i][j] < 0 ? -A[i][j] : A[i][j];
                if (v > m) m = v;
            }
    return m;
}

/*
 * The seam: read the 2x2 block, skip if already diagonal, then hand off to
 * the two optimization halves.
 */
static void jacobi_rotate(mat_t M, mat_t U, mat_t V, int p, int q)
{
    const int16_t b = M[p][q];
    const int16_t c = M[q][p];

    if (b == 0 && c == 0)          // already zero -> identity rotation
        return;

    int16_t cl, sl, cr, sr;
    compute_rotation_factors(M[p][p], b, c, M[q][q], &cl, &sl, &cr, &sr); // Amir
    apply_rotations(M, U, V, p, q, cl, sl, cr, sr);                       // Param
}

// On output M is (approximately) diagonal = Sigma; U, V orthogonal (Q11).
static int svd_jacobi(mat_t M, mat_t U, mat_t V, int verbose)
{
    identity_q11(U);
    identity_q11(V);

    int sweep, prev_off = 32767;
    for (sweep = 1; sweep <= MAX_SWEEPS; sweep++) {
        for (int p = 0; p < N - 1; p++)
            for (int q = p + 1; q < N; q++)
                jacobi_rotate(M, U, V, p, q);

        const int off = max_offdiagonal(M);
        if (verbose)
            printf("sweep %2d: max|off-diagonal| = %d\n", sweep, off);
        // Stop when below the threshold OR when the off-diagonal stops
        // shrinking (it has hit the fixed-point rounding floor -- more sweeps
        // only waste cycles).
        if (off <= OFFDIAG_EPS || off >= prev_off)
            break;
        prev_off = off;
    }
    return sweep;
}

// Force non-negative singular values: flip sign of column k of U if m_kk < 0.
static void normalize_singular_values(mat_t M, mat_t U)
{
    for (int k = 0; k < N; k++) {
        if (M[k][k] < 0) {
            M[k][k] = -M[k][k];
            for (int i = 0; i < N; i++)
                U[i][k] = -U[i][k];
        }
    }
}

// Sort singular values descending; apply the same column permutation to U, V.
static void sort_singular_values(mat_t M, mat_t U, mat_t V)
{
    for (int i = 0; i < N - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < N; j++)
            if (M[j][j] > M[max_idx][max_idx])
                max_idx = j;
        if (max_idx != i) {
            int16_t tmp = M[i][i];
            M[i][i] = M[max_idx][max_idx];
            M[max_idx][max_idx] = tmp;
            for (int k = 0; k < N; k++) {
                tmp = U[k][i]; U[k][i] = U[k][max_idx]; U[k][max_idx] = tmp;
                tmp = V[k][i]; V[k][i] = V[k][max_idx]; V[k][max_idx] = tmp;
            }
        }
    }
}

/*
 * Reconstruct U * Sigma * V^T in floating point and compare with the original.
 * U, V are Q11 (divide by 2048 each), Sigma is raw.
 */
static double reconstruction_error(const mat_t M_orig,
                                   const mat_t U, const mat_t Sigma,
                                   const mat_t V)
{
    const double denom = (double)Q11_ONE * (double)Q11_ONE;
    double err = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += (double)U[i][k] * (double)Sigma[k][k] * (double)V[j][k];
            s /= denom;
            double d = (double)M_orig[i][j] - s;
            err += d * d;
        }
    return sqrt(err);
}

// Print the reconstructed U * Sigma * V^T (rounded to int) for eyeball checking.
static void print_reconstruction(const mat_t U, const mat_t Sigma, const mat_t V)
{
    const double denom = (double)Q11_ONE * (double)Q11_ONE;
    printf("Reconstructed U*Sigma*V^T (rounded) =\n");
    for (int i = 0; i < N; i++) {
        printf("  ");
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += (double)U[i][k] * (double)Sigma[k][k] * (double)V[j][k];
            printf("%8ld ", lround(s / denom));
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    // 6x6 input matrix; every entry is a 12-bit signed value in [-2048, 2047].
    mat_t M_orig = {
        {  1024,  -512,   777,  -311,   126,  2047 },
        {  -420,   140,   790,  -530,  -880,   333 },
        { -2048,  -100,   450,   900,  1500, -1234 },
        {   340,   160,   380,  -190,  -640,   512 },
        {   720,  1980, -1750,   210,   -55,  -900 },
        {  -333,   888, -1234,  2047, -2048,   101 }
    };

    // Enforce the 12-bit input constraint from the spec.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (M_orig[i][j] < INPUT_MIN || M_orig[i][j] > INPUT_MAX) {
                fprintf(stderr,
                        "error: M[%d][%d] = %d outside 12-bit range [%d, %d]\n",
                        i, j, M_orig[i][j], INPUT_MIN, INPUT_MAX);
                return 1;
            }

    mat_t M, U, V;
    mat_copy(M, M_orig);

    mat_print_int("Input matrix M (raw int16)", M_orig);

    const int sweeps = svd_jacobi(M, U, V, 1);
    printf("\nStopped after %d sweep(s).\n\n", sweeps);

    normalize_singular_values(M, U);
    sort_singular_values(M, U, V);

    mat_print_int("Sigma (fixed-point singular values, sorted descending)", M);

    // Validate against the double golden singular values.
    printf("Singular value    fixed-point   double golden    |diff|\n");
    double max_diff = 0.0;
    int pass = 1;
    for (int k = 0; k < N; k++) {
        double diff = fabs((double)M[k][k] - GOLDEN_SV[k]);
        if (diff > max_diff) max_diff = diff;
        if (diff > SV_TOL) pass = 0;
        printf("   sigma[%d]         %8d      %9.2f     %7.2f\n",
               k, M[k][k], GOLDEN_SV[k], diff);
    }

    const double err = reconstruction_error(M_orig, U, M, V);
    printf("\nMax singular-value error vs double golden : %.2f  (tol %.0f)\n",
           max_diff, SV_TOL);
    printf("Reconstruction error ||M - U*Sigma*V^T||_F : %.2f\n", err);
    printf("VALIDATION: %s\n\n", pass ? "PASS" : "CHECK (exceeds tolerance)");

    // Show the orthogonal factors and the reconstruction vs the original.
    mat_print_int("U (Q11 = value x 2048)", U);
    mat_print_int("V (Q11 = value x 2048)", V);
    print_reconstruction(U, M, V);
    mat_print_int("Original M (should match the reconstruction)", M_orig);

    // ---- Whole-SVD timing (single number, free of process-spawn noise) ----
    const int REPS = 100000;
    mat_t Mt, Ut, Vt;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) {
        mat_copy(Mt, M_orig);
        svd_jacobi(Mt, Ut, Vt, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((t1.tv_sec - t0.tv_sec) * 1e9 +
                 (t1.tv_nsec - t0.tv_nsec)) / 1e3 / REPS;
    printf("Whole-SVD timing: %.2f us/decomposition  (%d reps, %d sweeps each)\n",
           us, REPS, sweeps);

    return 0;
}
