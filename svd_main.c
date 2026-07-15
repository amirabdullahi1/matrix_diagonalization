/*
 * svd_main.c  --  Two-sided Jacobi SVD, cyclic-by-rows.
 *
 * This file owns the algorithm scaffolding (matrix helpers, the sweep loop,
 * post-processing, verification, and main). The two optimization halves live
 * in separate files and meet at the seam inside jacobi_rotate():
 *
 *   svd_angles.c  [Amir]  -> compute_rotation_factors()
 *   svd_rotate.c  [Param] -> apply_rotations()
 *
 * Build: make        (links all three .c files into ./svd)
 *        make run    (build + run the golden-reference tests)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svd_common.h"

static void mat_identity(mat_t A)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i == j) ? 1.0 : 0.0;
}

static void mat_copy(mat_t dst, const mat_t src)
{
    memcpy(dst, src, sizeof(mat_t));
}

static void mat_print(const char *label, const mat_t A)
{
    printf("%s =\n", label);
    for (int i = 0; i < N; i++) {
        printf("  ");
        for (int j = 0; j < N; j++)
            printf("%10.4f ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

// Frobenius norm of the off-diagonal elements; drives the convergence test.
static double off_diagonal_norm(const mat_t A)
{
    double s = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j)
                s += A[i][j] * A[i][j];
    return sqrt(s);
}

/*
 * The seam: read the 2x2 block, skip if already negligible, then hand off to
 * the two optimization halves. This glue stays simple on purpose.
 */
static void jacobi_rotate(mat_t M, mat_t U, mat_t V, int p, int q)
{
    const double b = M[p][q];
    const double c = M[q][p];

    // If b and c are already negligible, skip this pair.
    if (fabs(b) + fabs(c) < 1e-15)
        return;

    double cl, sl, cr, sr;
    compute_rotation_factors(M[p][p], b, c, M[q][q], &cl, &sl, &cr, &sr); // Amir
    apply_rotations(M, U, V, p, q, cl, sl, cr, sr);                       // Param
}

// On input:  M holds the matrix to decompose.
// On output: M is diagonal (= Sigma), U and V are orthogonal,
//            and U * Sigma * V^T = M_original.  Returns sweeps performed.
static int svd_jacobi(mat_t M, mat_t U, mat_t V, int verbose)
{
    mat_identity(U);
    mat_identity(V);

    int sweep;
    for (sweep = 1; sweep <= MAX_SWEEPS; sweep++) {
        // Cyclic-by-rows pair ordering over all i<j:
        // (0,1),(0,2),...,(0,5),(1,2),...,(4,5)  -> N*(N-1)/2 = 15 pairs
        for (int p = 0; p < N - 1; p++)
            for (int q = p + 1; q < N; q++)
                jacobi_rotate(M, U, V, p, q);

        const double off = off_diagonal_norm(M);
        if (verbose) {
            printf("--- End of sweep %d  (off-diagonal norm = %.6e) ---\n",
                   sweep, off);
            mat_print("M", M);
        }
        if (off < CONVERGENCE_EPS)
            break;
    }
    return sweep;
}

// Force non-negative singular values: flip sign of column k of U if m_kk < 0.
static void normalize_singular_values(mat_t M, mat_t U)
{
    for (int k = 0; k < N; k++) {
        if (M[k][k] < 0.0) {
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
            double tmp = M[i][i];
            M[i][i] = M[max_idx][max_idx];
            M[max_idx][max_idx] = tmp;
            for (int k = 0; k < N; k++) {
                tmp = U[k][i]; U[k][i] = U[k][max_idx]; U[k][max_idx] = tmp;
                tmp = V[k][i]; V[k][i] = V[k][max_idx]; V[k][max_idx] = tmp;
            }
        }
    }
}

// Reconstruct U * Sigma * V^T and compare with the original (Frobenius norm).
static double reconstruction_error(const mat_t M_orig,
                                   const mat_t U, const mat_t Sigma,
                                   const mat_t V)
{
    mat_t US, USV;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            US[i][j] = U[i][j] * Sigma[j][j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += US[i][k] * V[j][k];   // V^T[k][j] = V[j][k]
            USV[i][j] = s;
        }
    double err = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double d = M_orig[i][j] - USV[i][j];
            err += d * d;
        }
    return sqrt(err);
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
                        "error: M[%d][%d] = %.0f outside 12-bit range [%d, %d]\n",
                        i, j, M_orig[i][j], INPUT_MIN, INPUT_MAX);
                return 1;
            }

    mat_t M, U, V;
    mat_copy(M, M_orig);

    mat_print("Input matrix M", M_orig);

    const int sweeps = svd_jacobi(M, U, V, 1);
    printf("Converged in %d sweeps.\n\n", sweeps);

    normalize_singular_values(M, U);
    sort_singular_values(M, U, V);

    mat_print("Sigma (singular values, sorted descending)", M);
    mat_print("U", U);
    mat_print("V", V);

    const double err = reconstruction_error(M_orig, U, M, V);
    printf("Reconstruction error ||M - U*Sigma*V^T||_F = %.3e\n", err);

    return 0;
}
