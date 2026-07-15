/*
 * svd_reference.c
 *
 * Algorithm: two-sided Jacobi rotation, cyclic-by-rows pair ordering.
 * For each pair (i,j):
 *   - Extract the 2x2 submatrix [m_ii m_ij; m_ji m_jj]
 *   - Compute left/right rotation angles via the two-angle method:
 *       theta_sum  = atan2(m_ji + m_ij, m_jj - m_ii)
 *       theta_diff = atan2(m_ji - m_ij, m_jj + m_ii)
 *       theta_l = (theta_sum - theta_diff) / 2
 *       theta_r = (theta_sum + theta_diff) / 2
 *   - Apply R_l * M * R_r^T to M (zeros m_ij and m_ji in the 2x2 block)
 *   - Accumulate rotations into U and V so that U * Sigma * V^T = M_original
 *
 * Build: gcc -O2 -Wall -Wextra -o svd_reference svd_reference.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define N 6                       // matrix dimension (6x6 per spec)
#define MAX_SWEEPS 30             // hard cap on sweeps
#define CONVERGENCE_EPS 1e-9      // off-diagonal Frobenius norm threshold

// Per the program spec the input matrix entries are 12-bit signed values,
// i.e. each m_ij lies in the range [-2048, 2047]. The reference math itself
// stays in double precision to serve as the golden model.
#define INPUT_BITS 12
#define INPUT_MIN  (-(1 << (INPUT_BITS - 1)))   // -2048
#define INPUT_MAX  ((1 << (INPUT_BITS - 1)) - 1)//  2047

typedef double mat_t[N][N];


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

// Frobenius norm of off-diagonal elements: measures how far from diagonal we are. Drives the convergence test.
static double off_diagonal_norm(const mat_t A)
{
    double s = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j)
                s += A[i][j] * A[i][j];
    return sqrt(s);
}

// Core: two-sided Jacobi rotation for a single (p,q) pair           

/*
 * Apply rotation pair (p,q) to M, U, V in place.
 *
 * Math:
 *   Let a = M[p][p], b = M[p][q], c = M[q][p], d = M[q][q].
 *   theta_sum  = atan2(c + b, d - a)
 *   theta_diff = atan2(c - b, d + a)
 *   theta_l = (theta_sum - theta_diff) / 2
 *   theta_r = (theta_sum + theta_diff) / 2
 *
 * Then M <- R_l * M * R_r^T, where R_l and R_r are 2x2 rotations
 * acting on rows {p,q} and columns {p,q} respectively.
 *
 * Accumulate into U and V so the invariant U * M_current * V^T = M_original
 * is preserved. Specifically:
 *   U <- U * R_l^T
 *   V <- V * R_r^T
 */
static void jacobi_rotate(mat_t M, mat_t U, mat_t V, int p, int q)
{
    const double a = M[p][p];
    const double b = M[p][q];
    const double c = M[q][p];
    const double d = M[q][q];

    // If b and c are already negligible, skip this pair. 
    if (fabs(b) + fabs(c) < 1e-15)
        return;

    /* ==== OPT[trig]: rotation-angle computation — main acceleration target ====
     * In the fixed-point build the two atan2() calls become the piecewise-linear
     * integer arctan (this is the custom instruction). The two calls are
     * data-independent -> schedule their micro-ops across both firmware
     * issue slots to run them concurrently. */
    const double theta_sum  = atan2(c + b, d - a);
    const double theta_diff = atan2(c - b, d + a);
    const double theta_l = 0.5 * (theta_sum - theta_diff);
    const double theta_r = 0.5 * (theta_sum + theta_diff);

    /* ==== OPT[trig]: 4 independent sin/cos -> piecewise-linear sin/cos.
     * sin and cos must share the same accuracy or the rotation stops being
     * orthogonal. Pack two evaluations per microcycle in the 2-issue firmware. */
    const double cl = cos(theta_l), sl = sin(theta_l);
    const double cr = cos(theta_r), sr = sin(theta_r);

    /* ==== OPT[rot]: the four rotation loops below (M-rows, M-cols, U, V) are the
     * MAC kernels. Each iteration does two independent updates of the form
     * (c*x - s*y) and (s*x + c*y) on the same loaded pair -> one per firmware
     * issue slot; iterations are independent across k -> unroll (N=6 is fixed)
     * and software-pipeline the loads, or vectorize as 4-lane NEON MACs. */

    /* Apply left rotation R_l to rows p and q of M (M <- R_l * M).
     * R_l = [[cl, -sl], [sl, cl]], so on rows {p,q}:
     *   new row_p =  cl * row_p - sl * row_q
     *   new row_q =  sl * row_p + cl * row_q                      */
    for (int k = 0; k < N; k++) {
        const double mp = M[p][k];
        const double mq = M[q][k];
        M[p][k] =  cl * mp - sl * mq;
        M[q][k] =  sl * mp + cl * mq;
    }

    /* Apply right rotation R_r^T to columns p and q of M (M <- M * R_r^T).
     * R_r^T = [[cr, sr], [-sr, cr]], so on cols {p,q}:
     *   new col_p =  cr * col_p - sr * col_q
     *   new col_q =  sr * col_p + cr * col_q                      */
    for (int k = 0; k < N; k++) {
        const double mp = M[k][p];
        const double mq = M[k][q];
        M[k][p] =  cr * mp - sr * mq;
        M[k][q] =  sr * mp + cr * mq;
    }

    /* Maintain the invariant  U * M_current * V^T = M_original.
     *
     * After M_current <- R_l * M_current * R_r^T, the invariant
     * forces  U <- U * R_l^T  and  V <- V * R_r^T.
     *
     * Derivation: write M_old = R_l^T * M_new * R_r. Then
     *   U_old * M_old * V_old^T
     *     = U_old * R_l^T * M_new * R_r * V_old^T
     *     = (U_old * R_l^T) * M_new * (V_old * R_r^T)^T.
     *
     * Right-multiplying U by R_l^T = [[cl, sl], [-sl, cl]] updates
     * columns {p,q} of U:
     *   new U[:,p] =  cl * U[:,p] - sl * U[:,q]
     *   new U[:,q] =  sl * U[:,p] + cl * U[:,q]                   */
    for (int k = 0; k < N; k++) {
        const double up = U[k][p];
        const double uq = U[k][q];
        U[k][p] =  cl * up - sl * uq;
        U[k][q] =  sl * up + cl * uq;
    }

    // V <- V * R_r^T  (same column update pattern with cr, sr).  
    for (int k = 0; k < N; k++) {
        const double vp = V[k][p];
        const double vq = V[k][q];
        V[k][p] =  cr * vp - sr * vq;
        V[k][q] =  sr * vp + cr * vq;
    }
}

// SVD driver: cyclic-by-rows sweeps                                 

// On input:  M holds the matrix to decompose.
 // On output: M is diagonal (= Sigma), U and V are orthogonal,
 //            and U * Sigma * V^T = M_original.
 // Returns: number of sweeps performed.
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

// Post-processing: force positive singular values, sort descending  

// Convention: singular values are non-negative and sorted in
// descending order. A negative diagonal entry m_kk is fixed by
// flipping the sign of column k of U (or V); we choose U here.

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

// Sort singular values in descending order, applying the same column
// permutation to U and V to preserve U * Sigma * V^T.

static void sort_singular_values(mat_t M, mat_t U, mat_t V)
{
    for (int i = 0; i < N - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < N; j++)
            if (M[j][j] > M[max_idx][max_idx])
                max_idx = j;
        if (max_idx != i) {
            // Swap diagonal entries 
            double tmp = M[i][i];
            M[i][i] = M[max_idx][max_idx];
            M[max_idx][max_idx] = tmp;
            // Swap columns i and max_idx of U and V 
            for (int k = 0; k < N; k++) {
                tmp = U[k][i]; U[k][i] = U[k][max_idx]; U[k][max_idx] = tmp;
                tmp = V[k][i]; V[k][i] = V[k][max_idx]; V[k][max_idx] = tmp;
            }
        }
    }
}


// Verification: reconstruct U * Sigma * V^T and compare with original


static double reconstruction_error(const mat_t M_orig,
                                   const mat_t U, const mat_t Sigma,
                                   const mat_t V)
{
    mat_t US, USV;
    //US = U * Sigma  (Sigma is diagonal, so this is column scaling) 
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            US[i][j] = U[i][j] * Sigma[j][j];
    //USV = US * V^T 
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