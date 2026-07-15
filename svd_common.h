/*
 * svd_common.h - shared configuration and the interface contract between
 * the two halves of the Jacobi inner loop.
 *
 * The inner loop is split at a natural seam. The ONLY data that crosses the
 * seam is the four rotation factors (cl, sl, cr, sr), so the two halves can be
 * developed and optimized independently and still link into one program:
 *
 *   compute_rotation_factors()   [svd_angles.c  - Amir]
 *       (a,b,c,d)  ->  cl, sl, cr, sr           (trig: arctan + sin/cos)
 *
 *   apply_rotations()            [svd_rotate.c  - Param]
 *       cl,sl,cr,sr + matrices   ->  rotated M, U, V   (MAC kernels)
 *
 * The driver (svd_main.c) glues them together and runs the golden-reference
 * tests, so `make run` keeps validating the whole thing after either half
 * changes.
 */
#ifndef SVD_COMMON_H
#define SVD_COMMON_H

#include <math.h>

#define N 6                       // matrix dimension (6x6 per spec)
#define MAX_SWEEPS 30             // hard cap on sweeps
#define CONVERGENCE_EPS 1e-9      // off-diagonal Frobenius norm threshold

// 12-bit signed input range: each matrix entry lies in [-2048, 2047].
#define INPUT_BITS 12
#define INPUT_MIN  (-(1 << (INPUT_BITS - 1)))   // -2048
#define INPUT_MAX  ((1 << (INPUT_BITS - 1)) - 1)//  2047

/*
 * Fixed-point contract (used once you port off double precision; unused while
 * the reference stays in double). When Amir returns cl/sl/cr/sr as
 * Q(TRIG_SHIFT) integers, Param's MACs must rescale each product with
 * `>> TRIG_SHIFT`. Keep this the single source of truth for that shift so both
 * halves stay in agreement.
 *
 *   #define TRIG_SHIFT 11        // 1.0 represented as 2^11 (Q11)
 */

typedef double mat_t[N][N];

// ---- The seam ----------------------------------------------------------

// [svd_angles.c] Rotation-angle + factor generation.
// Input : the 2x2 block  a=M[p][p], b=M[p][q], c=M[q][p], d=M[q][q].
// Output: left/right cosine & sine factors of the two rotation angles.
void compute_rotation_factors(double a, double b, double c, double d,
                              double *cl, double *sl,
                              double *cr, double *sr);

// [svd_rotate.c] Apply the left+right rotations to M and accumulate
// them into U and V, using the factors from compute_rotation_factors().
void apply_rotations(mat_t M, mat_t U, mat_t V, int p, int q,
                     double cl, double sl, double cr, double sr);

#endif // SVD_COMMON_H
