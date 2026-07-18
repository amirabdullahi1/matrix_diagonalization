/*
 * svd_common.h - shared configuration and the interface contract between
 * the two halves of the Jacobi inner loop (FIXED-POINT build).
 *
 *   compute_rotation_factors()   [svd_angles/svd_slopes_approx_int.c - Amir]
 *       (a,b,c,d) int16  ->  cl,sl,cr,sr int16 (Q11)   piecewise-linear trig
 *
 *   apply_rotations()            [svd_rotate.c - Param]
 *       Q11 factors + int16 matrices  ->  rotated M, U, V   (MAC kernels)
 *
 * Fixed-point layout:
 *   M entries : raw signed integers (Q0)  -- the data matrix / singular values
 *   U, V      : Q11 (value * 2048)        -- orthogonal factors, |entry| <= 1
 *   cl,sl,... : Q11 rotation factors
 * apply_rotations() applies the SAME `>> TRIG_SHIFT` to all three, because each
 * MAC multiplies the operand by a Q11 factor (one Q11 scale to remove).
 */
#ifndef SVD_COMMON_H
#define SVD_COMMON_H

#include <math.h>
#include <stdint.h>

#define N 6                       // matrix dimension (6x6 per spec)
#define MAX_SWEEPS 50             // hard cap on sweeps (fixed-point may need more)
#define TRIG_SHIFT 11             // Q11: 1.0 == 2048  (matches trig PLA_SF)
#define OFFDIAG_EPS 3             // convergence: max |off-diagonal| in raw units

// 12-bit signed input range: each matrix entry lies in [-2048, 2047].
#define INPUT_BITS 12
#define INPUT_MIN  (-(1 << (INPUT_BITS - 1)))   // -2048
#define INPUT_MAX  ((1 << (INPUT_BITS - 1)) - 1)//  2047

typedef int16_t mat_t[N][N];

// ---- The seam ----------------------------------------------------------

// [svd_angles/svd_slopes_approx_int.c / Amir] Rotation-angle + factor generation.
// Input : the 2x2 block  a=M[p][p], b=M[p][q], c=M[q][p], d=M[q][q]  (raw int16).
// Output: left/right cosine & sine factors in Q11.
void compute_rotation_factors(int16_t a, int16_t b, int16_t c, int16_t d,
                              int16_t *cl, int16_t *sl,
                              int16_t *cr, int16_t *sr);

// [svd_rotate.c / Param] Apply the left+right rotations to M and accumulate
// them into U and V, using the Q11 factors from compute_rotation_factors().
void apply_rotations(mat_t M, mat_t U, mat_t V, int p, int q,
                     int16_t cl, int16_t sl, int16_t cr, int16_t sr);

#endif // SVD_COMMON_H
