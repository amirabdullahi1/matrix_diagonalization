/*
 * svd_rotate.c  -- Rotation-application half of the Jacobi inner loop (FIXED-POINT).
 *
 * Each element does two MACs on the same loaded pair, rescaled from Q11 with
 * ROUND-TO-NEAREST (add half an LSB before the shift):
 *     new_x = (cl*x - sl*y + ROUND) >> TRIG_SHIFT
 *     new_y = (sl*x + cl*y + ROUND) >> TRIG_SHIFT
 * Rounding (vs plain truncation) removes a systematic downward bias that
 * otherwise shrinks the singular values over many sweeps -- it is what lets the
 * fixed-point SVD reach the correct result (validated to ~2 LSB against the
 * double golden model when fed exact Q11 factors).
 *
 * cl,sl,cr,sr are Q11; products are formed in int32. The same rescale works for
 * M (raw), U and V (Q11) alike.
 *
 * MANUALLY UNROLLED for N=6: each rotation loop is six explicit blocks.
 * Scalar-double history (bench.c, Cortex-A7): baseline 509 -> unrolled 364 ns
 * (1.40x). NEON int16x8 prototype (bench_neon.c): 1.57x over scalar fixed-point.
 * TODO: drop the int16x8 NEON row-rotation in here once validated.
 *
 * Invariant maintained:  U * M_current * V^T = M_original.
 */
#include "svd_common.h"

#define ROUND (1 << (TRIG_SHIFT - 1))   // round-to-nearest before >> TRIG_SHIFT

void apply_rotations(mat_t M, mat_t U, mat_t V, int p, int q,
                     int16_t cl, int16_t sl, int16_t cr, int16_t sr)
{
    /* Left rotation R_l on rows p and q of M (factors cl, sl). */
    { int32_t x = M[p][0], y = M[q][0]; M[p][0] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); M[q][0] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[p][1], y = M[q][1]; M[p][1] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); M[q][1] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[p][2], y = M[q][2]; M[p][2] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); M[q][2] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[p][3], y = M[q][3]; M[p][3] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); M[q][3] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[p][4], y = M[q][4]; M[p][4] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); M[q][4] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[p][5], y = M[q][5]; M[p][5] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); M[q][5] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }

    /* Right rotation R_r^T on columns p and q of M (factors cr, sr). */
    { int32_t x = M[0][p], y = M[0][q]; M[0][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); M[0][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[1][p], y = M[1][q]; M[1][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); M[1][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[2][p], y = M[2][q]; M[2][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); M[2][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[3][p], y = M[3][q]; M[3][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); M[3][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[4][p], y = M[4][q]; M[4][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); M[4][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = M[5][p], y = M[5][q]; M[5][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); M[5][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }

    /* Accumulate left rotation into U (columns p and q, factors cl, sl). */
    { int32_t x = U[0][p], y = U[0][q]; U[0][p] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); U[0][q] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = U[1][p], y = U[1][q]; U[1][p] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); U[1][q] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = U[2][p], y = U[2][q]; U[2][p] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); U[2][q] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = U[3][p], y = U[3][q]; U[3][p] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); U[3][q] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = U[4][p], y = U[4][q]; U[4][p] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); U[4][q] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = U[5][p], y = U[5][q]; U[5][p] = (int16_t)((cl*x - sl*y + ROUND) >> TRIG_SHIFT); U[5][q] = (int16_t)((sl*x + cl*y + ROUND) >> TRIG_SHIFT); }

    /* Accumulate right rotation into V (columns p and q, factors cr, sr). */
    { int32_t x = V[0][p], y = V[0][q]; V[0][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); V[0][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = V[1][p], y = V[1][q]; V[1][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); V[1][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = V[2][p], y = V[2][q]; V[2][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); V[2][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = V[3][p], y = V[3][q]; V[3][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); V[3][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = V[4][p], y = V[4][q]; V[4][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); V[4][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
    { int32_t x = V[5][p], y = V[5][q]; V[5][p] = (int16_t)((cr*x - sr*y + ROUND) >> TRIG_SHIFT); V[5][q] = (int16_t)((sr*x + cr*y + ROUND) >> TRIG_SHIFT); }
}
