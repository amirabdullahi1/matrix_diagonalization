/*
 * svd_rotate.c  -- Rotation-application half of the Jacobi inner loop.
 *
 * OPTIMIZATION TARGET: the four MAC kernels below. Each iteration does two
 * independent updates of the form (c*x - s*y) and (s*x + c*y) on the same
 * loaded pair -> one per firmware issue slot; iterations are independent
 * across k -> unroll (N=6 is fixed) and software-pipeline the loads, or
 * vectorize as 4-lane NEON MACs.
 *
 * Fixed-point TODO: when cl/sl/cr/sr arrive as Q(TRIG_SHIFT) integers, rescale
 * each product with `>> TRIG_SHIFT` and use int32_t accumulators.
 *
 * Invariant maintained:  U * M_current * V^T = M_original.
 * After  M <- R_l * M * R_r^T, that forces  U <- U * R_l^T  and  V <- V * R_r^T.
 */
#include "svd_common.h"

void apply_rotations(mat_t M, mat_t U, mat_t V, int p, int q,
                     double cl, double sl, double cr, double sr)
{
    /* Left rotation R_l on rows p and q of M  (M <- R_l * M):
     *   new row_p =  cl * row_p - sl * row_q
     *   new row_q =  sl * row_p + cl * row_q                      */
    for (int k = 0; k < N; k++) {
        const double mp = M[p][k];
        const double mq = M[q][k];
        M[p][k] =  cl * mp - sl * mq;
        M[q][k] =  sl * mp + cl * mq;
    }

    /* Right rotation R_r^T on columns p and q of M  (M <- M * R_r^T):
     *   new col_p =  cr * col_p - sr * col_q
     *   new col_q =  sr * col_p + cr * col_q                      */
    for (int k = 0; k < N; k++) {
        const double mp = M[k][p];
        const double mq = M[k][q];
        M[k][p] =  cr * mp - sr * mq;
        M[k][q] =  sr * mp + cr * mq;
    }

    /* Accumulate left rotation into U  (U <- U * R_l^T), columns p and q:
     *   new U[:,p] =  cl * U[:,p] - sl * U[:,q]
     *   new U[:,q] =  sl * U[:,p] + cl * U[:,q]                   */
    for (int k = 0; k < N; k++) {
        const double up = U[k][p];
        const double uq = U[k][q];
        U[k][p] =  cl * up - sl * uq;
        U[k][q] =  sl * up + cl * uq;
    }

    /* Accumulate right rotation into V  (V <- V * R_r^T), same pattern. */
    for (int k = 0; k < N; k++) {
        const double vp = V[k][p];
        const double vq = V[k][q];
        V[k][p] =  cr * vp - sr * vq;
        V[k][q] =  sr * vp + cr * vq;
    }
}
