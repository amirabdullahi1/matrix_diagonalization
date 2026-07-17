/*
 * svd_angles.c   Rotation-angle / trig half of the Jacobi inner loop.
 *
 * OPTIMIZATION TARGET: this file is the transcendental bottleneck and the home
 * of the custom trig instruction + 2-issue firmware work.
 *
 * Current state: floating-point golden reference (atan2/sin/cos from math.h).
 * Fixed-point TODO: replace the bodies below with the piecewise-linear integer
 * arctan/sin/cos and return cl/sl/cr/sr in Q(TRIG_SHIFT). sin and cos MUST be
 * computed with matched accuracy, or the rotation stops being orthogonal.
 */
#include "svd_common.h"

/*
 * Two-angle method:
 *   theta_sum  = atan2(c + b, d - a)
 *   theta_diff = atan2(c - b, d + a)
 *   theta_l = (theta_sum - theta_diff) / 2   (left rotation angle)
 *   theta_r = (theta_sum + theta_diff) / 2   (right rotation angle)
 */
void compute_rotation_factors(double a, double b, double c, double d,
                              double *cl, double *sl,
                              double *cr, double *sr)
{
    /* ==== OPT[trig]: two data-independent arctan evaluations.
     * -> piecewise-linear integer arctan (the custom instruction);
     * schedule both across the two firmware issue slots to run concurrently. */
    const double theta_sum  = atan2(c + b, d - a);
    const double theta_diff = atan2(c - b, d + a);
    const double theta_l = 0.5 * (theta_sum - theta_diff);
    const double theta_r = 0.5 * (theta_sum + theta_diff);

    /* ==== OPT[trig]: four independent sin/cos -> piecewise-linear sin/cos.
     * sin and cos must share accuracy to keep each rotation orthogonal.
     * Pack two evaluations per microcycle in the 2-issue firmware. */
    *cl = cos(theta_l); *sl = sin(theta_l);
    *cr = cos(theta_r); *sr = sin(theta_r);
}
