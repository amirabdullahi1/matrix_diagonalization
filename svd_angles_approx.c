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
    const double theta_l_sq = theta_l * theta_l;
    const double theta_r_sq = theta_r * theta_r;

    /* ==== OPT[trig]: four independent sin/cos -> piecewise-linear sin/cos.
     * sin and cos must share accuracy to keep each rotation orthogonal.
     * Pack two evaluations per microcycle in the 2-issue firmware. */
    
    /* ==== Taylor Series Factorials
    * sin   3! = 1*2*3 = 6,     5! = 1*2*3*4*5 = 120    7! = 1*2*3*4*5*6*7 = 5040
    * cos   2! = 1*2 = 2,       4! = 1*2*3*4 = 24       6! = 1*2*3*4*5*6 = 720*/
    
    /* ==== Taylor Series Coefficients */
    const double coeff_2 = -1.0f / 2.0f;
    const double coeff_4 =  1.0f / 24.0f;
    const double coeff_6 = -1.0f / 720.0f;

    const double coeff_3 = -1.0f / 6.0f;
    const double coeff_5 =  1.0f / 120.0f;
    const double coeff_7 = -1.0f / 5040.0f;

    /* ==== Taylor Series Cos Expansion theta_l */
    *cl = 1.0f + theta_l_sq * (
        coeff_2 + theta_l_sq * (
            coeff_4 + theta_l_sq * coeff_6
        )
    );
    
    /* ==== Taylor Series Sin Expansion theta_l */
    *sl = theta_l * (
        1.0f + theta_l_sq * (
            coeff_3 + theta_l_sq * (
                coeff_5 + theta_l_sq * coeff_7
            )
        )
    );

    /* ==== Taylor Series Cos Expansion theta_r */
    *cr = 1.0f + theta_r_sq * (
        coeff_2 + theta_r_sq * (
            coeff_4 + theta_r_sq * coeff_6
        )
    );

    /* ==== Taylor Series Sin Expansion theta_r */
    *sr = theta_r * (
        1.0f + theta_r_sq * (
            coeff_3 + theta_r_sq * (
                coeff_5 + theta_r_sq * coeff_7
            )
        )
    );
}
