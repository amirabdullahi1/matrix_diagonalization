/*
 * svd_angles.c   Rotation-angle / trig half of the Jacobi inner loop.
 *
 * OPTIMIZATION TARGET: this file is the transcendental bottleneck and the home
 * of the custom trig instruction + 2-issue firmware work.
 *
 * Current state: floating-point golden reference (atan2/sin/cos from math.h).
 * Fixed-point TODO: replace the bodies below with the approximation for
 * arctan/sin/cos and return cl/sl/cr/sr in Q(TRIG_SHIFT). sin and cos MUST be
 * computed with matched accuracy, or the rotation stops being orthogonal.
 */
#define PI_OVER_2 1.5707963267948966
#include "svd_common.h"

/* ==== Arctan Chebyshev Polynomial Expansion x */
void compute_chebyshev_arctan(double o, double a, double *theta) 
{
    /* ==== Arctan Chebyshev Polynomial Coefficients */
    const double coef_1 = 0.9993162682146682;
    const double coef_3 = -0.3222835041258624;
    const double coef_5 = 0.14902466975931372;
    const double coef_7 = -0.04085799434591237;

    if (a == 0) {
        *theta = PI_OVER_2;
        return;
    }

    const double x = o/a;
    const double x_sq = x * x;
    *theta = x * (
        coef_1 +
        x_sq * (
            coef_3 +
            x_sq * (
                coef_5 +
                x_sq * coef_7
            )
        )
    );
}

/* ==== Cosine Taylor Series Expansion theta_x */
void compute_taylor_cosine(double *cx, double theta_x)
{
    /* ==== Cosine Taylor Series Factorials
        2! = 1*2 = 2
        4! = 1*2*3*4 = 24
        6! = 1*2*3*4*5*6 = 720
    */

    /* ==== Cosine Taylor Series Coefficients */
    const double coef_2 = -1.0 / 2.0;
    const double coef_4 =  1.0 / 24.0;
    const double coef_6 = -1.0 / 720.0;

    const double theta_x_sq = theta_x * theta_x;
    *cx = 1.0 + theta_x_sq * (
        coef_2 + theta_x_sq * (
            coef_4 + theta_x_sq * coef_6
        )
    );
}

/* ==== Sine Taylor Series Expansion theta_x */
void compute_taylor_sine(double *sx, double theta_x) 
{
    /* ==== Sine Taylor Series Factorials
        3! = 1*2*3 = 6     
        5! = 1*2*3*4*5 = 120    
        7! = 1*2*3*4*5*6*7 = 5040
    */

    /* ==== Sine Taylor Series Coefficients */
    const double coef_3 = -1.0 / 6.0;
    const double coef_5 =  1.0 / 120.0;
    const double coef_7 = -1.0 / 5040.0;
    
    const double theta_x_sq = theta_x * theta_x;
    *sx = theta_x * (
        1.0 + theta_x_sq * (
            coef_3 + theta_x_sq * (
                coef_5 + theta_x_sq * coef_7
            )
        )
    );
}

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
     * -> approximation for integer arctan (the custom instruction);
     * schedule both across the two firmware issue slots to run concurrently. */
    double theta;
    double theta_sum;
    double theta_diff;
    
    const double num_sum = c + b;
    const double den_sum = d - a;

    const double num_diff = c - b;
    const double den_diff = d + a;

    if (fabs(num_sum) > fabs(den_sum)) {
        compute_chebyshev_arctan(den_sum, num_sum, &theta);
        theta_sum = PI_OVER_2 - theta;
    } else {
        compute_chebyshev_arctan(num_sum, den_sum, &theta);
        theta_sum = theta;
    }

    if (fabs(num_diff) > fabs(den_diff)) {
        compute_chebyshev_arctan(den_diff, num_diff, &theta);
        theta_diff = PI_OVER_2 - theta;
    } else {
        compute_chebyshev_arctan(num_diff, den_diff, &theta);
        theta_diff = theta;
    }

    const double theta_l = 0.5 * (theta_sum - theta_diff);
    const double theta_r = 0.5 * (theta_sum + theta_diff);

    /* ==== OPT[trig]: sin and cos must share accuracy to keep each rotation orthogonal.
     * Pack two evaluations per microcycle in the 2-issue firmware. */

    /* ==== Taylor Series Expansions */
    compute_taylor_cosine(cl, theta_l);
    compute_taylor_sine(sl, theta_l);
    compute_taylor_cosine(cr, theta_r);
    compute_taylor_sine(sr, theta_r);
}
