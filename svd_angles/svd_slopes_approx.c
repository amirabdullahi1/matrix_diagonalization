/*
 * svd_angles.c   Rotation-angle / trig half of the Jacobi inner loop.
 *
 * Fixed-point TODO: replace the bodies below with the approximation for
 * arctan/sin/cos and return cl/sl/cr/sr in Q(TRIG_SHIFT). sin and cos MUST be
 * computed with matched accuracy, or the rotation stops being orthogonal.
 */
#define PI_OVER_2 1.5707963267948966
#include "svd_common.h"

/* ==== Arctan piecewise-linear approximation, x = o/a in [-1, 1] */
void compute_pla_arctan(double o, double a, double *theta)
{
    /* Breakpoints on [0, 1], step = 0.125 (8 segments, 9 points).
     * bp_y[i] = atan(bp_x[i]). atan is odd, so we only need x >= 0
     * here and flip the sign of the result at the end. */
    static const double bp_x[9] = {
        0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000
    };
    static const double bp_y[9] = {
        0.000000000, 0.124354995, 0.244978663, 0.358770670, 0.463647609,
        0.558599216, 0.643501109, 0.718830000, 0.785398163
    };
    /* slope[i] = (bp_y[i+1] - bp_y[i]) / (bp_x[i+1] - bp_x[i]), precomputed */
    static const double bp_slope[8] = {
        0.994839960, 0.964989344, 0.910336056, 0.839015512,
        0.759612856, 0.679215144, 0.602631128, 0.532545304
    };

    if (a == 0) {
        *theta = PI_OVER_2;
        return;
    }

    const double x  = o / a;
    const double ax = (x < 0.0) ? -x : x;

    /* which segment does ax fall in? */
    int i = (int)(ax / 0.125);
    if (i > 7) i = 7;   /* clamp in case ax rounds up to exactly 1.0 */

    /* linear interpolation using the precomputed slope for segment i */
    const double result = bp_y[i] + bp_slope[i] * (ax - bp_x[i]);

    *theta = (x < 0.0) ? -result : result;
}

/* ==== Cosine piecewise-linear approximation, theta in [-pi/2, pi/2] */
void compute_pla_cosine(double *cx, double theta_x)
{
    /* Breakpoints on [0, pi/2], step = pi/16 (8 segments, 9 points).
     * cos is even, so we only need theta >= 0 here. */
    static const double bp_t[9] = {
        0.00000000, 0.19634954, 0.39269908, 0.58904862, 0.78539816,
        0.98174770, 1.17809725, 1.37444679, 1.57079633
    };
    static const double bp_c[9] = {
        1.00000000, 0.98078528, 0.92387953, 0.83146961, 0.70710678,
        0.55557023, 0.38268343, 0.19509032, 0.00000000
    };
    /* slope[i] = (bp_c[i+1] - bp_c[i]) / (bp_t[i+1] - bp_t[i]), precomputed */
    static const double bp_slope[8] = {
        -0.097861, -0.289833, -0.470604, -0.633331,
        -0.771776, -0.880476, -0.955399, -0.993587
    };

    double at = (theta_x < 0.0) ? -theta_x : theta_x;
    if (at > PI_OVER_2) at = PI_OVER_2;   /* clamp */

    int i = (int)(at / 0.19634954);
    if (i > 7) i = 7;

    *cx = bp_c[i] + bp_slope[i] * (at - bp_t[i]);
}

/* ==== Sine piecewise-linear approximation, theta in [-pi/2, pi/2] */
void compute_pla_sine(double *sx, double theta_x)
{
    /* Same breakpoint grid as cosine above (must match, so cos/sin
     * error is matched segment-for-segment). sin is odd, so we only
     * need theta >= 0 here and flip the sign of the result at the end. */
    static const double bp_t[9] = {
        0.00000000, 0.19634954, 0.39269908, 0.58904862, 0.78539816,
        0.98174770, 1.17809725, 1.37444679, 1.57079633
    };
    static const double bp_s[9] = {
        0.00000000, 0.19509032, 0.38268343, 0.55557023, 0.70710678,
        0.83146961, 0.92387953, 0.98078528, 1.00000000
    };
    /* slope[i] = (bp_s[i+1] - bp_s[i]) / (bp_t[i+1] - bp_t[i]), precomputed */
    static const double bp_slope[8] = {
        0.993587, 0.955399, 0.880476, 0.771776,
        0.633331, 0.470604, 0.289833, 0.097861
    };

    double at = (theta_x < 0.0) ? -theta_x : theta_x;
    if (at > PI_OVER_2) at = PI_OVER_2;   /* clamp */

    int i = (int)(at / 0.19634954);
    if (i > 7) i = 7;

    const double result = bp_s[i] + bp_slope[i] * (at - bp_t[i]);

    *sx = (theta_x < 0.0) ? -result : result;
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

    compute_taylor_cosine(cl, theta_l);
    compute_taylor_sine(sl, theta_l);
    compute_taylor_cosine(cr, theta_r);
    compute_taylor_sine(sr, theta_r);
}
