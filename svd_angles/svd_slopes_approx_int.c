/*
 * svd_angles.c   Rotation-angle / trig half of the Jacobi inner loop.
 *
 * Fixed-point TODO: replace the bodies below with the approximation for
 * arctan/sin/cos and return cl/sl/cr/sr in Q(TRIG_SHIFT). sin and cos MUST be
 * computed with matched accuracy, or the rotation stops being orthogonal.
 *
 * Q11 fixed point: every constant below is the real value multiplied by
 * 2^PLA_SF = 2048 with the fractional remainder truncated (matches a plain
 * (int16_t) cast, i.e. truncation toward zero).
 */
#define PLA_SF 11
#define PI_OVER_2 3216
#include "svd_common.h"

static inline int16_t iabs16(int16_t x)
{
    return (x < 0) ? -x : x;
}

/* ==== Arctan piecewise-linear approximation, x = o/a in [-1, 1] */
void compute_pla_arctan(int16_t o, int16_t a, int16_t *theta)
{
    /* Breakpoints on [0, 1], step = 0.125 (8 segments, 9 points), Q11.
     * bp_y[i] = trunc(atan(bp_x_real[i]) * 2048). atan is odd, so we only
     * need x >= 0 here and flip the sign of the result at the end. */
    static const int16_t bp_x[9] = {
        0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048
    };
    static const int16_t bp_y[9] = {
        0, 254, 501, 734, 949, 1144, 1317, 1472, 1608
    };
    static const int16_t bp_slope[8] = {
        2037, 1976, 1864, 1718, 1555, 1391, 1234, 1090
    };

    if (a == 0) {
        *theta = PI_OVER_2;
        return;
    }

    /* Pre-scale o by 2048 before the divide so the quotient keeps its
     * fractional bits in Q11 (a plain integer divide would otherwise
     * truncate every |o/a| < 1 straight to 0). int32_t avoids overflow
     * on the shift. */
    const int16_t x  = (int16_t)(((int32_t)o << PLA_SF) / a);
    const int16_t ax = iabs16(x);

    /* which segment does ax fall in? (2^8 = 256 = trunc(0.125 * 2048)) */
    int i = ax >> 8;
    if (i > 7) i = 7;   /* clamp in case ax rounds up to exactly 1.0 */

    /* linear interpolation using the precomputed slope for segment i.
     * bp_slope[i] * delta is Q22, so shift back down to Q11 before adding. */
    const int16_t result = bp_y[i] +
        (int16_t)(((int32_t)bp_slope[i] * (ax - bp_x[i])) >> PLA_SF);

    *theta = (x < 0) ? -result : result;
}

/* ==== Cosine piecewise-linear approximation, theta in [-pi/2, pi/2] */
void compute_pla_cosine(int16_t *cx, int16_t theta_x)
{
    /* Breakpoints on [0, pi/2], step = pi/16 (8 segments, 9 points), Q11.
     * cos is even, so we only need theta >= 0 here. */
    static const int16_t bp_t[9] = {
        0, 402, 804, 1206, 1608, 2010, 2412, 2814, 3216
    };
    static const int16_t bp_c[9] = {
        2048, 2008, 1892, 1702, 1448, 1137, 783, 399, 0
    };
    static const int16_t bp_slope[8] = {
        -200, -593, -963, -1297, -1580, -1803, -1956, -2034
    };

    int16_t at = iabs16(theta_x);
    if (at > PI_OVER_2) at = PI_OVER_2;   /* clamp */

    /* which segment does at fall in? (402 = trunc(pi/16 * 2048)) 
     * i = at / 402, replaced with a multiply-shift (M=2609, S=20) 
    * so no integer divide is needed. Exact match to floor(at/402). */
    int i = ((int32_t)at * 2609) >> 20;
    if (i > 7) i = 7;

    *cx = bp_c[i] +
        (int16_t)(((int32_t)bp_slope[i] * (at - bp_t[i])) >> PLA_SF);
}

/* ==== Sine piecewise-linear approximation, theta in [-pi/2, pi/2] */
void compute_pla_sine(int16_t *sx, int16_t theta_x)
{
    /* Same breakpoint grid as cosine above (must match, so cos/sin
     * error is matched segment-for-segment). sin is odd, so we only
     * need theta >= 0 here and flip the sign of the result at the end. */
    static const int16_t bp_t[9] = {
        0, 402, 804, 1206, 1608, 2010, 2412, 2814, 3216
    };
    static const int16_t bp_s[9] = {
        0, 399, 783, 1137, 1448, 1702, 1892, 2008, 2048
    };
    static const int16_t bp_slope[8] = {
        2034, 1956, 1803, 1580, 1297, 963, 593, 200
    };

    int16_t at = iabs16(theta_x);
    if (at > PI_OVER_2) at = PI_OVER_2;   /* clamp */

    /* which segment does at fall in? (402 = trunc(pi/16 * 2048)) 
     * i = at / 402, replaced with a multiply-shift (M=2609, S=20) 
     * so no integer divide is needed. Exact match to floor(at/402). */
    int i = ((int32_t)at * 2609) >> 20;
    if (i > 7) i = 7;

    const int16_t result = bp_s[i] +
        (int16_t)(((int32_t)bp_slope[i] * (at - bp_t[i])) >> PLA_SF);

    *sx = (theta_x < 0) ? -result : result;
}

/*
 * Two-angle method:
 *   theta_sum  = atan2(c + b, d - a)
 *   theta_diff = atan2(c - b, d + a)
 *   theta_l = (theta_sum - theta_diff) / 2   (left rotation angle)
 *   theta_r = (theta_sum + theta_diff) / 2   (right rotation angle)
 */
void compute_rotation_factors(int16_t a, int16_t b, int16_t c, int16_t d,
                              int16_t *cl, int16_t *sl,
                              int16_t *cr, int16_t *sr)
{
    int16_t theta;
    int16_t theta_sum;
    int16_t theta_diff;

    const int16_t num_sum = c + b;
    const int16_t den_sum = d - a;

    const int16_t num_diff = c - b;
    const int16_t den_diff = d + a;

    if (iabs16(num_sum) > iabs16(den_sum)) {
        compute_pla_arctan(den_sum, num_sum, &theta);
        theta_sum = PI_OVER_2 - theta;
    } else {
        compute_pla_arctan(num_sum, den_sum, &theta);
        theta_sum = theta;
    }

    if (iabs16(num_diff) > iabs16(den_diff)) {
        compute_pla_arctan(den_diff, num_diff, &theta);
        theta_diff = PI_OVER_2 - theta;
    } else {
        compute_pla_arctan(num_diff, den_diff, &theta);
        theta_diff = theta;
    }

    const int16_t theta_l = (theta_sum - theta_diff) / 2;
    const int16_t theta_r = (theta_sum + theta_diff) / 2;

    compute_pla_cosine(cl, theta_l);
    compute_pla_sine(sl, theta_l);
    compute_pla_cosine(cr, theta_r);
    compute_pla_sine(sr, theta_r);
}

int main (void) {
    return 0;
}
