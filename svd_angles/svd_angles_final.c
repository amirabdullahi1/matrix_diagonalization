/*
 * svd_angles_taylor.c -- Rotation-angle / trig half using TAYLOR-SERIES sin/cos.
 *
 * Replaces the piecewise-linear sin/cos, which systematically undershot the
 * unit circle (cos^2+sin^2 averaged 0.76% below 1), contracting every rotation
 * ~0.38% and losing ~25% of the singular values over the 51 non-converging
 * sweeps. Taylor sin/cos are accurate to < ~1 Q11, so cos^2+sin^2 ~ 1 and each
 * rotation stays (near) orthogonal.
 *
 * Arctan is kept piecewise-linear -- it is already accurate to ~0.002 rad and
 * was not the source of the error.
 *
 * Q11 fixed point: 2^PLA_SF = 2048 == 1.0. Angles are in [-pi/2, pi/2].
 *   sin(x) = x - x^3/6 + x^5/120 - x^7/5040
 *   cos(x) = 1 - x^2/2 + x^4/24 - x^6/720
 */
#define PLA_SF 11
#define PI_OVER_2 3216
#include "svd_common.h"

static inline int16_t iabs16(int16_t x) { return (x < 0) ? -x : x; }

/* Q11 multiply: (a*b) with one 2^11 scale removed. int32 avoids overflow. */
static inline int32_t qmul(int32_t a, int32_t b) { return (a * b) >> PLA_SF; }

/* ==== Arctan piecewise-linear (kept -- accurate), x = o/a in [-1, 1]. */
static void compute_pla_arctan(int16_t o, int16_t a, int16_t *theta)
{
    static const int16_t bp_x[9]     = {0,256,512,768,1024,1280,1536,1792,2048};
    static const int16_t bp_y[9]     = {0,254,501,734,949,1144,1317,1472,1608};
    static const int16_t bp_slope[8] = {2037,1976,1864,1718,1555,1391,1234,1090};

    if (a == 0) { *theta = PI_OVER_2; return; }
    const int16_t x  = (int16_t)(((int32_t)o << PLA_SF) / a);
    const int16_t ax = iabs16(x);
    int i = ax >> 8; if (i > 7) i = 7;
    const int16_t r = bp_y[i] +
        (int16_t)(((int32_t)bp_slope[i] * (ax - bp_x[i])) >> PLA_SF);
    *theta = (x < 0) ? -r : r;
}

/* ==== Taylor sine, theta_x in Q11, [-pi/2, pi/2]. */
static void compute_taylor_sine(int16_t *sx, int16_t theta_x)
{
    const int32_t x  = theta_x;
    const int32_t x2 = qmul(x, x);
    const int32_t x3 = qmul(x2, x);
    const int32_t x5 = qmul(x3, x2);
    const int32_t x7 = qmul(x5, x2);
    *sx = (int16_t)(x - x3 / 6 + x5 / 120 - x7 / 5040);
}

/* ==== Taylor cosine, theta_x in Q11, [-pi/2, pi/2]. */
static void compute_taylor_cosine(int16_t *cx, int16_t theta_x)
{
    const int32_t x  = theta_x;
    const int32_t x2 = qmul(x, x);
    const int32_t x4 = qmul(x2, x2);
    const int32_t x6 = qmul(x4, x2);
    *cx = (int16_t)((1 << PLA_SF) - x2 / 2 + x4 / 24 - x6 / 720);
}

/*
 * Two-angle method:
 *   theta_sum  = atan2(c + b, d - a)   theta_diff = atan2(c - b, d + a)
 *   theta_l = (theta_sum - theta_diff)/2   theta_r = (theta_sum + theta_diff)/2
 */
void compute_rotation_factors(int16_t a, int16_t b, int16_t c, int16_t d,
                              int16_t *cl, int16_t *sl,
                              int16_t *cr, int16_t *sr)
{
    int16_t theta_sum, theta_diff;
    const int16_t num_sum  = c + b, den_sum  = d - a;
    const int16_t num_diff = c - b, den_diff = d + a;

    if (iabs16(num_sum) > iabs16(den_sum)) {
        compute_pla_arctan(den_sum, num_sum, &theta_sum);
        theta_sum = PI_OVER_2 - theta_sum;
    } else {
        compute_pla_arctan(num_sum, den_sum, &theta_sum);
    }
    if (iabs16(num_diff) > iabs16(den_diff)) {
        compute_pla_arctan(den_diff, num_diff, &theta_diff);
        theta_diff = PI_OVER_2 - theta_diff;
    } else {
        compute_pla_arctan(num_diff, den_diff, &theta_diff);
    }

    const int16_t theta_l = (theta_sum - theta_diff) / 2;
    const int16_t theta_r = (theta_sum + theta_diff) / 2;

    compute_taylor_cosine(cl, theta_l);
    compute_taylor_sine(sl, theta_l);
    compute_taylor_cosine(cr, theta_r);
    compute_taylor_sine(sr, theta_r);
}
