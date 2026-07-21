/*
 * svd_slopes_trig_new.c
 *
 * Three NEW custom instructions instead of doing the trig math in software.
 * Each obeys ARM's operand limit -- at most 2 inputs, 1 result:
 *
 *   EXECUTE_ARCTAN(o, a) -> theta   (2 in, 1 out)
 *   EXECUTE_COS(theta)   -> cx      (1 in, 1 out)
 *   EXECUTE_SIN(theta)   -> sx      (1 in, 1 out)
 *
 * This file documents the intended call site; it is deliberately NOT added
 * to the Makefile's SPLIT_SRC and will not compile/link as-is.
 */
#define PI_OVER_2 3216
#include "svd_common.h"

static inline __attribute__((always_inline))
int16_t iabs16(int16_t x)
{
    return (x < 0) ? -x : x;
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
    int16_t theta_sum;
    int16_t theta_diff;

    const int16_t num_sum = c + b;
    const int16_t den_sum = d - a;

    const int16_t num_diff = c - b;
    const int16_t den_diff = d + a;

    if (iabs16(num_sum) > iabs16(den_sum)) {
        __asm__ __volatile__ ("EXECUTE_ARCTAN %0, %1, %2" : "=r"(theta_sum) : "r"(den_sum), "r"(num_sum));
        theta_sum = PI_OVER_2 - theta_sum;
    } else {
        __asm__ __volatile__ ("EXECUTE_ARCTAN %0, %1, %2" : "=r"(theta_sum) : "r"(num_sum), "r"(den_sum));
    }

    if (iabs16(num_diff) > iabs16(den_diff)) {
        __asm__ __volatile__("EXECUTE_ARCTAN %0, %1, %2" : "=r"(theta_diff) : "r"(den_diff), "r"(num_diff));
        theta_diff = PI_OVER_2 - theta_diff;
    } else {
        __asm__ __volatile__ ("EXECUTE_ARCTAN %0, %1, %2" : "=r"(theta_diff) : "r"(num_diff), "r"(den_diff));
    }

    const int16_t theta_l = (theta_sum - theta_diff) / 2;
    const int16_t theta_r = (theta_sum + theta_diff) / 2;

    int16_t vcl, vsl, vcr, vsr;
    __asm__ __volatile__ ("EXECUTE_COS %0, %1" : "=r"(vcl) : "r"(theta_l));
    __asm__ __volatile__ ("EXECUTE_SIN %0, %1" : "=r"(vsl) : "r"(theta_l));
    __asm__ __volatile__ ("EXECUTE_COS %0, %1" : "=r"(vcr) : "r"(theta_r));
    __asm__ __volatile__ ("EXECUTE_SIN %0, %1" : "=r"(vsr) : "r"(theta_r));
    *cl = vcl;
    *sl = vsl;
    *cr = vcr;
    *sr = vsr;
}
