/*
 * bench_neon.c  --  standalone int16x8 NEON prototype for the rotation MACs.
 *
 * Goal: prove that vectorizing the rotation with NEON beats the scalar
 * fixed-point version, and get a speedup number -- WITHOUT needing the full
 * mat_t migration. This is Param's headline experiment.
 *
 * It rotates two int16 rows (rp, rq) by Q11 factors (cl, sl):
 *     rp' = (cl*rp - sl*rq) >> 11
 *     rq' = (sl*rp + cl*rq) >> 11
 * -- exactly the row-rotation MAC from apply_rotations(), but in fixed point.
 *
 * The row is padded from N=6 to 8 so it fills one int16x8 vector cleanly.
 * (The strided column rotations don't vectorize this way -- that's the
 * documented finding; this prototype targets the contiguous row rotation.)
 *
 * Build (on the Cortex-A7 VM, with the -mfpu=neon flags on):  make neon
 * Run:   ./bench_neon
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define LEN         8          // N=6 padded to a full int16x8 vector
#define TRIG_SHIFT  11         // Q11: 1.0 == 2048
#define ITERS       5000000L

/* ---- scalar fixed-point rotation (the baseline to beat) ---- */
__attribute__((noinline))
static void rot_scalar(int16_t *rp, int16_t *rq, int cl, int sl)
{
    for (int k = 0; k < LEN; k++) {
        int32_t p = rp[k], q = rq[k];
        rp[k] = (int16_t)((cl * p - sl * q) >> TRIG_SHIFT);
        rq[k] = (int16_t)((sl * p + cl * q) >> TRIG_SHIFT);
    }
}

/* ---- NEON int16x8 rotation: all 8 lanes at once ---- */
__attribute__((noinline))
static void rot_neon(int16_t *rp, int16_t *rq, int cl, int sl)
{
    const int16x8_t vp = vld1q_s16(rp);
    const int16x8_t vq = vld1q_s16(rq);
    const int16x4_t clv = vdup_n_s16((int16_t)cl);
    const int16x4_t slv = vdup_n_s16((int16_t)sl);

    /* rp' = cl*rp - sl*rq  (widen int16->int32, low and high halves) */
    int32x4_t p_lo = vmull_s16(vget_low_s16(vp),  clv);
    p_lo = vmlsl_s16(p_lo, vget_low_s16(vq),  slv);
    int32x4_t p_hi = vmull_s16(vget_high_s16(vp), clv);
    p_hi = vmlsl_s16(p_hi, vget_high_s16(vq), slv);

    /* rq' = sl*rp + cl*rq */
    int32x4_t q_lo = vmull_s16(vget_low_s16(vp),  slv);
    q_lo = vmlal_s16(q_lo, vget_low_s16(vq),  clv);
    int32x4_t q_hi = vmull_s16(vget_high_s16(vp), slv);
    q_hi = vmlal_s16(q_hi, vget_high_s16(vq), clv);

    /* shift right by 11 and narrow back to int16, recombine to 8 lanes */
    const int16x8_t rp_new = vcombine_s16(vshrn_n_s32(p_lo, TRIG_SHIFT),
                                          vshrn_n_s32(p_hi, TRIG_SHIFT));
    const int16x8_t rq_new = vcombine_s16(vshrn_n_s32(q_lo, TRIG_SHIFT),
                                          vshrn_n_s32(q_hi, TRIG_SHIFT));
    vst1q_s16(rp, rp_new);
    vst1q_s16(rq, rq_new);
}

static double ns_per_call(void (*fn)(int16_t *, int16_t *, int, int),
                          int16_t *rp, int16_t *rq, int cl, int sl)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long it = 0; it < ITERS; it++)
        fn(rp, rq, cl, sl);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ITERS;
}

int main(void)
{
    /* Q11 factors for a ~0.1 rad rotation (orthogonal -> values stay bounded). */
    const int cl = 2038, sl = 204;   // round(cos(0.1)*2048), round(sin(0.1)*2048)

    int16_t seed_p[LEN] = { 1000, -800, 600, -400, 200, -100, 0, 0 };
    int16_t seed_q[LEN] = { -500,  700, -300, 900, -600, 250, 0, 0 };

    /* --- correctness: scalar vs NEON on identical input --- */
    int16_t sp[LEN], sq[LEN], np[LEN], nq[LEN];
    memcpy(sp, seed_p, sizeof sp); memcpy(sq, seed_q, sizeof sq);
    memcpy(np, seed_p, sizeof np); memcpy(nq, seed_q, sizeof nq);
    rot_scalar(sp, sq, cl, sl);
    rot_neon(np, nq, cl, sl);
    int ok = (memcmp(sp, np, sizeof sp) == 0) && (memcmp(sq, nq, sizeof sq) == 0);
    printf("correctness (NEON == scalar): %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("  scalar rp:"); for (int k=0;k<LEN;k++) printf(" %d", sp[k]);
        printf("\n  neon   rp:"); for (int k=0;k<LEN;k++) printf(" %d", np[k]);
        printf("\n");
    }

    /* --- timing --- */
    int16_t a[LEN], b[LEN];
    memcpy(a, seed_p, sizeof a); memcpy(b, seed_q, sizeof b);
    double t_scalar = ns_per_call(rot_scalar, a, b, cl, sl);
    memcpy(a, seed_p, sizeof a); memcpy(b, seed_q, sizeof b);
    double t_neon = ns_per_call(rot_neon, a, b, cl, sl);

    printf("scalar fixed-point : %.2f ns/call\n", t_scalar);
    printf("NEON  int16x8      : %.2f ns/call\n", t_neon);
    printf("speedup            : %.2fx\n", t_scalar / t_neon);

    /* checksum so nothing is optimized away */
    long sum = 0; for (int k=0;k<LEN;k++) sum += a[k] + b[k];
    printf("checksum (ignore): %ld\n", sum);
    return 0;
}
