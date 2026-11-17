/* Per-pixel maths of the lens falloff correction.
 *
 * Shared by the CUDA kernel and the host-side test tool, so the code that is
 * tested against the numpy reference on a desktop is the code that runs on the
 * GPU. Must match vignettecorrect/reference.py.
 */
#ifndef VIGNETTE_MATH_H
#define VIGNETTE_MATH_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define VC_FN __host__ __device__ inline
#else
#define VC_FN static inline
#endif

#define VC_TABLE_SIZE 301    /* falloff samples over normalised radius 0..VC_MAX_RADIUS */
#define VC_MAX_RADIUS 1.5f

#define VC_MODE_KNEE 0
#define VC_MODE_CLIP 1

typedef struct {
  float centre_x_px;       /* lens centre, pixels from the left edge */
  float centre_y_px;       /* lens centre, pixels from the top edge */
  float half_diagonal_px;  /* half diagonal of the reference frame: r = 1 */
  float knee;              /* highlight protection point, used in knee mode */
  int mode;                /* VC_MODE_KNEE or VC_MODE_CLIP */
} VcParams;

/* Gain (>= 1) for the pixel at column x, row y. table has VC_TABLE_SIZE entries. */
VC_FN float vc_gain(const float *table, const VcParams *p, int x, int y)
{
  /* Radius is measured from the pixel's centre, hence the + 0.5. */
  float dx = (float) x + 0.5f - p->centre_x_px;
  float dy = (float) y + 0.5f - p->centre_y_px;
  float r = sqrtf(dx * dx + dy * dy) / p->half_diagonal_px;
  float pos = r * ((VC_TABLE_SIZE - 1) / VC_MAX_RADIUS);
  if (pos >= (float) (VC_TABLE_SIZE - 1))
    return 1.0f / table[VC_TABLE_SIZE - 1];  /* past the table: hold the last value */
  int i = (int) pos;
  float falloff = table[i] + (pos - (float) i) * (table[i + 1] - table[i]);
  return 1.0f / falloff;
}

VC_FN uint32_t vc_round_u8(float value)
{
  /* rintf rounds half to even, as numpy.round does. */
  return (uint32_t) fminf(255.0f, rintf(value));
}

/* Correct one pixel held as a little-endian 32-bit word: three colour bytes in
 * the low 24 bits (their order does not matter) and a fourth byte, alpha, in
 * the top 8 bits, which is returned unchanged. */
VC_FN uint32_t vc_correct_word(uint32_t pixel, float gain, const VcParams *p)
{
  float c0 = (float) (pixel & 0xFFu);
  float c1 = (float) ((pixel >> 8) & 0xFFu);
  float c2 = (float) ((pixel >> 16) & 0xFFu);
  float scale = gain;  /* clip mode: clip(round(pixel * gain)) per channel */
  if (p->mode == VC_MODE_KNEE) {
    float brightest = fmaxf(c0, fmaxf(c1, c2));
    float start = p->knee / gain;  /* original value that reaches the knee at full gain */
    float target = brightest > start
        ? p->knee + (255.0f - p->knee) * (brightest - start) / (255.0f - start)
        : brightest * gain;
    scale = target / fmaxf(brightest, 1.0f);
  }
  return (pixel & 0xFF000000u) | vc_round_u8(c0 * scale) | (vc_round_u8(c1 * scale) << 8) |
         (vc_round_u8(c2 * scale) << 16);
}

#endif /* VIGNETTE_MATH_H */
