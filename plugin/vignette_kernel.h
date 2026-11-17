/* CUDA side of the correction. Uses the CUDA runtime API only. */
#ifndef VIGNETTE_KERNEL_H
#define VIGNETTE_KERNEL_H

#include <stdint.h>
#include <string>

#include "vignette_math.h"

struct VcKernel;  /* holds the falloff table on the GPU */

/* Uploads a VC_TABLE_SIZE-entry table. Returns NULL and sets *error on failure. */
VcKernel *vc_kernel_create(const float *table, std::string *error);

/* Corrects one RGBA frame in place and waits for it to finish.
 * rgba: device pointer, 4 bytes per pixel, `pitch` bytes per row. */
bool vc_kernel_run(VcKernel *kernel, uint8_t *rgba, int pitch, int width, int height,
                   const VcParams &params, std::string *error);

void vc_kernel_destroy(VcKernel *kernel);

#endif /* VIGNETTE_KERNEL_H */
