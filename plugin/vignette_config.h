/* Per-camera correction config: GLib key-file parser, validation and the
 * falloff lookup table. Mirrors vignettecorrect/config.py. Depends on GLib only. */
#ifndef VIGNETTE_CONFIG_H
#define VIGNETTE_CONFIG_H

#include <string>
#include <vector>

#include "vignette_math.h"

struct VignetteConfig {
  std::vector<double> image_height_mm;        /* chart radii, strictly increasing from 0 */
  std::vector<double> relative_illumination;  /* chart brightness, centre = 1 */
  double pixel_size_um = 0.0;                 /* at the resolution of the frames */
  double centre_x = 0.5;                      /* lens centre, fraction of width */
  double centre_y = 0.5;                      /* lens centre, fraction of height */
  int mode = VC_MODE_KNEE;
  double knee = 200.0;
  int reference_width = 0;                    /* 0: use the frame's own size */
  int reference_height = 0;
};

/* Loads and validates a config file. On failure returns false and sets *error. */
bool vignette_config_load(const char *path, VignetteConfig *config, std::string *error);

/* Size whose half diagonal defines r = 1 for a width x height frame. */
void vignette_reference_size(const VignetteConfig &config, int width, int height,
                             int *ref_width, int *ref_height);

/* Falloff sampled evenly over normalised radius 0..VC_MAX_RADIUS, VC_TABLE_SIZE
 * entries, for a width x height frame. Same values as reference.lens_table. */
std::vector<double> vignette_build_table(const VignetteConfig &config, int width, int height);

/* Kernel parameters for a width x height frame. */
VcParams vignette_params(const VignetteConfig &config, int width, int height);

#endif /* VIGNETTE_CONFIG_H */
