#include "vignette_config.h"

#include <cmath>
#include <cstring>

#include <glib.h>

static const char *GROUP = "vignette";

static const char *KNOWN_KEYS[] = {
  "image-height-mm", "relative-illumination", "pixel-size-um", "centre-x", "centre-y",
  "highlight-mode", "knee", "reference-width", "reference-height",
};

static bool fail(std::string *error, const std::string &message)
{
  if (error)
    *error = message;
  return false;
}

static bool take_error(GError *gerror, const char *key, std::string *error)
{
  std::string message = std::string(key) + ": " + gerror->message;
  g_error_free(gerror);
  return fail(error, message);
}

static bool read_double_list(GKeyFile *file, const char *key, std::vector<double> *out,
                             std::string *error)
{
  GError *gerror = NULL;
  gsize length = 0;
  gdouble *values = g_key_file_get_double_list(file, GROUP, key, &length, &gerror);
  if (gerror)
    return take_error(gerror, key, error);
  out->assign(values, values + length);
  g_free(values);
  return true;
}

/* Reads an optional double; leaves *out alone when the key is absent. */
static bool read_double(GKeyFile *file, const char *key, double *out, std::string *error)
{
  if (!g_key_file_has_key(file, GROUP, key, NULL))
    return true;
  GError *gerror = NULL;
  double value = g_key_file_get_double(file, GROUP, key, &gerror);
  if (gerror)
    return take_error(gerror, key, error);
  *out = value;
  return true;
}

static bool read_int(GKeyFile *file, const char *key, int *out, std::string *error)
{
  if (!g_key_file_has_key(file, GROUP, key, NULL))
    return true;
  GError *gerror = NULL;
  int value = g_key_file_get_integer(file, GROUP, key, &gerror);
  if (gerror)
    return take_error(gerror, key, error);
  *out = value;
  return true;
}

static bool validate(const VignetteConfig &c, std::string *error)
{
  const std::vector<double> &mm = c.image_height_mm;
  const std::vector<double> &ri = c.relative_illumination;
  if (mm.size() < 2 || mm.size() != ri.size())
    return fail(error, "chart needs two or more points, and equal-length lists");
  if (mm[0] != 0.0)
    return fail(error, "image-height-mm must start at 0");
  for (size_t i = 0; i < mm.size(); ++i) {
    if (i > 0 && !(mm[i] > mm[i - 1]))
      return fail(error, "image-height-mm must strictly increase");
    if (!(ri[i] > 0.0 && ri[i] <= 1.0))
      return fail(error, "relative-illumination must be in (0, 1]");
    if (i > 0 && ri[i] > ri[i - 1])
      return fail(error, "relative-illumination must never rise with radius");
  }
  if (!(c.pixel_size_um > 0.0))
    return fail(error, "pixel-size-um must be positive");
  if (!(c.centre_x >= 0.0 && c.centre_x <= 1.0 && c.centre_y >= 0.0 && c.centre_y <= 1.0))
    return fail(error, "centre-x and centre-y must be in [0, 1]");
  if (!(c.knee > 0.0 && c.knee < 255.0))
    return fail(error, "knee must be between 0 and 255, exclusive");
  if ((c.reference_width == 0) != (c.reference_height == 0))
    return fail(error, "set both reference-width and reference-height, or neither");
  if (c.reference_width < 0 || c.reference_height < 0)
    return fail(error, "reference size must be positive");
  return true;
}

bool vignette_config_load(const char *path, VignetteConfig *config, std::string *error)
{
  GKeyFile *file = g_key_file_new();
  GError *gerror = NULL;
  if (!g_key_file_load_from_file(file, path, G_KEY_FILE_NONE, &gerror)) {
    std::string message = std::string(path) + ": " + gerror->message;
    g_error_free(gerror);
    g_key_file_free(file);
    return fail(error, message);
  }

  VignetteConfig c;
  bool ok = true;
  gchar *mode = NULL;

  if (!g_key_file_has_group(file, GROUP)) {
    ok = fail(error, "missing [vignette] group");
  }

  if (ok) {
    gsize count = 0;
    gchar **keys = g_key_file_get_keys(file, GROUP, &count, NULL);
    for (gsize i = 0; ok && i < count; ++i) {
      bool known = false;
      for (const char *name : KNOWN_KEYS)
        known = known || strcmp(keys[i], name) == 0;
      if (!known)
        ok = fail(error, std::string("unknown key ") + keys[i]);
    }
    g_strfreev(keys);
  }

  for (const char *required : {"image-height-mm", "relative-illumination", "pixel-size-um"}) {
    if (ok && !g_key_file_has_key(file, GROUP, required, NULL))
      ok = fail(error, std::string("missing required key ") + required);
  }

  ok = ok && read_double_list(file, "image-height-mm", &c.image_height_mm, error);
  ok = ok && read_double_list(file, "relative-illumination", &c.relative_illumination, error);
  ok = ok && read_double(file, "pixel-size-um", &c.pixel_size_um, error);
  ok = ok && read_double(file, "centre-x", &c.centre_x, error);
  ok = ok && read_double(file, "centre-y", &c.centre_y, error);
  ok = ok && read_double(file, "knee", &c.knee, error);
  ok = ok && read_int(file, "reference-width", &c.reference_width, error);
  ok = ok && read_int(file, "reference-height", &c.reference_height, error);

  if (ok && g_key_file_has_key(file, GROUP, "highlight-mode", NULL)) {
    mode = g_key_file_get_string(file, GROUP, "highlight-mode", NULL);
    g_strstrip(mode);
    if (strcmp(mode, "knee") == 0)
      c.mode = VC_MODE_KNEE;
    else if (strcmp(mode, "clip") == 0)
      c.mode = VC_MODE_CLIP;
    else
      ok = fail(error, "highlight-mode must be knee or clip");
  }
  g_free(mode);
  g_key_file_free(file);

  ok = ok && validate(c, error);
  if (!ok) {
    if (error)
      *error = std::string(path) + ": " + *error;
    return false;
  }
  *config = c;
  return true;
}

void vignette_reference_size(const VignetteConfig &config, int width, int height,
                             int *ref_width, int *ref_height)
{
  *ref_width = config.reference_width > 0 ? config.reference_width : width;
  *ref_height = config.reference_height > 0 ? config.reference_height : height;
}

/* numpy.interp for one x: linear inside the chart, end values outside it. */
static double interp(double x, const std::vector<double> &xp, const std::vector<double> &fp)
{
  if (x <= xp.front())
    return fp.front();
  if (x >= xp.back())
    return fp.back();
  size_t j = 0;
  while (j + 2 < xp.size() && xp[j + 1] <= x)
    ++j;
  double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  return slope * (x - xp[j]) + fp[j];
}

std::vector<double> vignette_build_table(const VignetteConfig &config, int width, int height)
{
  int ref_w, ref_h;
  vignette_reference_size(config, width, height, &ref_w, &ref_h);
  double half_diagonal_mm =
      std::hypot((double) ref_h, (double) ref_w) / 2.0 * config.pixel_size_um / 1000.0;
  double step = (double) VC_MAX_RADIUS / (VC_TABLE_SIZE - 1);
  std::vector<double> table(VC_TABLE_SIZE);
  for (int i = 0; i < VC_TABLE_SIZE; ++i) {
    double r = i == VC_TABLE_SIZE - 1 ? (double) VC_MAX_RADIUS : i * step;
    table[i] = interp(r * half_diagonal_mm, config.image_height_mm, config.relative_illumination);
  }
  return table;
}

VcParams vignette_params(const VignetteConfig &config, int width, int height)
{
  int ref_w, ref_h;
  vignette_reference_size(config, width, height, &ref_w, &ref_h);
  VcParams p;
  p.centre_x_px = (float) (config.centre_x * width);
  p.centre_y_px = (float) (config.centre_y * height);
  p.half_diagonal_px = (float) (std::hypot((double) ref_h, (double) ref_w) / 2.0);
  p.knee = (float) config.knee;
  p.mode = config.mode;
  return p;
}
