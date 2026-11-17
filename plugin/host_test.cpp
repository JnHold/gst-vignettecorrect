/* Host-side test tool: runs the config parser and the per-pixel maths on the
 * CPU, with no GPU or GStreamer, so pytest can compare them with the numpy
 * reference.
 *
 *   host_test table <config> <height> <width>
 *       prints the VC_TABLE_SIZE falloff values, one per line
 *   host_test apply <config> <height> <width> <in.rgb> <out.rgb>
 *       corrects a raw packed (H, W, 3) uint8 image
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vignette_config.h"

static int usage()
{
  fprintf(stderr, "usage: host_test table <config> <height> <width>\n"
                  "       host_test apply <config> <height> <width> <in.rgb> <out.rgb>\n");
  return 2;
}

int main(int argc, char **argv)
{
  if (argc < 5)
    return usage();
  const std::string command = argv[1];
  int height = atoi(argv[3]);
  int width = atoi(argv[4]);
  if (height <= 0 || width <= 0)
    return usage();

  VignetteConfig config;
  std::string error;
  if (!vignette_config_load(argv[2], &config, &error)) {
    fprintf(stderr, "config error: %s\n", error.c_str());
    return 1;
  }
  std::vector<double> table = vignette_build_table(config, width, height);

  if (command == "table" && argc == 5) {
    for (double value : table)
      printf("%.17g\n", value);
    return 0;
  }

  if (command == "apply" && argc == 7) {
    std::vector<float> table_f(table.begin(), table.end());
    VcParams params = vignette_params(config, width, height);
    std::vector<uint8_t> image((size_t) height * width * 3);
    FILE *in = fopen(argv[5], "rb");
    if (!in || fread(image.data(), 1, image.size(), in) != image.size()) {
      fprintf(stderr, "could not read %zu bytes from %s\n", image.size(), argv[5]);
      return 1;
    }
    fclose(in);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        uint8_t *px = &image[((size_t) y * width + x) * 3];
        uint32_t alpha = (uint32_t) ((x + y) & 0xFF) << 24;  /* must come back unchanged */
        uint32_t word = alpha | px[0] | ((uint32_t) px[1] << 8) | ((uint32_t) px[2] << 16);
        uint32_t out = vc_correct_word(word, vc_gain(table_f.data(), &params, x, y), &params);
        if ((out & 0xFF000000u) != alpha) {
          fprintf(stderr, "alpha byte changed at %d,%d\n", x, y);
          return 1;
        }
        px[0] = out & 0xFF;
        px[1] = (out >> 8) & 0xFF;
        px[2] = (out >> 16) & 0xFF;
      }
    }
    FILE *out = fopen(argv[6], "wb");
    if (!out || fwrite(image.data(), 1, image.size(), out) != image.size()) {
      fprintf(stderr, "could not write %s\n", argv[6]);
      return 1;
    }
    fclose(out);
    return 0;
  }
  return usage();
}
