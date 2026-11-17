/* vignettecorrect: in-place lens falloff correction on NVMM RGBA frames.
 *
 * Jetson only: frames are NvBufSurface surface-array memory, mapped to CUDA
 * through an EGL image. Corrects every frame in the buffer, so it works on a
 * single stream or on a batch after nvstreammux.
 *
 * The buffer's pixels are modified without taking a writable copy, so do not
 * place this element on one branch of a tee.
 *
 * Mapping a frame into CUDA costs more than correcting it, and upstream
 * elements recycle a small pool of buffers. So the mapping is kept on the
 * buffer's GstMemory (as qdata) and reused each time that buffer comes round.
 * GStreamer calls the qdata's destroy function when the memory is freed, just
 * before the surface itself is destroyed, and the mapping is released there.
 */
#include <string>
#include <vector>

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>

#include "nvbufsurface.h"

#include "vignette_config.h"
#include "vignette_kernel.h"

#ifndef PACKAGE
#define PACKAGE "gst-vignettecorrect"
#endif

GST_DEBUG_CATEGORY_STATIC(gst_vignettecorrect_debug);
#define GST_CAT_DEFAULT gst_vignettecorrect_debug

#define GST_TYPE_VIGNETTECORRECT (gst_vignettecorrect_get_type())
#define GST_VIGNETTECORRECT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_VIGNETTECORRECT, GstVignetteCorrect))

typedef struct _GstVignetteCorrect {
  GstBaseTransform parent;

  /* properties */
  gchar *config_file;
  gboolean enable;
  gboolean cache_mappings;

  /* state, owned by the streaming thread between start and stop */
  VignetteConfig *config;  /* loaded in start */
  VcKernel *kernel;        /* created in set_caps */
  VcParams params;
  gint width;
  gint height;

  /* statistics, reported at stop; times in microseconds */
  guint64 frames;
  guint64 mappings_made;
  gint64 time_mapping;  /* EGL map and CUDA registration */
  gint64 time_kernel;   /* the correction itself */
  gint64 time_total;    /* whole of transform_ip */
} GstVignetteCorrect;

typedef struct _GstVignetteCorrectClass {
  GstBaseTransformClass parent_class;
} GstVignetteCorrectClass;

GType gst_vignettecorrect_get_type(void);
G_DEFINE_TYPE(GstVignetteCorrect, gst_vignettecorrect, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_CONFIG_FILE, PROP_ENABLE, PROP_CACHE_MAPPINGS };

#define VIGNETTE_CAPS GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "RGBA")

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(VIGNETTE_CAPS));
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(VIGNETTE_CAPS));

/* ---- CUDA mappings kept on a GstMemory ------------------------------------ */

/* One frame of a batch, mapped into CUDA. */
struct FrameMapping {
  CUgraphicsResource resource = NULL;
  uint8_t *data = NULL;         /* device pointer to the RGBA plane */
  int pitch = 0;                /* bytes per row */
  bool egl_mapped_here = false; /* this element created the EGL image */
};

/* All frames of one NvBufSurface. Owns no reference to the element, because it
 * can outlive it: it is freed when the GstMemory is. */
struct MemoryMappings {
  NvBufSurface *surface;
  guint32 gpu_id;
  std::vector<FrameMapping> frames;
};

static GQuark mappings_quark(void)
{
  static gsize quark = 0;  /* g_once_init_enter needs a pointer-sized variable */
  if (g_once_init_enter(&quark))
    g_once_init_leave(&quark, (gsize) g_quark_from_static_string("vignettecorrect-mappings"));
  return (GQuark) quark;
}

/* Releases the mappings. `surface_alive` is false when the surface they were
 * made for has been replaced, in which case its EGL image cannot be unmapped. */
static void mappings_release(MemoryMappings *mappings, bool surface_alive)
{
  /* May run on any thread, so make the CUDA context current here. */
  cudaSetDevice(mappings->gpu_id);
  guint released = 0;
  for (size_t i = 0; i < mappings->frames.size(); ++i) {
    FrameMapping &frame = mappings->frames[i];
    if (frame.resource) {
      cuGraphicsUnregisterResource(frame.resource);
      ++released;
    }
    if (frame.egl_mapped_here && surface_alive)
      NvBufSurfaceUnMapEglImage(mappings->surface, (int) i);
  }
  GST_CAT_DEBUG(gst_vignettecorrect_debug, "released %u mapping(s) of surface %p", released,
                mappings->surface);
  delete mappings;
}

static void mappings_destroy_notify(gpointer data)
{
  mappings_release((MemoryMappings *) data, true);
}

/* Returns the mappings stored on `memory`, creating an empty set if needed. */
static MemoryMappings *mappings_for(GstMemory *memory, NvBufSurface *surface)
{
  GstMiniObject *object = GST_MINI_OBJECT_CAST(memory);
  MemoryMappings *mappings = (MemoryMappings *) gst_mini_object_get_qdata(object, mappings_quark());
  if (mappings && mappings->surface != surface) {
    /* The memory now holds a different surface: the old mappings are stale. */
    gst_mini_object_steal_qdata(object, mappings_quark());
    mappings_release(mappings, false);
    mappings = NULL;
  }
  if (!mappings) {
    mappings = new MemoryMappings();
    mappings->surface = surface;
    mappings->gpu_id = surface->gpuId;
    mappings->frames.resize(surface->batchSize);
    gst_mini_object_set_qdata(object, mappings_quark(), mappings, mappings_destroy_notify);
  }
  return mappings;
}

/* Maps frame `index` of the surface into CUDA. Returns an empty string on success. */
static std::string map_frame(NvBufSurface *surface, guint index, FrameMapping *mapping)
{
  NvBufSurfaceParams *frame = &surface->surfaceList[index];
  if (frame->mappedAddr.eglImage == NULL) {
    if (NvBufSurfaceMapEglImage(surface, index) != 0)
      return "NvBufSurfaceMapEglImage failed";
    mapping->egl_mapped_here = true;
  }

  CUeglFrame egl_frame;
  std::string error;
  CUresult status = cuGraphicsEGLRegisterImage(&mapping->resource, frame->mappedAddr.eglImage,
                                               CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    mapping->resource = NULL;
    error = "cuGraphicsEGLRegisterImage failed: " + std::to_string((int) status);
  } else {
    status = cuGraphicsResourceGetMappedEglFrame(&egl_frame, mapping->resource, 0, 0);
    if (status != CUDA_SUCCESS)
      error = "cuGraphicsResourceGetMappedEglFrame failed: " + std::to_string((int) status);
    else if (egl_frame.frameType != CU_EGL_FRAME_TYPE_PITCH || egl_frame.planeCount != 1)
      error = "unexpected EGL frame layout";
    else if (egl_frame.width != frame->width || egl_frame.height != frame->height)
      error = "EGL frame size does not match the surface";
  }

  if (error.empty()) {
    mapping->data = (uint8_t *) egl_frame.frame.pPitch[0];
    mapping->pitch = (int) egl_frame.pitch;
    return error;
  }
  if (mapping->resource)
    cuGraphicsUnregisterResource(mapping->resource);
  if (mapping->egl_mapped_here)
    NvBufSurfaceUnMapEglImage(surface, index);
  *mapping = FrameMapping();
  return error;
}

/* ---- element -------------------------------------------------------------- */

static void gst_vignettecorrect_set_property(GObject *object, guint id, const GValue *value,
                                             GParamSpec *pspec)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(object);
  switch (id) {
    case PROP_CONFIG_FILE:
      g_free(self->config_file);
      self->config_file = g_value_dup_string(value);
      break;
    case PROP_ENABLE:
      self->enable = g_value_get_boolean(value);
      break;
    case PROP_CACHE_MAPPINGS:
      self->cache_mappings = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
  }
}

static void gst_vignettecorrect_get_property(GObject *object, guint id, GValue *value,
                                             GParamSpec *pspec)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(object);
  switch (id) {
    case PROP_CONFIG_FILE:
      g_value_set_string(value, self->config_file);
      break;
    case PROP_ENABLE:
      g_value_set_boolean(value, self->enable);
      break;
    case PROP_CACHE_MAPPINGS:
      g_value_set_boolean(value, self->cache_mappings);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
  }
}

static void gst_vignettecorrect_finalize(GObject *object)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(object);
  g_free(self->config_file);
  G_OBJECT_CLASS(gst_vignettecorrect_parent_class)->finalize(object);
}

static gboolean gst_vignettecorrect_start(GstBaseTransform *trans)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(trans);
  if (!self->config_file) {
    GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS, ("config-file is not set"), (NULL));
    return FALSE;
  }
  std::string error;
  VignetteConfig *config = new VignetteConfig();
  if (!vignette_config_load(self->config_file, config, &error)) {
    delete config;
    GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS, ("invalid config: %s", error.c_str()), (NULL));
    return FALSE;
  }
  self->config = config;
  self->frames = self->mappings_made = 0;
  self->time_mapping = self->time_kernel = self->time_total = 0;
  return TRUE;
}

static gboolean gst_vignettecorrect_stop(GstBaseTransform *trans)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(trans);
  if (self->frames > 0) {
    double n = (double) self->frames * 1000.0;  /* microseconds -> milliseconds per frame */
    GST_INFO_OBJECT(self, "%" G_GUINT64_FORMAT " frames, %" G_GUINT64_FORMAT " mappings made; "
                    "per frame: mapping %.3f ms, kernel %.3f ms, total %.3f ms", self->frames,
                    self->mappings_made, self->time_mapping / n, self->time_kernel / n,
                    self->time_total / n);
  }
  vc_kernel_destroy(self->kernel);
  self->kernel = NULL;
  delete self->config;
  self->config = NULL;
  return TRUE;
}

static gboolean gst_vignettecorrect_set_caps(GstBaseTransform *trans, GstCaps *incaps,
                                             GstCaps *outcaps)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(trans);
  GstVideoInfo info;
  if (!gst_video_info_from_caps(&info, incaps)) {
    GST_ELEMENT_ERROR(self, STREAM, FORMAT, ("could not read the video caps"), (NULL));
    return FALSE;
  }
  self->width = GST_VIDEO_INFO_WIDTH(&info);
  self->height = GST_VIDEO_INFO_HEIGHT(&info);
  self->params = vignette_params(*self->config, self->width, self->height);

  std::vector<double> table = vignette_build_table(*self->config, self->width, self->height);
  std::vector<float> table_f(table.begin(), table.end());

  std::string error;
  vc_kernel_destroy(self->kernel);
  self->kernel = NULL;
  cudaError_t status = cudaSetDevice(0);
  if (status == cudaSuccess)
    self->kernel = vc_kernel_create(table_f.data(), &error);
  else
    error = cudaGetErrorString(status);
  if (!self->kernel) {
    GST_ELEMENT_ERROR(self, LIBRARY, INIT, ("CUDA set-up failed: %s", error.c_str()), (NULL));
    return FALSE;
  }
  /* Table entry (VC_TABLE_SIZE - 1) * 2 / 3 is normalised radius 1.0. */
  GST_INFO_OBJECT(self, "%dx%d, gain at r=1 %.3f, mode %s", self->width, self->height,
                  1.0 / table[(VC_TABLE_SIZE - 1) * 2 / 3],
                  self->params.mode == VC_MODE_KNEE ? "knee" : "clip");
  return TRUE;
}

/* Corrects frame `index` of the batch. Returns an empty string on success. */
static std::string correct_frame(GstVignetteCorrect *self, NvBufSurface *surface, guint index,
                                 FrameMapping *mapping)
{
  NvBufSurfaceParams *frame = &surface->surfaceList[index];
  if (frame->colorFormat != NVBUF_COLOR_FORMAT_RGBA)
    return "frame is not RGBA (NvBufSurface colour format " +
           std::to_string((int) frame->colorFormat) + ")";
  if ((gint) frame->width != self->width || (gint) frame->height != self->height)
    return "frame is " + std::to_string(frame->width) + "x" + std::to_string(frame->height) +
           ", caps said " + std::to_string(self->width) + "x" + std::to_string(self->height);

  std::string error;
  gint64 t0 = g_get_monotonic_time();
  if (!mapping->data) {
    error = map_frame(surface, index, mapping);
    if (!error.empty())
      return error;
    self->mappings_made += 1;
  }
  gint64 t1 = g_get_monotonic_time();
  vc_kernel_run(self->kernel, mapping->data, mapping->pitch, self->width, self->height,
                self->params, &error);
  gint64 t2 = g_get_monotonic_time();
  self->time_mapping += t1 - t0;
  self->time_kernel += t2 - t1;
  self->frames += 1;
  return error;
}

static GstFlowReturn gst_vignettecorrect_transform_ip(GstBaseTransform *trans, GstBuffer *buffer)
{
  GstVignetteCorrect *self = GST_VIGNETTECORRECT(trans);
  if (!self->enable)
    return GST_FLOW_OK;

  gint64 t_start = g_get_monotonic_time();
  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("could not map the input buffer"), (NULL));
    return GST_FLOW_ERROR;
  }
  NvBufSurface *surface = (NvBufSurface *) map.data;
  GstMemory *memory = gst_buffer_peek_memory(buffer, 0);

  std::string error;
  if (surface->memType != NVBUF_MEM_SURFACE_ARRAY && surface->memType != NVBUF_MEM_DEFAULT)
    error = "unsupported memory type " + std::to_string((int) surface->memType) +
            "; this element supports Jetson surface-array memory only";

  /* Makes the CUDA primary context current on this streaming thread. */
  if (error.empty()) {
    cudaError_t status = cudaSetDevice(surface->gpuId);
    if (status != cudaSuccess)
      error = std::string("cudaSetDevice: ") + cudaGetErrorString(status);
  }

  if (error.empty()) {
    MemoryMappings *mappings = mappings_for(memory, surface);
    if (surface->numFilled > mappings->frames.size())
      error = "batch has more frames than the surface was allocated for";
    for (guint i = 0; error.empty() && i < surface->numFilled; ++i)
      error = correct_frame(self, surface, i, &mappings->frames[i]);
    if (!self->cache_mappings) {
      /* Replacing the qdata runs its destroy function, releasing the mappings. */
      gst_mini_object_set_qdata(GST_MINI_OBJECT_CAST(memory), mappings_quark(), NULL, NULL);
    }
  }

  gst_buffer_unmap(buffer, &map);
  self->time_total += g_get_monotonic_time() - t_start;
  if (!error.empty()) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("%s", error.c_str()), (NULL));
    return GST_FLOW_ERROR;
  }
  return GST_FLOW_OK;
}

static void gst_vignettecorrect_class_init(GstVignetteCorrectClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  object_class->set_property = gst_vignettecorrect_set_property;
  object_class->get_property = gst_vignettecorrect_get_property;
  object_class->finalize = gst_vignettecorrect_finalize;

  g_object_class_install_property(object_class, PROP_CONFIG_FILE,
      g_param_spec_string("config-file", "Config file",
          "Path to this camera's correction config (GLib key-file)", NULL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                         GST_PARAM_MUTABLE_READY)));
  g_object_class_install_property(object_class, PROP_ENABLE,
      g_param_spec_boolean("enable", "Enable",
          "Correct frames; false passes them through untouched", TRUE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(object_class, PROP_CACHE_MAPPINGS,
      g_param_spec_boolean("cache-mappings", "Cache mappings",
          "Keep each pooled buffer mapped into CUDA between frames; false maps and "
          "unmaps every frame, which is slower", TRUE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(element_class, "Lens falloff correction",
      "Filter/Effect/Video", "Corrects lens vignetting on NVMM RGBA frames with CUDA",
      "gst-vignettecorrect");
  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  transform_class->start = GST_DEBUG_FUNCPTR(gst_vignettecorrect_start);
  transform_class->stop = GST_DEBUG_FUNCPTR(gst_vignettecorrect_stop);
  transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_vignettecorrect_set_caps);
  transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_vignettecorrect_transform_ip);
}

static void gst_vignettecorrect_init(GstVignetteCorrect *self)
{
  self->config_file = NULL;
  self->enable = TRUE;
  self->cache_mappings = TRUE;
  self->config = NULL;
  self->kernel = NULL;
  self->width = 0;
  self->height = 0;
  self->frames = self->mappings_made = 0;
  self->time_mapping = self->time_kernel = self->time_total = 0;

  GstBaseTransform *trans = GST_BASE_TRANSFORM(self);
  gst_base_transform_set_in_place(trans, TRUE);
  /* Caps are unchanged and NVMM buffers must not be copied, so run in
   * passthrough mode; transform_ip is still called for every buffer. */
  gst_base_transform_set_passthrough(trans, TRUE);
}

static gboolean plugin_init(GstPlugin *plugin)
{
  GST_DEBUG_CATEGORY_INIT(gst_vignettecorrect_debug, "vignettecorrect", 0,
                          "lens falloff correction");
  return gst_element_register(plugin, "vignettecorrect", GST_RANK_NONE,
                              GST_TYPE_VIGNETTECORRECT);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, vignettecorrect,
                  "Lens falloff (vignetting) correction for NVMM frames", plugin_init, "0.1.0",
                  "MIT/X11", PACKAGE, "https://github.com/")
