# Builds the vignettecorrect GStreamer element on a Jetson with DeepStream.
#
#   make             builds libgstvignettecorrect.so
#   make host_test   builds the CPU-only test tool (needs GLib only; any machine)
#
# To try the element without installing it:
#   GST_PLUGIN_PATH=$PWD gst-inspect-1.0 vignettecorrect

CUDA_DIR ?= /usr/local/cuda
DS_DIR ?= /opt/nvidia/deepstream/deepstream
TEGRA_LIB_DIR ?= /usr/lib/aarch64-linux-gnu/tegra
CXX ?= g++
NVCC ?= $(CUDA_DIR)/bin/nvcc

LIB := libgstvignettecorrect.so
GST_PKGS := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

# -ffp-contract=off and --fmad=false keep the float maths free of fused
# multiply-adds, so results match the numpy reference as closely as possible.
CXXFLAGS += -O2 -fPIC -Wall -std=c++14 -ffp-contract=off
NVCCFLAGS += -O2 --fmad=false -Xcompiler -fPIC

GST_CFLAGS = $(shell pkg-config --cflags $(GST_PKGS)) -I$(CUDA_DIR)/include -I$(DS_DIR)/sources/includes
GST_LIBS = $(shell pkg-config --libs $(GST_PKGS)) \
	-L$(CUDA_DIR)/lib64 -lcudart -L$(TEGRA_LIB_DIR) -lcuda \
	-L$(DS_DIR)/lib -lnvbufsurface -Wl,-rpath,$(DS_DIR)/lib

HEADERS := vignette_math.h vignette_config.h vignette_kernel.h

all: $(LIB)

vignette_config.o: vignette_config.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(shell pkg-config --cflags glib-2.0) -c -o $@ $<

gstvignettecorrect.o: gstvignettecorrect.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(GST_CFLAGS) -c -o $@ $<

vignette_kernel.o: vignette_kernel.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(LIB): gstvignettecorrect.o vignette_config.o vignette_kernel.o
	$(CXX) -shared -Wl,--no-undefined -o $@ $^ $(GST_LIBS)

host_test: host_test.cpp vignette_config.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(shell pkg-config --cflags glib-2.0) -o $@ host_test.cpp vignette_config.cpp \
		$(shell pkg-config --libs glib-2.0) -lm

clean:
	rm -f *.o $(LIB) host_test

.PHONY: all clean
