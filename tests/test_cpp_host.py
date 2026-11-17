"""Checks the C++ config parser and per-pixel maths against the numpy reference.

Builds plugin/host_test, which runs the same functions as the CUDA kernel on the
CPU. Needs g++ and the GLib development files; skipped when they are missing.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vignettecorrect.config import VignetteConfig, read_config, write_config
from vignettecorrect.reference import apply_gain, lens_table

ROOT = Path(__file__).parent.parent
PLUGIN = ROOT / "plugin"
EXAMPLE = ROOT / "configs" / "kowa_lm3ncm_wp__bfs_pge_63s4c.txt"
MM = (0.0, 0.45, 0.9, 1.35, 1.8, 2.25, 2.7, 3.15, 3.6, 4.05, 4.3, 4.5)
RI = (1.00, 1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.88, 0.78, 0.60, 0.50, 0.34)


def _have_toolchain() -> bool:
    if not (shutil.which("g++") and shutil.which("make") and shutil.which("pkg-config")):
        return False
    return subprocess.run(["pkg-config", "--exists", "glib-2.0"]).returncode == 0


pytestmark = pytest.mark.skipif(not _have_toolchain(), reason="needs g++, make and GLib dev files")


@pytest.fixture(scope="module")
def host_test() -> Path:
    subprocess.run(["make", "-C", str(PLUGIN), "host_test"], check=True, capture_output=True)
    return PLUGIN / "host_test"


def cpp_table(host_test, config_path, h, w) -> np.ndarray:
    out = subprocess.run([str(host_test), "table", str(config_path), str(h), str(w)],
                         check=True, capture_output=True, text=True).stdout
    return np.array([float(line) for line in out.split()])


def cpp_apply(host_test, config_path, image, tmp_path) -> np.ndarray:
    h, w = image.shape[:2]
    src, dst = tmp_path / "in.rgb", tmp_path / "out.rgb"
    image.tofile(src)
    subprocess.run([str(host_test), "apply", str(config_path), str(h), str(w), str(src), str(dst)],
                   check=True, capture_output=True)
    return np.fromfile(dst, dtype=np.uint8).reshape(h, w, 3)


def assert_matches_reference(host_test, config, tmp_path, h, w, seed):
    path = tmp_path / "config.txt"
    write_config(config, path)
    image = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    expected = apply_gain(image, config.gain_map(h, w), knee=config.knee,
                          highlight_mode=config.highlight_mode)
    actual = cpp_apply(host_test, path, image, tmp_path)
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    # float32 table lookup against numpy's float64 one can flip a rounding.
    assert diff.max() <= 1
    assert (diff > 0).mean() < 0.005


def test_table_matches_lens_table_on_the_real_sensor(host_test):
    table = cpp_table(host_test, EXAMPLE, 2048, 3072)
    np.testing.assert_allclose(table, lens_table(MM, RI, 2.4, (2048, 3072)), rtol=0, atol=1e-12)


def test_table_uses_the_reference_size(host_test, tmp_path):
    config = VignetteConfig(MM, RI, pixel_size_um=3.45, reference_width=2048,
                            reference_height=1536)
    path = tmp_path / "config.txt"
    write_config(config, path)
    table = cpp_table(host_test, path, 1250, 2047)
    np.testing.assert_allclose(table, lens_table(MM, RI, 3.45, (1536, 2048)), rtol=0, atol=1e-12)


def test_example_config_parses_the_same_in_both_languages(host_test):
    config = read_config(EXAMPLE)
    table = cpp_table(host_test, EXAMPLE, 2048, 3072)
    expected = lens_table(config.image_height_mm, config.relative_illumination,
                          config.pixel_size_um, (2048, 3072))
    np.testing.assert_allclose(table, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("h, w", [(2048 // 8, 3072 // 8), (97, 131)])
def test_knee_mode_matches_reference(host_test, tmp_path, h, w):
    # A larger pixel size makes a small test frame span the whole chart.
    config = VignetteConfig(MM, RI, pixel_size_um=2.4 * 3072 / w)
    assert_matches_reference(host_test, config, tmp_path, h, w, seed=h)


def test_clip_mode_matches_reference(host_test, tmp_path):
    config = VignetteConfig(MM, RI, pixel_size_um=40.0, highlight_mode="clip")
    assert_matches_reference(host_test, config, tmp_path, 120, 180, seed=7)


def test_off_centre_lens_knee_and_reference_size(host_test, tmp_path):
    config = VignetteConfig(MM, RI, pixel_size_um=30.0, centre_x=0.45, centre_y=0.55,
                            knee=180.0, reference_width=240, reference_height=160)
    assert_matches_reference(host_test, config, tmp_path, 120, 180, seed=11)


@pytest.mark.parametrize("text", [
    "[other]\npixel-size-um=2.4\n",
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\n",
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\npixel-size-um=2.4\nstrength=1\n",
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=0.9;1\npixel-size-um=2.4\n",
    "[vignette]\nimage-height-mm=0.5;1\nrelative-illumination=1;0.9\npixel-size-um=2.4\n",
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\npixel-size-um=2.4\n"
    "highlight-mode=none\n",
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\npixel-size-um=2.4\nknee=255\n",
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\npixel-size-um=2.4\n"
    "reference-width=100\n",
])
def test_cpp_rejects_the_configs_python_rejects(host_test, tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    result = subprocess.run([str(host_test), "table", str(path), "10", "10"], capture_output=True)
    assert result.returncode == 1
    with pytest.raises(ValueError):
        read_config(path)


def test_cpp_rejects_a_missing_file(host_test, tmp_path):
    result = subprocess.run([str(host_test), "table", str(tmp_path / "nope.txt"), "10", "10"],
                            capture_output=True)
    assert result.returncode == 1
