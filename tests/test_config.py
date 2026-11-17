from pathlib import Path

import numpy as np
import pytest

from vignettecorrect.config import VignetteConfig, read_config, write_config
from vignettecorrect.reference import gain_map, lens_table

EXAMPLE = Path(__file__).parent.parent / "configs" / "kowa_lm3ncm_wp__bfs_pge_63s4c.txt"
MM = (0.0, 1.0, 2.0)
RI = (1.0, 0.9, 0.5)


def test_example_config_loads_the_kowa_curve():
    config = read_config(EXAMPLE)
    assert config.image_height_mm[-1] == 4.5
    assert config.relative_illumination[-1] == 0.34
    assert len(config.image_height_mm) == len(config.relative_illumination) == 12
    assert config.pixel_size_um == 2.4
    assert (config.centre_x, config.centre_y) == (0.5, 0.5)
    assert config.highlight_mode == "knee"
    assert config.knee == 200.0
    assert config.reference_width is None


def test_example_config_gives_the_expected_corner_gain():
    gain = read_config(EXAMPLE).gain_map(2048, 3072)
    assert gain[0, 0] == pytest.approx(2.52, abs=0.02)


def test_round_trip(tmp_path):
    config = VignetteConfig(MM, RI, pixel_size_um=3.45, centre_x=0.48, centre_y=0.52,
                            highlight_mode="clip", knee=180.0,
                            reference_width=2048, reference_height=1536)
    path = tmp_path / "camera.txt"
    write_config(config, path)
    assert read_config(path) == config


def test_defaults_apply_when_optional_keys_are_absent(tmp_path):
    path = tmp_path / "minimal.txt"
    path.write_text("[vignette]\nimage-height-mm=0;1;2\n"
                    "relative-illumination=1;0.9;0.5\npixel-size-um=2.4\n")
    config = read_config(path)
    assert config == VignetteConfig(MM, RI, pixel_size_um=2.4)


def test_config_gain_map_matches_the_reference_functions():
    config = VignetteConfig(MM, RI, pixel_size_um=20.0, reference_width=200,
                            reference_height=100)
    table = lens_table(MM, RI, 20.0, (100, 200))
    expected = gain_map(table, 60, 80, reference_size=(100, 200))
    np.testing.assert_array_equal(config.gain_map(60, 80), expected)


@pytest.mark.parametrize("text", [
    "[other]\npixel-size-um=2.4\n",                                        # wrong group
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\n",      # no pixel size
    "[vignette]\nimage-height-mm=0;1\nrelative-illumination=1;0.9\n"
    "pixel-size-um=2.4\nstrength=0.6\n",                                   # unknown key
])
def test_read_rejects_bad_files(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_config(path)


@pytest.mark.parametrize("kwargs", [
    dict(image_height_mm=(0.0, 1.0), relative_illumination=(1.0, 0.9, 0.8)),  # lengths
    dict(image_height_mm=(0.5, 1.0), relative_illumination=(1.0, 0.9)),       # not from 0
    dict(image_height_mm=(0.0, 1.0, 1.0), relative_illumination=(1.0, 0.9, 0.8)),
    dict(image_height_mm=(0.0, 1.0), relative_illumination=(0.9, 1.0)),       # rises
    dict(image_height_mm=(0.0, 1.0), relative_illumination=(1.0, 0.0)),       # zero
    dict(image_height_mm=MM, relative_illumination=RI, pixel_size_um=0.0),
    dict(image_height_mm=MM, relative_illumination=RI, centre_x=1.5),
    dict(image_height_mm=MM, relative_illumination=RI, highlight_mode="none"),
    dict(image_height_mm=MM, relative_illumination=RI, knee=255.0),
    dict(image_height_mm=MM, relative_illumination=RI, reference_width=100),
])
def test_invalid_settings_are_rejected(kwargs):
    kwargs.setdefault("pixel_size_um", 2.4)
    with pytest.raises(ValueError):
        VignetteConfig(**kwargs)
