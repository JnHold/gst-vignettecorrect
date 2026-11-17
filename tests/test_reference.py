import numpy as np
import pytest

from vignettecorrect.reference import MAX_RADIUS, TABLE_SIZE, apply_gain, gain_map, lens_table

# Kowa LM3NCM-WP (3.5 mm F2.4) relative illumination, 587.56 nm.
KOWA_MM = (0.0, 0.45, 0.9, 1.35, 1.8, 2.25, 2.7, 3.15, 3.6, 4.05, 4.3, 4.5)
KOWA_RI = (1.00, 1.00, 0.99, 0.98, 0.97, 0.96, 0.94, 0.88, 0.78, 0.60, 0.50, 0.34)
H, W = 2048, 3072  # FLIR BFS-PGE-63S4C-C
PIXEL_UM = 2.4


def kowa_table(size=(H, W)):
    return lens_table(KOWA_MM, KOWA_RI, PIXEL_UM, size)


def falloff_at(table, r):
    return float(np.interp(r, np.linspace(0.0, MAX_RADIUS, len(table)), table))


def pixel(value, gain, **kwargs):
    """Correct one grey pixel and return its value."""
    image = np.full((1, 1, 3), value, dtype=np.uint8)
    return int(apply_gain(image, np.full((1, 1), gain, dtype=np.float32), **kwargs)[0, 0, 0])


# --- curve -----------------------------------------------------------------

def test_table_shape_and_centre():
    table = kowa_table()
    assert table.shape == (TABLE_SIZE,)
    assert table[0] == 1.0


@pytest.mark.parametrize("r, brightness, gain", [
    (0.5, 0.96, 1.04),
    (0.7, 0.89, 1.13),
    (0.85, 0.71, 1.40),
    (1.0, 0.40, 2.53),
])
def test_curve_reproduces_published_values_on_its_sensor(r, brightness, gain):
    falloff = falloff_at(kowa_table(), r)
    assert falloff == pytest.approx(brightness, abs=0.006)
    assert 1.0 / falloff == pytest.approx(gain, abs=0.02)


def test_left_and_right_edges():
    gain = gain_map(kowa_table(), H, W)
    assert 1.0 / gain[H // 2, 0] == pytest.approx(0.75, abs=0.006)
    assert 1.0 / gain[H // 2, W - 1] == pytest.approx(0.75, abs=0.006)


def test_falloff_never_rises_with_radius():
    assert np.all(np.diff(kowa_table()) <= 0)


def test_table_holds_last_value_past_the_chart():
    assert kowa_table()[-1] == KOWA_RI[-1]


# --- gain map --------------------------------------------------------------

def test_gain_map_shape_dtype_and_floor():
    gain = gain_map(kowa_table(), H, W)
    assert gain.shape == (H, W)
    assert gain.dtype == np.float32
    assert gain.min() >= 1.0


def test_gain_map_is_symmetric_about_the_centre():
    gain = gain_map(kowa_table((40, 60)), 40, 60)
    np.testing.assert_allclose(gain, gain[::-1, :], rtol=1e-6)
    np.testing.assert_allclose(gain, gain[:, ::-1], rtol=1e-6)


def test_gain_map_peaks_in_the_corners():
    gain = gain_map(kowa_table(), H, W)
    assert gain.max() == gain[0, 0]
    assert gain[0, 0] == pytest.approx(2.52, abs=0.02)


def test_cropped_frame_uses_its_reference_size():
    table = kowa_table()
    full = gain_map(table, H, W)
    ch, cw = 1000, 1600  # centred crop
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    crop = gain_map(table, ch, cw, reference_size=(H, W))
    np.testing.assert_allclose(crop, full[y0:y0 + ch, x0:x0 + cw], rtol=1e-5)
    assert crop.max() < full.max()


def test_off_centre_lens_moves_the_minimum():
    gain = gain_map(kowa_table((100, 100)), 100, 100, cx=0.25, cy=0.75)
    y, x = np.unravel_index(np.argmin(gain), gain.shape)
    assert gain[75, 25] == gain[y, x]


# --- highlight knee --------------------------------------------------------

@pytest.mark.parametrize("original, gain, expected", [
    (40, 1.0, 40), (40, 1.13, 45), (40, 1.34, 54), (40, 2.5, 100),
    (80, 1.0, 80), (80, 1.13, 90), (80, 1.34, 107), (80, 2.5, 200),
    (120, 1.0, 120), (120, 1.13, 136), (120, 1.34, 161), (120, 2.5, 213),
    (160, 1.0, 160), (160, 1.13, 181), (160, 1.34, 206), (160, 2.5, 225),
    (200, 1.0, 200), (200, 1.13, 216), (200, 1.34, 226), (200, 2.5, 238),
    (230, 1.0, 230), (230, 1.13, 237), (230, 1.34, 242), (230, 2.5, 247),
    (255, 1.0, 255), (255, 1.13, 255), (255, 1.34, 255), (255, 2.5, 255),
])
def test_knee_matches_the_published_table(original, gain, expected):
    assert pixel(original, gain) == expected


def test_gain_one_is_the_identity():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    out = apply_gain(image, np.ones((32, 48), dtype=np.float32))
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize("gain", [1.0, 1.13, 1.34, 2.0, 2.53])
def test_knee_output_is_monotonic_and_never_darker(gain):
    values = [pixel(v, gain) for v in range(256)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert all(out >= v for v, out in enumerate(values))
    assert values[0] == 0 and values[255] == 255


def test_knee_adds_no_clipping():
    rng = np.random.default_rng(1)
    # The knee maps 254 to 254.7 at this gain, which rounds to 255, so "no added
    # clipping" holds for originals up to 253.
    image = rng.integers(0, 254, size=(64, 64, 3), dtype=np.uint8)
    out = apply_gain(image, np.full((64, 64), 2.53, dtype=np.float32))
    assert out.max() < 255


def test_dark_pixels_get_full_gain():
    assert pixel(20, 2.5) == 50


def test_channels_scale_together():
    image = np.array([[[30, 60, 90]]], dtype=np.uint8)
    out = apply_gain(image, np.full((1, 1), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(out, [[[60, 120, 180]]])
    # Above the knee the ratios are kept too, to rounding.
    bright = np.array([[[60, 120, 180]]], dtype=np.uint8)
    out = apply_gain(bright, np.full((1, 1), 2.0, dtype=np.float32)).astype(np.float64)
    assert out[0, 0, 0] / out[0, 0, 2] == pytest.approx(60 / 180, abs=0.01)
    assert out[0, 0, 1] / out[0, 0, 2] == pytest.approx(120 / 180, abs=0.01)


def test_correcting_synthetic_falloff_recovers_the_flat_image():
    h, w = 120, 180
    gain = gain_map(kowa_table((h, w)), h, w)
    flat = np.full((h, w, 3), 70.0)  # stays below the knee at every gain
    vignetted = np.round(flat / gain[:, :, None]).astype(np.uint8)
    out = apply_gain(vignetted, gain).astype(np.float64)
    assert np.abs(out - 70.0).max() <= 2.0  # 8-bit rounding of the dark corners


# --- clip mode -------------------------------------------------------------

def test_clip_mode_is_the_pure_map():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    gain = gain_map(kowa_table((32, 48)), 32, 48)
    out = apply_gain(image, gain, highlight_mode="clip")
    expected = np.clip(np.round(image.astype(np.float32) * gain[:, :, None]), 0, 255)
    np.testing.assert_array_equal(out, expected.astype(np.uint8))


def test_clip_mode_clips_where_the_knee_does_not():
    assert pixel(160, 2.5, highlight_mode="clip") == 255
    assert pixel(160, 2.5, highlight_mode="knee") == 225


def test_modes_agree_below_the_knee():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 70, size=(32, 48, 3), dtype=np.uint8)  # 70 * 2.53 < 200
    gain = gain_map(kowa_table((32, 48)), 32, 48)
    np.testing.assert_array_equal(apply_gain(image, gain, highlight_mode="knee"),
                                  apply_gain(image, gain, highlight_mode="clip"))


# --- input checks ----------------------------------------------------------

def test_rejects_wrong_image_type_and_shape():
    gain = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_gain(np.zeros((4, 4, 3), dtype=np.float32), gain)
    with pytest.raises(ValueError):
        apply_gain(np.zeros((4, 4, 4), dtype=np.uint8), gain)
    with pytest.raises(ValueError):
        apply_gain(np.zeros((4, 5, 3), dtype=np.uint8), gain)
    with pytest.raises(ValueError):
        apply_gain(np.zeros((4, 4, 3), dtype=np.uint8), gain, highlight_mode="none")
