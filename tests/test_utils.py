"""Tests for app.core.utils helpers."""

from importlib.metadata import version
from unittest.mock import patch

import pytest

from app.core.utils import model_fingerprint, quantization_enabled


class TestQuantizationEnabled:
    @pytest.mark.parametrize(
        "device, quantize_cpu, expected",
        [("cpu", True, True), ("cpu", False, False), ("cuda", True, False), ("cuda", False, False)],
    )
    def test_only_on_cpu_when_setting_enabled(self, device, quantize_cpu, expected):
        with patch("app.core.utils.settings.quantize_cpu", quantize_cpu):
            assert quantization_enabled(device) is expected


class TestModelFingerprint:
    """The cache key fingerprint must change whenever tagging output may change."""

    def test_contains_library_and_model_versions(self):
        from pie_extended.cli.utils import get_model

        with patch("app.core.utils.get_device", return_value="cuda"):
            fingerprint = model_fingerprint("freem")

        assert version("pie-extended") in fingerprint
        assert version("PaPie") in fingerprint
        assert str(get_model("freem").VERSION) in fingerprint
        assert "cuda" in fingerprint

    def test_changes_with_cpu_quantization(self):
        with patch("app.core.utils.get_device", return_value="cpu"):
            with patch("app.core.utils.settings.quantize_cpu", False):
                float_fingerprint = model_fingerprint("freem")
            with patch("app.core.utils.settings.quantize_cpu", True):
                int8_fingerprint = model_fingerprint("freem")

        assert float_fingerprint != int8_fingerprint

    def test_differs_between_models(self):
        with patch("app.core.utils.get_device", return_value="cpu"):
            assert model_fingerprint("freem") != model_fingerprint("lasla")

    def test_unknown_module_does_not_raise(self):
        with patch("app.core.utils.get_device", return_value="cpu"):
            assert isinstance(model_fingerprint("not_a_pie_module"), str)
