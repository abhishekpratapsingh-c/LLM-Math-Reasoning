"""
test_gemini_timeout.py
----------------------
Unit tests verifying the finite request timeout configuration for Gemini in main.py.

Tests:
1. Default 90-second timeout (90000 ms) when GEMINI_TIMEOUT_SECONDS is absent.
2. Custom integer timeout conversion (e.g. 45s -> 45000 ms).
3. Custom float timeout conversion (e.g. 12.5s -> 12500 ms).
4. Whitespace handling in numeric string.
5. Rejection of zero timeout (0 -> ValueError).
6. Rejection of negative timeout (-10 -> ValueError).
7. Rejection of non-numeric strings ('abc' -> ValueError).
8. Rejection of NaN values ('nan', 'NaN' -> ValueError).
9. Rejection of infinite values ('inf', '-inf', 'Infinity' -> ValueError).
10. Rejection of empty string timeout ('' -> ValueError).
11. Unchanged Groq client behavior and initialization.
12. Protection against leaking GEMINI_API_KEY in timeout error messages.
13. Preservation of custom GEMINI_MODEL configuration.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_llm_client


class TestGeminiTimeoutConfiguration(unittest.TestCase):
    """Test suite for Gemini HTTP request timeout configuration in get_llm_client()."""

    def setUp(self):
        # Clean environment baseline for each test
        self.patcher = patch.dict(os.environ, {}, clear=False)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch("google.genai.Client")
    def test_01_default_timeout_90_seconds(self, mock_client_cls):
        """Verifies default 90-second (90000 ms) timeout when GEMINI_TIMEOUT_SECONDS is absent."""
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345"}, clear=False):
            if "GEMINI_TIMEOUT_SECONDS" in os.environ:
                del os.environ["GEMINI_TIMEOUT_SECONDS"]

            client, model = get_llm_client("gemini")

            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 90000},
            )
            self.assertEqual(model, "gemini-3.1-flash-lite")
            self.assertEqual(client, mock_instance)

    @patch("google.genai.Client")
    def test_02_custom_integer_timeout_conversion(self, mock_client_cls):
        """Verifies custom integer timeout conversion (e.g. 45s -> 45000 ms)."""
        mock_client_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "45"}):
            client, model = get_llm_client("gemini")

            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 45000},
            )

    @patch("google.genai.Client")
    def test_03_custom_float_timeout_conversion(self, mock_client_cls):
        """Verifies custom float timeout conversion (e.g. 12.5s -> 12500 ms)."""
        mock_client_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "12.5"}):
            client, model = get_llm_client("gemini")

            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 12500},
            )

    @patch("google.genai.Client")
    def test_04_whitespace_in_timeout_string(self, mock_client_cls):
        """Verifies stripped whitespace in valid numeric string (e.g. '  60  ' -> 60000 ms)."""
        mock_client_cls.return_value = MagicMock()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "  60  "}):
            client, model = get_llm_client("gemini")

            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 60000},
            )

    def test_05_invalid_timeout_zero(self):
        """Verifies that timeout of 0 raises ValueError (must be strictly positive)."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "0"}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertIn("positive finite number", str(ctx.exception))

    def test_06_invalid_timeout_negative(self):
        """Verifies that negative timeout raises ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "-15"}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertIn("positive finite number", str(ctx.exception))

    def test_07_invalid_timeout_non_numeric(self):
        """Verifies that non-numeric string raises ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "invalid_number"}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertIn("positive finite number", str(ctx.exception))

    def test_08_invalid_timeout_nan(self):
        """Verifies that NaN values ('nan', 'NaN', 'NAN') raise ValueError."""
        for nan_val in ["nan", "NaN", "NAN"]:
            with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": nan_val}):
                with self.assertRaises(ValueError) as ctx:
                    get_llm_client("gemini")
                self.assertIn("positive finite number", str(ctx.exception))

    def test_09_invalid_timeout_infinite(self):
        """Verifies that infinite values ('inf', '-inf', 'Infinity', '-Infinity') raise ValueError."""
        for inf_val in ["inf", "-inf", "Infinity", "-Infinity", "+inf"]:
            with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": inf_val}):
                with self.assertRaises(ValueError) as ctx:
                    get_llm_client("gemini")
                self.assertIn("positive finite number", str(ctx.exception))

    def test_10_invalid_timeout_empty_string(self):
        """Verifies that an empty string raises ValueError."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": ""}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertIn("positive finite number", str(ctx.exception))

    @patch("openai.OpenAI")
    def test_11_groq_client_behavior_unchanged(self, mock_openai_cls):
        """Verifies that get_llm_client('groq') remains completely unchanged."""
        mock_groq_instance = MagicMock()
        mock_openai_cls.return_value = mock_groq_instance

        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-key-abc", "GROQ_MODEL": "custom-groq-model"}):
            client, model = get_llm_client("groq")

            mock_openai_cls.assert_called_once_with(
                api_key="groq-key-abc",
                base_url="https://api.groq.com/openai/v1",
            )
            self.assertEqual(model, "custom-groq-model")
            self.assertEqual(client, mock_groq_instance)

    def test_12_api_key_not_exposed_in_error_message(self):
        """Verifies that GEMINI_API_KEY is never exposed in exception strings."""
        secret_key = "TOP_SECRET_GEMINI_KEY_XYZ_98765"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret_key, "GEMINI_TIMEOUT_SECONDS": "-999"}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertNotIn(secret_key, str(ctx.exception))

    @patch("google.genai.Client")
    def test_13_custom_model_selection_preserved(self, mock_client_cls):
        """Verifies that GEMINI_MODEL override is respected alongside timeout configuration."""
        mock_client_cls.return_value = MagicMock()

        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "secret-key-12345",
            "GEMINI_MODEL": "gemini-1.5-pro-preview",
            "GEMINI_TIMEOUT_SECONDS": "60",
        }):
            client, model = get_llm_client("gemini")

            self.assertEqual(model, "gemini-1.5-pro-preview")
            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 60000},
            )

    @patch("google.genai.Client")
    def test_14_very_small_timeout_rounds_to_at_least_1ms(self, mock_client_cls):
        """Verifies that very small positive timeouts that would otherwise round to 0 ms result in at least 1 ms."""
        mock_client_cls.return_value = MagicMock()

        # 0.0001 seconds = 0.1 ms -> would round to 0 without safeguard
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "0.0001"}):
            client, model = get_llm_client("gemini")
            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 1},
            )

        mock_client_cls.reset_mock()
        # Even smaller positive timeout (0.00000001s = 0.00001 ms)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "0.00000001"}):
            client, model = get_llm_client("gemini")
            mock_client_cls.assert_called_once_with(
                api_key="secret-key-12345",
                http_options={"timeout": 1},
            )

    def test_15_very_large_finite_timeout_overflows_to_value_error(self):
        """Verifies that a very large finite timeout that overflows during seconds-to-milliseconds conversion raises ValueError."""
        # 1e308 is a finite float, but 1e308 * 1000 overflows double-precision float to infinity
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "1e308"}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertTrue(
                "not finite" in str(ctx.exception) or "cannot be represented safely" in str(ctx.exception),
                f"Unexpected exception message: {ctx.exception}"
            )

    def test_16_very_large_finite_timeout_exceeds_maxsize_raises_value_error(self):
        """Verifies that a very large finite timeout (e.g. 1e305 s) whose ms exceeds sys.maxsize raises ValueError."""
        # 1e305 * 1000 = 1e308 (finite float), but int(round(1e308)) > sys.maxsize
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key-12345", "GEMINI_TIMEOUT_SECONDS": "1e305"}):
            with self.assertRaises(ValueError) as ctx:
                get_llm_client("gemini")
            self.assertTrue(
                "exceeds maximum system integer size" in str(ctx.exception)
                or "cannot be represented safely" in str(ctx.exception),
                f"Unexpected exception message: {ctx.exception}"
            )


if __name__ == "__main__":
    unittest.main()
