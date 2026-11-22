"""
test_llm_reasoning.py
---------------------
Unit tests for Phase 3, Objective 18: LLM Mathematical Problem Solver.

Verifies:
- Valid mathematical problem input handling
- Empty / whitespace / None input rejection
- Successful LLM reasoning generation
- Handling of API failures and network errors
- Handling of rate-limit / quota errors (429 / RESOURCE_EXHAUSTED)
- Invalid provider selection rejection
- Exact preservation of raw, unparsed reasoning text

Uses 100% mocked API calls. Zero external API calls made during tests.
"""

import unittest
from unittest.mock import patch, MagicMock
import llm_reasoning


MOCK_REASONING_RESPONSE = """To find the total cost of the items purchased by the shopkeeper, we follow these steps:

Step 1: Calculate the total cost of the notebooks.
The shopkeeper buys 3 notebooks at ₹40 each.
Cost of notebooks = 3 * 40 = 120

Step 2: Calculate the total cost of the pens.
The shopkeeper buys 2 pens at ₹15 each.
Cost of pens = 2 * 15 = 30

Step 3: Calculate the total cost of all items.
Total cost = 120 + 30 = 150

Final Answer: 150"""


class TestLLMReasoning(unittest.TestCase):
    """Test suite for generate_reasoning in llm_reasoning.py."""

    @patch("llm_reasoning.call_llm")
    @patch("llm_reasoning.get_llm_client")
    def test_valid_input_and_successful_response_gemini(self, mock_get_client, mock_call_llm):
        """Test valid mathematical problem input successfully generates reasoning using Gemini."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, "gemini-3.1-flash-lite")
        mock_call_llm.return_value = MOCK_REASONING_RESPONSE

        problem = "A shopkeeper buys 3 notebooks for ₹40 each and 2 pens for ₹15 each. What is the total cost?"
        result = llm_reasoning.generate_reasoning(problem, provider="gemini")

        mock_get_client.assert_called_once_with("gemini")
        mock_call_llm.assert_called_once()
        args, kwargs = mock_call_llm.call_args
        self.assertEqual(args[0], mock_client)
        self.assertEqual(args[1], "gemini")
        self.assertEqual(args[2], "gemini-3.1-flash-lite")
        self.assertIn(problem, args[3])
        self.assertIn("Requirements:", args[3])
        self.assertIn("clearly numbered steps", args[3])
        self.assertEqual(result, MOCK_REASONING_RESPONSE)

    @patch("llm_reasoning.call_llm")
    @patch("llm_reasoning.get_llm_client")
    def test_valid_input_and_successful_response_groq(self, mock_get_client, mock_call_llm):
        """Test valid mathematical problem input successfully generates reasoning using Groq."""
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, "openai/gpt-oss-120b")
        mock_call_llm.return_value = MOCK_REASONING_RESPONSE

        problem = "What is 15 * 4 + 20 / 2?"
        result = llm_reasoning.generate_reasoning(problem, provider="groq")

        mock_get_client.assert_called_once_with("groq")
        self.assertEqual(result, MOCK_REASONING_RESPONSE)

    def test_empty_input(self):
        """Test that empty string, whitespace-only, or None input raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            llm_reasoning.generate_reasoning("")
        self.assertIn("empty", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            llm_reasoning.generate_reasoning("   \n\t  ")
        self.assertIn("empty", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            llm_reasoning.generate_reasoning(None)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_invalid_provider_selection(self):
        """Test that unsupported or invalid provider names raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            llm_reasoning.generate_reasoning("What is 5 + 5?", provider="openai_direct")
        self.assertIn("unsupported provider", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            llm_reasoning.generate_reasoning("What is 5 + 5?", provider="anthropic")
        self.assertIn("unsupported provider", str(ctx.exception).lower())

    @patch("llm_reasoning.call_llm")
    @patch("llm_reasoning.get_llm_client")
    def test_api_failure_handling(self, mock_get_client, mock_call_llm):
        """Test that backend API exceptions are caught and raised gracefully as RuntimeError."""
        mock_get_client.return_value = (MagicMock(), "gemini-3.1-flash-lite")
        mock_call_llm.side_effect = Exception("500 Internal Server Error: Remote host closed connection")

        with self.assertRaises(RuntimeError) as ctx:
            llm_reasoning.generate_reasoning("Solve x + 2 = 5", provider="gemini")
        self.assertIn("Inference error with LLM provider 'gemini'", str(ctx.exception))

    @patch("llm_reasoning.call_llm")
    @patch("llm_reasoning.get_llm_client")
    def test_rate_limit_handling(self, mock_get_client, mock_call_llm):
        """Test that rate-limit errors (429 / RESOURCE_EXHAUSTED) raise clear RuntimeError."""
        mock_get_client.return_value = (MagicMock(), "gemini-3.1-flash-lite")
        mock_call_llm.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for model")

        with self.assertRaises(RuntimeError) as ctx:
            llm_reasoning.generate_reasoning("Solve 2 * x = 10", provider="gemini")
        self.assertIn("Rate limit exceeded", str(ctx.exception))

    @patch("llm_reasoning.call_llm")
    @patch("llm_reasoning.get_llm_client")
    def test_empty_response_from_llm(self, mock_get_client, mock_call_llm):
        """Test that an empty or whitespace response from an LLM raises RuntimeError."""
        mock_get_client.return_value = (MagicMock(), "gemini-3.1-flash-lite")
        mock_call_llm.return_value = "   "

        with self.assertRaises(RuntimeError) as ctx:
            llm_reasoning.generate_reasoning("Solve 2 * x = 10", provider="gemini")
        self.assertIn("empty response", str(ctx.exception).lower())

    @patch("llm_reasoning.call_llm")
    @patch("llm_reasoning.get_llm_client")
    def test_preservation_of_raw_reasoning_response(self, mock_get_client, mock_call_llm):
        """Test that raw text is preserved completely without modification, truncation, or stripping."""
        complex_raw_response = (
            "   \n**Detailed Mathematical Derivation**\n\n"
            "Step 1: Let the unknown cost be $x$.\n"
            "Equation: 3 * 40 + 2 * 15 = x\n"
            "Step 2: 120 + 30 = 150\n"
            "Final Answer: 150\n\n   "
        )
        mock_get_client.return_value = (MagicMock(), "gemini-3.1-flash-lite")
        mock_call_llm.return_value = complex_raw_response

        res = llm_reasoning.generate_reasoning("Cost problem", provider="gemini")
        self.assertEqual(res, complex_raw_response)

    def test_client_injection_for_testing(self):
        """Test that pre-initialized client can be passed without initializing from .env."""
        mock_injected_client = MagicMock()
        mock_injected_response = "Step 1: 1+1=2\nFinal Answer: 2"

        with patch("llm_reasoning.call_llm", return_value=mock_injected_response) as mock_call:
            res = llm_reasoning.generate_reasoning(
                "1+1",
                provider="gemini",
                client=mock_injected_client,
                model_name="test-model",
            )
            self.assertEqual(res, mock_injected_response)
            mock_call.assert_called_once()
            self.assertEqual(mock_call.call_args[0][0], mock_injected_client)
            self.assertEqual(mock_call.call_args[0][2], "test-model")


if __name__ == "__main__":
    unittest.main()
