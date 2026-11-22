"""
test_step_extractor.py
----------------------
Unit tests for Phase 3, Objective 19: Extract Intermediate Mathematical Steps.

Verifies:
- Standard numbered steps (Step 1:, Step 2:)
- Markdown bold step headings (**Step 1:**, **Step 1: Title**)
- Alternative numbering formats (1., 2. and 1), 2))
- Final answer extraction and separation
- Multiple equations within a single step
- Missing final answer handling
- Empty / whitespace / None response handling
- Malformed numbering handling (skipped or non-sequential steps)
- Preservation of original mathematical text, formatting, and step order
- Strict refusal to silently invent missing reasoning steps

Uses 100% mocked / static text inputs. Zero external API calls.
"""

import unittest
from step_extractor import extract_reasoning_steps, ReasoningStep, ExtractedReasoning


class TestStepExtractor(unittest.TestCase):
    """Test suite for extract_reasoning_steps."""

    def test_standard_numbered_steps(self):
        """Test extraction with standard 'Step 1:', 'Step 2:', etc."""
        text = (
            "Step 1: Calculate notebook cost\n"
            "3 * 40 = 120\n\n"
            "Step 2: Calculate pen cost\n"
            "2 * 15 = 30\n\n"
            "Step 3: Sum total costs\n"
            "120 + 30 = 150\n\n"
            "Final Answer: 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.steps[0].step_number, 1)
        self.assertEqual(res.steps[1].step_number, 2)
        self.assertEqual(res.steps[2].step_number, 3)
        self.assertIn("3 * 40 = 120", res.steps[0].step_content)
        self.assertIn("2 * 15 = 30", res.steps[1].step_content)
        self.assertIn("120 + 30 = 150", res.steps[2].step_content)
        self.assertEqual(res.final_answer, "150")
        self.assertEqual(res.final_answer_raw, "Final Answer: 150")

    def test_markdown_bold_step_headings(self):
        """Test extraction with Markdown bold headings e.g. **Step 1: ...**."""
        text = (
            "Here is the solution:\n\n"
            "**Step 1: Calculate notebook total.**\n"
            "Cost = 3 * 40 = 120\n\n"
            "**Step 2: Calculate pen total.**\n"
            "Cost = 2 * 15 = 30\n\n"
            "**Step 3: Combine totals.**\n"
            "Total = 120 + 30 = 150\n\n"
            "**Final Answer:** 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.steps[0].step_header, "**Step 1: Calculate notebook total.**")
        self.assertEqual(res.steps[1].step_header, "**Step 2: Calculate pen total.**")
        self.assertEqual(res.steps[2].step_header, "**Step 3: Combine totals.**")
        self.assertEqual(res.final_answer, "150")
        self.assertEqual(res.final_answer_raw, "**Final Answer:** 150")

    def test_alternative_numbering_period_format(self):
        """Test extraction with alternative '1.', '2.' list numbering."""
        text = (
            "1. Multiply 3 by 40 to get 120.\n"
            "2. Multiply 2 by 15 to get 30.\n"
            "3. Add 120 and 30 to get 150.\n"
            "The final answer is: 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.steps[0].step_number, 1)
        self.assertEqual(res.steps[1].step_number, 2)
        self.assertEqual(res.steps[2].step_number, 3)
        self.assertEqual(res.final_answer, "150")

    def test_alternative_numbering_parenthesis_format(self):
        """Test extraction with alternative '1)', '2)' numbering format."""
        text = (
            "1) Cost of notebooks: 3 * 40 = 120\n"
            "2) Cost of pens: 2 * 15 = 30\n"
            "3) Total: 120 + 30 = 150\n"
            "Final Answer: 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 3)
        self.assertEqual(res.steps[0].step_number, 1)
        self.assertEqual(res.steps[1].step_number, 2)
        self.assertEqual(res.steps[2].step_number, 3)
        self.assertEqual(res.final_answer, "150")

    def test_multiple_equations_within_single_step(self):
        """Test that steps containing multiple intermediate calculations preserve all equations."""
        text = (
            "Step 1: Complete notebook calculations.\n"
            "Price per unit = 40\n"
            "Quantity = 3\n"
            "Subtotal = 3 * 40\n"
            "Subtotal = 120\n\n"
            "Step 2: Pen calculations.\n"
            "Subtotal = 2 * 15\n"
            "Subtotal = 30\n\n"
            "Final Answer: 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(len(res.steps), 2)
        step1_text = res.steps[0].text
        self.assertIn("Price per unit = 40", step1_text)
        self.assertIn("Quantity = 3", step1_text)
        self.assertIn("Subtotal = 3 * 40", step1_text)
        self.assertIn("Subtotal = 120", step1_text)

    def test_missing_final_answer(self):
        """Test extraction when the response contains steps but no explicit final answer."""
        text = (
            "Step 1: 3 * 40 = 120\n\n"
            "Step 2: 2 * 15 = 30\n\n"
            "Step 3: 120 + 30 = 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.steps), 3)
        self.assertIsNone(res.final_answer)
        self.assertIsNone(res.final_answer_raw)

    def test_empty_response(self):
        """Test graceful handling of empty string, whitespace-only, and None inputs."""
        res_empty = extract_reasoning_steps("")
        self.assertEqual(res_empty.status, "empty_response")
        self.assertEqual(res_empty.steps, [])
        self.assertIsNone(res_empty.final_answer)

        res_ws = extract_reasoning_steps("   \n\t   ")
        self.assertEqual(res_ws.status, "empty_response")
        self.assertEqual(res_ws.steps, [])

        res_none = extract_reasoning_steps(None)
        self.assertEqual(res_none.status, "empty_response")
        self.assertEqual(res_none.steps, [])

    def test_no_steps_found(self):
        """Test graceful handling when response is text without numbered steps."""
        text = "The total cost can be computed by multiplying 3 by 40 and adding 2 times 15, yielding 150."
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "no_steps_found")
        self.assertEqual(res.steps, [])
        self.assertIsNone(res.final_answer)

    def test_malformed_numbering_skipped_step(self):
        """Test that skipped step numbers (e.g., Step 1 then Step 3) are detected and never invented."""
        text = (
            "Step 1: Compute notebooks\n"
            "3 * 40 = 120\n\n"
            "Step 3: Final combination\n"
            "120 + 30 = 150\n\n"
            "Final Answer: 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "malformed")
        self.assertFalse(res.is_sequential)
        self.assertEqual(res.missing_step_numbers, [2])
        # Crucial check: never silently invent step 2
        self.assertEqual(len(res.steps), 2)
        self.assertEqual(res.steps[0].step_number, 1)
        self.assertEqual(res.steps[1].step_number, 3)

    def test_malformed_numbering_out_of_order(self):
        """Test that out-of-order steps preserve chronological extraction order and flag malformation."""
        text = (
            "Step 2: Pens\n"
            "2 * 15 = 30\n\n"
            "Step 1: Notebooks\n"
            "3 * 40 = 120\n\n"
            "Final Answer: 150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(res.status, "malformed")
        self.assertFalse(res.is_sequential)
        self.assertEqual(len(res.steps), 2)
        # Order is preserved as written in the text
        self.assertEqual(res.steps[0].step_number, 2)
        self.assertEqual(res.steps[1].step_number, 1)

    def test_preservation_of_mathematical_text_and_special_symbols(self):
        """Test that currency symbols (₹, $, €), operators (×, ÷), and whitespace are exactly preserved."""
        text = (
            "**Step 1: Calculate notebook cost**\n"
            "Notebooks = 3 × ₹40 = ₹120\n\n"
            "**Step 2: Calculate pen cost**\n"
            "Pens = 2 × ₹15 = ₹30\n\n"
            "Final Answer: ₹150"
        )
        res = extract_reasoning_steps(text)
        self.assertEqual(len(res.steps), 2)
        self.assertIn("3 × ₹40 = ₹120", res.steps[0].text)
        self.assertIn("2 × ₹15 = ₹30", res.steps[1].text)
        self.assertIn("₹150", res.final_answer)

    def test_actual_gemini_objective_18_output(self):
        """Test extraction directly against the actual Gemini output from Objective 18."""
        actual_gemini_output = (
            "To find the total cost of the items purchased by the shopkeeper, we follow these steps:\n\n"
            "**Step 1: Calculate the total cost of the notebooks.**\n"
            "The shopkeeper buys 3 notebooks at ₹40 each.\n"
            "Cost of notebooks = Number of notebooks × Cost per notebook\n"
            "Cost of notebooks = 3 × ₹40\n"
            "Cost of notebooks = ₹120\n\n"
            "**Step 2: Calculate the total cost of the pens.**\n"
            "The shopkeeper buys 2 pens at ₹15 each.\n"
            "Cost of pens = Number of pens × Cost per pen\n"
            "Cost of pens = 2 × ₹15\n"
            "Cost of pens = ₹30\n\n"
            "**Step 3: Calculate the total cost of all items.**\n"
            "Total cost = Cost of notebooks + Cost of pens\n"
            "Total cost = ₹120 + ₹30\n"
            "Total cost = ₹150\n\n"
            "Final Answer: 150"
        )
        res = extract_reasoning_steps(actual_gemini_output)
        self.assertEqual(res.status, "success")
        self.assertTrue(res.is_sequential)
        self.assertEqual(len(res.steps), 3)

        # Verify step headers
        self.assertEqual(res.steps[0].step_header, "**Step 1: Calculate the total cost of the notebooks.**")
        self.assertEqual(res.steps[1].step_header, "**Step 2: Calculate the total cost of the pens.**")
        self.assertEqual(res.steps[2].step_header, "**Step 3: Calculate the total cost of all items.**")

        # Verify step contents
        self.assertIn("Cost of notebooks = 3 × ₹40", res.steps[0].step_content)
        self.assertIn("Cost of pens = 2 × ₹15", res.steps[1].step_content)
        self.assertIn("Total cost = ₹120 + ₹30", res.steps[2].step_content)

        # Verify separated final answer
        self.assertEqual(res.final_answer, "150")
        self.assertEqual(res.final_answer_raw, "Final Answer: 150")

        # Verify final answer is not included in step 3's body
        self.assertNotIn("Final Answer", res.steps[2].text)


if __name__ == "__main__":
    unittest.main()
