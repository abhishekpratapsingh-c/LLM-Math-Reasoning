import os
import re
import pandas as pd


def extract_ground_truth(answer_str: str) -> str:
    """
    Extracts the ground truth numerical answer from GSM8K's answer column.
    In GSM8K, the final answer always appears after '####'.
    """
    if not isinstance(answer_str, str):
        return ""
    if "####" in answer_str:
        raw = answer_str.split("####")[-1].strip()
        cleaned = raw.replace(",", "").replace("$", "").strip()
        return cleaned
    return str(answer_str).replace(",", "").replace("$", "").strip()


def find_boxed_contents(text: str) -> list[str]:
    """
    Finds the full content of all \\boxed{...} expressions in text,
    properly accounting for nested curly braces (e.g. \\boxed{\$448{,}000}).
    """
    results = []
    idx = 0
    target = r"\boxed{"
    while True:
        pos = text.find(target, idx)
        if pos == -1:
            break
        start = pos + len(target)
        depth = 1
        curr = start
        while curr < len(text) and depth > 0:
            if text[curr] == "{":
                depth += 1
            elif text[curr] == "}":
                depth -= 1
            curr += 1
        if depth == 0:
            results.append(text[start : curr - 1])
            idx = curr
        else:
            idx = start
    return results


def clean_number(num_str: str) -> str:
    """
    Cleans a matched number string by removing commas, currency symbols, and whitespace.
    e.g., '$448,000' -> '448000', '1,250' -> '1250', '72' -> '72'
    """
    if not num_str:
        return ""
    return (
        num_str.replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .strip()
    )


def extract_model_answer(text: str) -> str:
    """
    Extracts the final predicted numerical answer from the LLM's response.
    Correctly handles numbers containing thousands separators (e.g. 448,000 -> 448000, 1,250 -> 1250).

    Priority order:
    1. GSM8K delimiter: '#### <number>'
    2. LaTeX \\boxed{...} (supports nested braces like \\boxed{\$448{,}000})
    3. Explicit phrases (e.g., 'final answer is: <number>', 'answer is <number>')
    4. Fallback: Last number in the text
    """
    if not text or not isinstance(text, str):
        return ""

    # Pre-clean LaTeX math thousands separator syntax ({,} -> ,) and unescape \$ -> $
    normalized_text = text.replace("{,}", ",").replace(r"\$", "$")

    # Regex pattern to match numbers with optional sign, thousands comma separators, and optional decimal
    num_pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    # 1. Match explicit GSM8K format '#### <number>'
    if "####" in normalized_text:
        match = re.search(rf"####\s*[\$€£]?\s*({num_pattern})", normalized_text)
        if match:
            return clean_number(match.group(1))

    # 2. Match LaTeX \boxed{...} handling nested braces
    boxed_contents = find_boxed_contents(text)
    if boxed_contents:
        last_boxed = boxed_contents[-1]
        norm_boxed = last_boxed.replace("{,}", ",").replace(r"\$", "$")
        nums = re.findall(num_pattern, norm_boxed)
        if nums:
            return clean_number(nums[-1])

    # 3. Match answer declaration phrasing
    phrasing_patterns = [
        rf"(?:the\s+)?final\s+answer\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
        rf"(?:the\s+)?answer\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
        rf"(?:the\s+)?result\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
        rf"(?:the\s+)?total\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
        rf"=\s*[\$€£]?\s*({num_pattern})\s*$",
    ]
    for pattern in phrasing_patterns:
        matches = re.findall(pattern, normalized_text, re.IGNORECASE)
        if matches:
            return clean_number(matches[-1])

    # 4. Fallback: extract the last numerical value in the text
    numbers = re.findall(num_pattern, normalized_text)
    if numbers:
        return clean_number(numbers[-1])

    return ""


def compare_answers(predicted: str, correct: str) -> bool:
    """
    Compares predicted answer with ground truth answer.
    Supports exact string match and numerical float equivalence (e.g. '72' == '72.0').
    """
    if not predicted or not correct:
        return False

    pred_clean = str(predicted).strip().replace(",", "")
    corr_clean = str(correct).strip().replace(",", "")

    # Direct string match
    if pred_clean == corr_clean:
        return True

    # Numeric comparison
    try:
        pred_val = float(pred_clean)
        corr_val = float(corr_clean)
        return abs(pred_val - corr_val) < 1e-5
    except (ValueError, TypeError):
        return False


def reevaluate_results_file(csv_path: str):
    """
    Recalculates predictions and correctness for an existing results CSV without making API calls.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    updated_predictions = []
    updated_is_correct = []

    for _, row in df.iterrows():
        pred = extract_model_answer(str(row["model_response"]))
        corr = extract_ground_truth(str(row["correct_answer"]))
        match = compare_answers(pred, corr)

        updated_predictions.append(pred)
        updated_is_correct.append(match)

    df["predicted_answer"] = updated_predictions
    df["is_correct"] = updated_is_correct

    df.to_csv(csv_path, index=False)

    total = len(df)
    correct = sum(updated_is_correct)
    incorrect = total - correct
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print("=" * 55)
    print("RECALCULATED EXPERIMENT SUMMARY")
    print("=" * 55)
    print(f"Total questions processed: {total}")
    print(f"Correct answers:           {correct}")
    print(f"Incorrect answers:         {incorrect}")
    print(f"Accuracy percentage:       {accuracy:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    results_file = os.path.join(os.path.dirname(__file__), "results", "baseline_results.csv")
    reevaluate_results_file(results_file)
