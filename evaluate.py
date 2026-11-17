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
    r"""
    Finds the full content of all \boxed{...} expressions in text,
    properly accounting for nested curly braces (e.g. \boxed{\$448{,}000}).
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
    e.g., '$448,000' -> '448000', '1,250' -> '1250', '448 000' -> '448000', '72' -> '72'
    """
    if not num_str:
        return ""
    s = str(num_str).strip()
    for sym in ["$", "€", "£", "¥"]:
        s = s.replace(sym, "")
    # Remove commas, regular spaces, and non-breaking spaces
    s = s.replace(",", "").replace(" ", "").replace("\u202f", "").replace("\u00a0", "")
    return s.strip()


def extract_first_number(text: str) -> str:
    """Finds the first valid numerical value in a text snippet."""
    if not text:
        return ""
    num_pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    match = re.search(num_pattern, text)
    if match:
        return clean_number(match.group(0))
    return ""


def extract_model_answer(text: str) -> str:
    r"""
    Extracts the final predicted numerical answer from the LLM's response.
    Prioritizes explicitly marked final answers:
    1. GSM8K delimiter: '#### <number>'
    2. LaTeX \boxed{...} (supports nested braces like \boxed{\$448{,}000})
    3. Dedicated Final Answer sections/headers (### Final answer, **State the final answer**, **Result**, etc.)
       prioritizing bolded numbers within that section.
    4. Explicit inline answer phrases (e.g., 'final answer is: <number>', '**Answer: <number>**').
    5. Bolded numbers in the concluding paragraph.
    6. Fallback (least preferred): Last number in the text (used only when no markers exist).
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize LaTeX math thousands separator ({,} -> ,) and unescape \$ -> $
    norm_text = text.replace("{,}", ",").replace(r"\$", "$")

    # Normalize space thousands separators: e.g. "448 000" or "448\u202f000" -> "448000"
    # Only collapses whitespace if it occurs between a digit and exactly 3 digits
    norm_text = re.sub(r"(?<=\d)[ \u202f\u00a0](?=\d{3}(?!\d))", "", norm_text)

    num_pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

    # 1. Match explicit GSM8K format '#### <number>'
    if "####" in norm_text:
        match = re.search(rf"####\s*[\$€£]?\s*({num_pattern})", norm_text)
        if match:
            return clean_number(match.group(1))

    # 2. Match LaTeX \boxed{...} handling nested braces
    boxed_contents = find_boxed_contents(text)
    if boxed_contents:
        last_boxed = boxed_contents[-1]
        norm_boxed = last_boxed.replace("{,}", ",").replace(r"\$", "$")
        norm_boxed = re.sub(r"(?<=\d)[ \u202f\u00a0](?=\d{3}(?!\d))", "", norm_boxed)

        # Check for bolded number inside box
        bold_in_box = re.findall(r"\*\*([^\*]+)\*\*", norm_boxed)
        for b in reversed(bold_in_box):
            num = extract_first_number(b)
            if num:
                return num
        # Extract last number inside box
        nums = re.findall(num_pattern, norm_boxed)
        if nums:
            return clean_number(nums[-1])

    # 3. Match dedicated Final Answer Section/Header
    section_regex = (
        r"(?:^|\n)\s*(?:###\s*|\*\*\s*|\b)"
        r"(?:Step\s*\d+\s*[:\-]\s*)?"
        r"(?:State\s+the\s+final\s+answer|State\s+the\s+answer|Final\s+answer|Answer|Result)\b"
        r"[\*\:\s\-]*(.*?)(?=\n\s*(?:###|\*\*Step|\Z))"
    )
    section_matches = list(re.finditer(section_regex, norm_text, re.IGNORECASE | re.DOTALL))
    if section_matches:
        final_section = section_matches[-1].group(1).strip()

        # Prioritize bolded numbers in the final answer section
        bold_matches = re.findall(r"\*\*([^\*]+)\*\*", final_section)
        for b in bold_matches:
            num = extract_first_number(b)
            if num:
                return num

        # If no bold numbers, check for inline phrasing in this section
        phrasing_in_sec = [
            rf"(?:the\s+)?final\s+answer\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
            rf"(?:the\s+)?answer\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
            rf"(?:is\s+equal\s+to|equals|=)\s*[\$€£]?\s*({num_pattern})",
        ]
        for pattern in phrasing_in_sec:
            m = re.findall(pattern, final_section, re.IGNORECASE)
            if m:
                return clean_number(m[-1])

        # Take the first number in the final answer section (answers are typically stated upfront)
        sec_nums = re.findall(num_pattern, final_section)
        if sec_nums:
            return clean_number(sec_nums[0])

    # 4. Match explicit inline phrasing throughout the document
    phrasing_patterns = [
        rf"(?:the\s+)?final\s+answer\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
        rf"(?:the\s+)?answer\s*(?:is|:|=)?\s*[\$€£]?\s*({num_pattern})",
        rf"\*\*(?:Answer|Result)\s*:\s*([^\*]+)\*\*",
        rf"=\s*[\$€£]?\s*({num_pattern})\s*$",
    ]
    for pattern in phrasing_patterns:
        matches = re.findall(pattern, norm_text, re.IGNORECASE)
        if matches:
            cand = matches[-1]
            num = extract_first_number(cand) if not re.match(rf"^{num_pattern}$", cand) else clean_number(cand)
            if num:
                return num

    # 5. Check for bolded numbers in the last paragraph
    paragraphs = [p.strip() for p in norm_text.split("\n\n") if p.strip()]
    if paragraphs:
        last_para = paragraphs[-1]
        bold_matches = re.findall(r"\*\*([^\*]+)\*\*", last_para)
        for b in reversed(bold_matches):
            num = extract_first_number(b)
            if num:
                return num

    # 6. Fallback (least preferred): Last number in the text
    numbers = re.findall(num_pattern, norm_text)
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
    Returns (original_accuracy, new_accuracy, changes_list).
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    orig_total = len(df)
    orig_correct = sum(1 for v in df["is_correct"] if str(v).lower() in ["true", "1"])
    orig_accuracy = (orig_correct / orig_total * 100) if orig_total > 0 else 0.0

    changes = []
    updated_predictions = []
    updated_is_correct = []

    for idx, row in df.iterrows():
        old_pred = str(row["predicted_answer"])
        old_correct = bool(str(row["is_correct"]).lower() in ["true", "1"])
        raw_response = str(row["model_response"])
        ground_truth = extract_ground_truth(str(row["correct_answer"]))

        new_pred = extract_model_answer(raw_response)
        new_correct = compare_answers(new_pred, ground_truth)

        updated_predictions.append(new_pred)
        updated_is_correct.append(new_correct)

        if old_pred != new_pred or old_correct != new_correct:
            changes.append({
                "index": idx + 1,
                "question": str(row["question"])[:60] + "...",
                "correct_answer": ground_truth,
                "old_predicted": old_pred,
                "new_predicted": new_pred,
                "old_is_correct": old_correct,
                "new_is_correct": new_correct,
            })

    df["predicted_answer"] = updated_predictions
    df["is_correct"] = updated_is_correct
    df.to_csv(csv_path, index=False)

    new_total = len(df)
    new_correct = sum(updated_is_correct)
    new_accuracy = (new_correct / new_total * 100) if new_total > 0 else 0.0

    return {
        "file": csv_path,
        "total": new_total,
        "orig_correct": orig_correct,
        "orig_accuracy": orig_accuracy,
        "new_correct": new_correct,
        "new_accuracy": new_accuracy,
        "changes": changes,
    }


if __name__ == "__main__":
    base_file = os.path.join(os.path.dirname(__file__), "results", "baseline_results.csv")
    sbs_file = os.path.join(os.path.dirname(__file__), "results", "step_by_step_results.csv")
    few_shot_file = os.path.join(os.path.dirname(__file__), "results", "few_shot_results.csv")
    structured_file = os.path.join(os.path.dirname(__file__), "results", "structured_results.csv")

    for f in [base_file, sbs_file, few_shot_file, structured_file]:
        if os.path.exists(f):
            res = reevaluate_results_file(f)
            if res:
                print(f"Results for {os.path.basename(res['file'])}:")
                print(f"  Original Accuracy: {res['orig_accuracy']:.2f}% ({res['orig_correct']}/{res['total']})")
                print(f"  New Accuracy:      {res['new_accuracy']:.2f}% ({res['new_correct']}/{res['total']})")
                print(f"  Changes ({len(res['changes'])}):")
                for c in res['changes']:
                    print(f"    Q{c['index']}: '{c['old_predicted']}' ({c['old_is_correct']}) -> '{c['new_predicted']}' ({c['new_is_correct']})")
