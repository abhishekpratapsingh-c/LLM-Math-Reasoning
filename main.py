import os
import sys
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

# Ensure current directory is accessible for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_ground_truth, extract_model_answer, compare_answers


# Prompt templates for Phase 1 experiments
PROMPTS = {
    "baseline": (
        "Solve the following mathematical problem and provide the final answer.\n\n"
        "Question: {question}"
    ),
    "step_by_step": (
        "Solve the following mathematical problem step by step.\n"
        "Break the problem into smaller calculations.\n"
        "Explain each step clearly and provide the final numerical answer.\n\n"
        "Question: {question}"
    ),
    "few_shot": (
        "Solve the following mathematical problems by showing step-by-step calculations and stating the final answer.\n\n"
        "Example 1:\n"
        "Question: Sarah has 5 boxes, each containing 4 chocolates. How many chocolates does Sarah have in total?\n"
        "Solution:\n"
        "Step 1: Identify the number of boxes: 5 boxes.\n"
        "Step 2: Identify the number of chocolates in each box: 4 chocolates.\n"
        "Step 3: Multiply the number of boxes by the chocolates per box: 5 * 4 = 20 chocolates.\n"
        "Final Answer: 20\n\n"
        "Example 2:\n"
        "Question: A person buys a notebook for ₹30 and a pen for ₹10. What is the total cost of the purchase?\n"
        "Solution:\n"
        "Step 1: Identify the cost of the notebook: ₹30.\n"
        "Step 2: Identify the cost of the pen: ₹10.\n"
        "Step 3: Add the cost of both items: 30 + 10 = ₹40.\n"
        "Final Answer: 40\n\n"
        "Now solve this problem:\n"
        "Question: {question}"
    ),
    "structured": (
        "Solve the following mathematical problem using this exact format. Follow all five steps in order without skipping any:\n"
        "1. Understand the problem.\n"
        "2. Identify the given values.\n"
        "3. Determine the required mathematical operation.\n"
        "4. Perform calculations step by step.\n"
        "5. Clearly state the final numerical answer in this exact format: Final Answer: <number>\n\n"
        "Question: {question}"
    ),
}

OUTPUT_FILES = {
    "baseline": "baseline_results.csv",
    "step_by_step": "step_by_step_results.csv",
    "few_shot": "few_shot_results.csv",
    "structured": "structured_results.csv",
}


def run_experiment(technique: str = "step_by_step", num_questions: int = 20):
    """
    Executes a mathematical reasoning experiment on GSM8K using a specified prompting technique.
    """
    if technique not in PROMPTS:
        print(f"Error: Unknown technique '{technique}'. Available: {list(PROMPTS.keys())}")
        return

    # 3 & 13. Load environment variables securely (API key is never printed or hardcoded)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: 'GROQ_API_KEY' not found in environment or .env file.")
        print("Please ensure your key is added to .env as: GROQ_API_KEY=<your_key>")
        return

    # Configure Groq client using OpenAI SDK
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # 2. Consistent model specification across experiments
    model_name = "openai/gpt-oss-120b"

    # 1. Read first 20 questions from data/gsm8k_data.csv (unchanged)
    data_path = os.path.join(os.path.dirname(__file__), "data", "gsm8k_data.csv")
    try:
        df = pd.read_csv(data_path)
        if df.empty or "question" not in df.columns or "answer" not in df.columns:
            print(f"Error: Dataset at {data_path} is missing 'question' or 'answer' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Could not find dataset at '{data_path}'.")
        return
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    df_subset = df.head(num_questions)
    experiment_title = technique.replace("_", " ").upper()
    print(f"Starting {experiment_title} Experiment ({len(df_subset)} questions) using model '{model_name}'...\n")

    results = []
    prompt_template = PROMPTS[technique]

    for idx, row in df_subset.iterrows():
        question = str(row["question"]).strip()
        raw_answer = str(row["answer"])

        # Ground truth extraction (using evaluate.py)
        correct_answer = extract_ground_truth(raw_answer)

        # 4. Construct prompt for current question
        prompt = prompt_template.format(question=question)

        print(f"[{idx + 1}/{len(df_subset)}] Processing question...")

        model_response = ""
        # 5 & 13. Call Groq API with robust error handling
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            model_response = response.choices[0].message.content or ""
        except APIStatusError as e:
            print(f"  API Status Error on question {idx + 1} (Status {e.status_code}): {e.response.text}")
            model_response = f"API_ERROR: HTTP {e.status_code}"
        except Exception as e:
            print(f"  Unexpected error on question {idx + 1}: {e}")
            model_response = f"ERROR: {e}"

        # 7. Reuse corrected answer extraction logic from evaluate.py
        predicted_answer = extract_model_answer(model_response)

        # Compare predicted answer with ground truth
        is_correct = compare_answers(predicted_answer, correct_answer)

        print(f"  Predicted: '{predicted_answer}' | Correct: '{correct_answer}' | Match: {is_correct}")

        # 6. Save experiment fields
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "model_response": model_response,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
        })

    # 9. Calculate overall accuracy
    total_processed = len(results)
    correct_count = sum(1 for r in results if r["is_correct"])
    incorrect_count = total_processed - correct_count
    accuracy = (correct_count / total_processed * 100) if total_processed > 0 else 0.0

    # 8 & 10. Save results separately without overwriting other experiment files
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_filename = OUTPUT_FILES[technique]
    results_csv_path = os.path.join(results_dir, output_filename)

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults successfully saved to: {results_csv_path}")

    # 9. Print experiment summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY: {experiment_title} REASONING")
    print("=" * 60)
    print(f"Total questions processed: {total_processed}")
    print(f"Correct answers:           {correct_count}")
    print(f"Incorrect answers:         {incorrect_count}")
    print(f"Accuracy percentage:       {accuracy:.2f}%")
    print("=" * 60)


def main():
    # Defaults to 'step_by_step'; can also be passed via CLI argument: python main.py baseline
    technique = sys.argv[1] if len(sys.argv) > 1 else "step_by_step"
    run_experiment(technique=technique, num_questions=20)


if __name__ == "__main__":
    main()
