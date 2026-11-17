import os
import sys
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

# Ensure current directory is on sys.path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_ground_truth, extract_model_answer, compare_answers


def main():
    # Load environment variables
    load_dotenv()

    # 14. Keep API key in .env and never print it
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

    # 2. Model specification
    model_name = "openai/gpt-oss-120b"

    # 1. Read first 20 questions from data/gsm8k_data.csv
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

    # Select only the first 20 questions
    df_subset = df.head(20)
    print(f"Starting Baseline Experiment: {len(df_subset)} questions using model '{model_name}'...\n")

    results = []

    for idx, row in df_subset.iterrows():
        question = str(row["question"]).strip()
        raw_answer = str(row["answer"])

        # 6. Extract ground truth answer (after ####)
        correct_answer = extract_ground_truth(raw_answer)

        # 3. Exact baseline prompt format
        prompt = (
            f"Solve the following mathematical problem and provide the final answer.\n\n"
            f"Question: {question}"
        )

        print(f"[{idx + 1}/{len(df_subset)}] Processing question...")

        model_response = ""
        # 4 & 11. Send question with API error handling
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

        # 6. Extract predicted answer from model response
        predicted_answer = extract_model_answer(model_response)

        # 7. Compare predicted answer with correct answer
        is_correct = compare_answers(predicted_answer, correct_answer)

        print(f"  Predicted: '{predicted_answer}' | Correct: '{correct_answer}' | Match: {is_correct}")

        # 5. Store question details
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "model_response": model_response,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
        })

    # 8. Calculate overall accuracy
    total_processed = len(results)
    correct_count = sum(1 for r in results if r["is_correct"])
    incorrect_count = total_processed - correct_count
    accuracy = (correct_count / total_processed * 100) if total_processed > 0 else 0.0

    # 9. Save all experiment results to results/baseline_results.csv
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = os.path.join(results_dir, "baseline_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults successfully saved to: {results_csv_path}")

    # 10. Print experiment summary
    print("\n" + "=" * 55)
    print("EXPERIMENT SUMMARY: BASELINE MATHEMATICAL REASONING")
    print("=" * 55)
    print(f"Total questions processed: {total_processed}")
    print(f"Correct answers:           {correct_count}")
    print(f"Incorrect answers:         {incorrect_count}")
    print(f"Accuracy percentage:       {accuracy:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
