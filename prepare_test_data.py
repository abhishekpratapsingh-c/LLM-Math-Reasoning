import os
import pandas as pd
from datasets import load_dataset


def prepare_gsm8k_test_data(num_samples: int = 300):
    """
    Downloads the official GSM8K dataset, extracts the first 300 examples
    from the test split, validates data integrity, and saves to data/gsm8k_test_300.csv.
    """
    print("Loading official GSM8K dataset ('openai/gsm8k', 'main')...")

    # 2. Load the official dataset
    try:
        dataset = load_dataset("openai/gsm8k", "main")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        return

    # Verify 'test' split exists
    if "test" not in dataset:
        print("Error: 'test' split not found in loaded dataset.")
        return

    test_split = dataset["test"]
    print(f"Available test split size: {len(test_split)} questions")

    # 3. Select exactly the first 300 questions from test split preserving order
    if len(test_split) < num_samples:
        print(f"Error: Not enough questions in test split ({len(test_split)} < {num_samples}).")
        return

    selected_data = test_split[:num_samples]

    # 4. Filter only the 'question' and 'answer' columns into a DataFrame
    df = pd.DataFrame(selected_data)[["question", "answer"]]

    # 8. Basic validation
    # Check 1: Column existence
    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        print(f"Validation Error: Missing required columns. Found: {list(df.columns)}")
        return

    # Check 2: Row count
    if len(df) != num_samples:
        print(f"Validation Error: Expected exactly {num_samples} rows, got {len(df)}.")
        return

    # Check 3: Missing or null values
    null_questions = df["question"].isna().sum()
    null_answers = df["answer"].isna().sum()
    empty_questions = (df["question"].astype(str).str.strip() == "").sum()
    empty_answers = (df["answer"].astype(str).str.strip() == "").sum()

    if null_questions > 0 or null_answers > 0 or empty_questions > 0 or empty_answers > 0:
        print("Validation Error: Found missing or empty question/answer fields in the dataset.")
        return

    # Resolve output path: data/gsm8k_test_300.csv
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gsm8k_test_300.csv")

    # Safety check: do not overwrite existing file
    if os.path.exists(output_path):
        print(f"File already exists at '{output_path}'. Not overwriting. Stopping.")
        return

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDataset successfully saved to: {output_path}")

    # Print summary and sample question for inspection
    print("\n" + "=" * 60)
    print("DATASET PREPARATION & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Rows Saved:        {len(df)}")
    print(f"Columns:                 {list(df.columns)}")
    print(f"Null or Missing Values:  0")
    print("=" * 60)
    print("\nSample Question (Item 1):")
    print("-" * 60)
    print(df.iloc[0]["question"])
    print("\nSample Answer (Item 1):")
    print("-" * 60)
    print(df.iloc[0]["answer"])
    print("=" * 60)


if __name__ == "__main__":
    prepare_gsm8k_test_data(num_samples=300)
