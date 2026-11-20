import os
import re
import sys
import time
import pandas as pd
from dotenv import load_dotenv

# Ensure current directory is accessible for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_ground_truth, extract_model_answer, compare_answers


# Prompt templates for Phase 1 experiments (preserved exactly)
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

# Output filenames mapped by provider and technique
# Groq outputs preserve existing filenames; Gemini outputs are prefixed with 'gemini_'
OUTPUT_FILES = {
    "groq": {
        "baseline":         "baseline_results.csv",
        "step_by_step":     "step_by_step_results.csv",
        "few_shot":         "few_shot_results.csv",
        "structured":       "structured_results.csv",
        "baseline_300":     "baseline_300_results.csv",
        "step_by_step_300": "step_by_step_300_results.csv",
        "few_shot_300":     "few_shot_300_results.csv",
        "structured_300":   "structured_300_results.csv",
    },
    "gemini": {
        "baseline":         "gemini_baseline_results.csv",
        "step_by_step":     "gemini_step_by_step_results.csv",
        "few_shot":         "gemini_few_shot_results.csv",
        "structured":       "gemini_structured_results.csv",
        "baseline_300":     "gemini_baseline_300_results.csv",
        "step_by_step_300": "gemini_step_by_step_300_results.csv",
        "few_shot_300":     "gemini_few_shot_300_results.csv",
        "structured_300":   "gemini_structured_300_results.csv",
    },
}

# Required columns in every results CSV (used for validation on resume)
_REQUIRED_COLS = {"question", "correct_answer", "model_response", "predicted_answer", "is_correct"}


def _save_results_safely(results: list, csv_path: str) -> bool:
    """
    Writes the results list to csv_path via a temporary file, then renames it.

    Writing to a temp file first means the original csv_path is never left in a
    partially written state: os.replace() on Windows replaces the target in one
    filesystem operation, so a crash during writing at worst leaves a stale .tmp
    file (which is cleaned up on the next attempt) while the original CSV is intact.

    Returns True on success, False on failure (the original file is preserved).
    """
    tmp_path = csv_path + ".tmp"
    try:
        pd.DataFrame(results).to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
        return True
    except Exception as e:
        print(f"  WARNING: Could not write results to disk: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def get_llm_client(provider: str):
    """
    Initializes and returns (client, model_name) for the requested provider ('groq' or 'gemini').
    Reads keys and configurable model names from environment variables.
    """
    load_dotenv()
    if provider == "groq":
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing 'GROQ_API_KEY' in environment or .env file.\n"
                "Please ensure GROQ_API_KEY is configured."
            )
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, model_name

    elif provider == "gemini":
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing 'GEMINI_API_KEY' in environment or .env file.\n"
                "Please ensure GEMINI_API_KEY is configured."
            )
        # Default to a verified, currently active Gemini text model; fully configurable via GEMINI_MODEL
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        client = genai.Client(api_key=api_key)
        return client, model_name

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: 'groq', 'gemini'")


def _extract_retry_delay(err: Exception) -> float | None:
    """
    Extracts recommended retry duration (in seconds) from a 429 / RESOURCE_EXHAUSTED error.
    Inspects error details and error message strings, falling back to None if not found.
    """
    # 1. Inspect structured details from google.genai error response if present
    details = getattr(err, "details", None)
    if isinstance(details, dict):
        error_dict = details.get("error", {}) if isinstance(details.get("error"), dict) else details
        raw_details = error_dict.get("details", [])
        if isinstance(raw_details, list):
            for d in raw_details:
                if isinstance(d, dict) and d.get("@type", "").endswith("RetryInfo"):
                    rd = d.get("retryDelay", "")
                    m = re.search(r"(\d+(?:\.\d+)?)", str(rd))
                    if m:
                        return float(m.group(1))

    # 2. Inspect error message string (e.g., 'Please retry in 2.744443109s.' or 'retryDelay: "3s"')
    err_str = str(err)
    m = re.search(r"retry\s+in\s+([\d.]+)\s*s", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"retryDelay[\'\":\s]+([\d.]+)", err_str, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


def call_llm(client, provider: str, model_name: str, prompt: str) -> str:
    """
    Executes an LLM inference call using provider-native SDK with temperature=0.0.
    For Gemini, handles rate limiting (HTTP 429 / RESOURCE_EXHAUSTED) with retry delays.
    Returns the string text generated by the model.
    """
    if provider == "groq":
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""
    elif provider == "gemini":
        from google.genai import types
        max_retries = 3
        fallback_delay = 5.0

        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_rate_limit:
                    retry_delay = _extract_retry_delay(e)
                    if retry_delay is not None and retry_delay > 60:
                        raise RuntimeError(
                            f"Gemini quota unavailable for extended period ({retry_delay:.1f}s > 60s). "
                            "Experiment can be safely resumed later."
                        ) from e

                    if attempt < max_retries:
                        wait_time = (retry_delay if retry_delay is not None else fallback_delay) + 1.0
                        print(f"    [Rate limit 429] Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                raise
    else:
        raise ValueError(f"Unknown provider '{provider}'")


def run_experiment(technique: str = "step_by_step", num_questions: int = None, provider: str = None):
    """
    Executes a mathematical reasoning experiment on GSM8K using a specified prompting technique.

    Provider can be specified via parameter or via LLM_PROVIDER in .env ('groq' or 'gemini').
    Techniques ending in '_300' automatically use the 300-question test split
    (data/gsm8k_test_300.csv) with the corresponding base prompt template.
    All other techniques use the first 20 questions from data/gsm8k_data.csv.

    Resume behaviour
    ----------------
    If the output CSV already exists it is validated then split into two groups:
      - Successful rows (is_api_error=False): loaded into memory and kept as-is;
        the corresponding questions are skipped in the main loop.
      - Failed rows (is_api_error=True): their questions are RE-ATTEMPTED; the old
        failed row is replaced in memory by the new result so the CSV always
        contains exactly one row per question.
    If the existing file is unreadable or missing required columns the experiment
    stops immediately with a clear error -- the file is never overwritten.

    Safe writes
    -----------
    Every per-question save goes through _save_results_safely(), which writes to
    a .tmp file first and then renames it, so the main CSV is never left partially
    written. If saving fails, the experiment halts immediately without making further
    API calls.
    """
    load_dotenv()
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()

    if provider not in OUTPUT_FILES:
        print(f"Error: Unknown LLM_PROVIDER '{provider}'. Available: {list(OUTPUT_FILES.keys())}")
        return

    provider_output_map = OUTPUT_FILES[provider]
    if technique not in provider_output_map:
        print(f"Error: Unknown technique '{technique}'. Available: {list(provider_output_map.keys())}")
        return

    # Determine whether this is a _300 scale-up run
    is_scale_300 = technique.endswith("_300")
    base_technique = technique[:-4] if is_scale_300 else technique  # strip '_300' suffix

    if base_technique not in PROMPTS:
        print(f"Error: No prompt template found for base technique '{base_technique}'.")
        return

    # Set dataset path and question count based on run type
    if is_scale_300:
        data_file = "gsm8k_test_300.csv"
        default_questions = 300
    else:
        data_file = "gsm8k_data.csv"
        default_questions = 20

    if num_questions is None:
        num_questions = default_questions

    # Initialize client and model for chosen provider
    try:
        client, model_name = get_llm_client(provider)
    except Exception as e:
        print(f"Error initializing client for provider '{provider}': {e}")
        return

    # Resolve output path early -- needed for resume check and safe writes
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_filename = provider_output_map[technique]
    results_csv_path = os.path.join(results_dir, output_filename)

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "data", data_file)
    try:
        df = pd.read_csv(data_path)
        if df.empty or "question" not in df.columns or "answer" not in df.columns:
            print(f"Error: Dataset at {data_path} is missing 'question' or 'answer' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Could not find dataset at '{data_path}'.")
        if is_scale_300:
            print("Hint: Run 'python prepare_test_data.py' first to generate the 300-question test dataset.")
        return
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    df_subset = df.head(num_questions)
    experiment_title = f"{provider.upper()} - {technique.replace('_', ' ').upper()}"

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------
    in_memory_results: list = []
    already_done_questions: set = set()   # successful -- skip entirely
    retry_questions: set = set()          # previous API error -- re-attempt

    if os.path.exists(results_csv_path):
        # Fix 1 -- Corrupted CSV protection:
        try:
            existing_df = pd.read_csv(results_csv_path)
        except Exception as e:
            print(f"Error: Could not read existing results file at '{results_csv_path}'.")
            print(f"  Reason: {e}")
            print("Stopping to protect existing data. Inspect or rename the file before retrying.")
            return

        missing_cols = _REQUIRED_COLS - set(existing_df.columns)
        if missing_cols:
            print(f"Error: Existing results file is missing required columns: {missing_cols}")
            print("Stopping to protect existing data. Inspect or rename the file before retrying.")
            return

        # Normalise is_api_error (backward compat: older files without this column)
        if "is_api_error" not in existing_df.columns:
            existing_df["is_api_error"] = False
        existing_df["is_api_error"] = (
            existing_df["is_api_error"]
            .fillna(False)
            .apply(lambda x: str(x).lower() in ["true", "1"])
        )

        success_mask = ~existing_df["is_api_error"]

        # Fix 2 -- Retry API failures:
        successful_df = existing_df[success_mask].copy()
        failed_df     = existing_df[~success_mask].copy()

        in_memory_results      = successful_df.to_dict("records")
        already_done_questions = set(successful_df["question"].astype(str).str.strip())
        retry_questions        = set(failed_df["question"].astype(str).str.strip())

        n_done  = len(already_done_questions)
        n_retry = len(retry_questions)
        n_new   = max(0, num_questions - n_done - n_retry)
        print(f"Resuming {experiment_title}:")
        print(f"  Provider:              {provider}")
        print(f"  Model:                 {model_name}")
        print(f"  Completed (skipping):  {n_done}")
        print(f"  API errors (retrying): {n_retry}")
        print(f"  New questions:         {n_new}")
        print()
    # ------------------------------------------------------------------

    print(f"Starting {experiment_title} Experiment ({len(df_subset)} questions total) "
          f"using provider '{provider}' and model '{model_name}'...\n")

    prompt_template = PROMPTS[base_technique]
    error_count = 0

    for idx, row in df_subset.iterrows():
        question = str(row["question"]).strip()

        # Skip questions that already have a successful saved result
        if question in already_done_questions:
            print(f"[{experiment_title}] [{idx + 1}/{len(df_subset)}] Already completed. Skipping.")
            continue

        is_retry = question in retry_questions
        action_label = "Retrying previous API error" if is_retry else "Processing question"
        print(f"[{experiment_title}] [{idx + 1}/{len(df_subset)}] {action_label}...")

        raw_answer     = str(row["answer"])
        correct_answer = extract_ground_truth(raw_answer)
        prompt         = prompt_template.format(question=question)

        # API call
        is_api_error   = False
        model_response = ""
        try:
            model_response = call_llm(client, provider, model_name, prompt)
        except Exception as e:
            print(f"  WARNING: API/Provider error on Q{idx + 1}: {e}")
            model_response = f"API_ERROR: {e}"
            is_api_error   = True
            error_count   += 1

        # Skip answer extraction for API errors -- no parseable response exists
        predicted_answer = "" if is_api_error else extract_model_answer(model_response)
        is_correct       = False if is_api_error else compare_answers(predicted_answer, correct_answer)

        result_row = {
            "question":        question,
            "correct_answer":  correct_answer,
            "model_response":  model_response,
            "predicted_answer": predicted_answer,
            "is_correct":      is_correct,
            "is_api_error":    is_api_error,
        }

        # Append to the in-memory list (retry questions naturally replace nothing
        # because their old failed row was excluded when we seeded in_memory_results)
        in_memory_results.append(result_row)

        # Fix 3 -- Safe write: rewrite the full in-memory list via a temp file.
        # The original CSV is never left in a partially written state.
        if not _save_results_safely(in_memory_results, results_csv_path):
            print(f"\nCRITICAL ERROR: Failed to save results to '{results_csv_path}' on Q{idx + 1}.")
            print("Stopping experiment immediately to prevent loss of unsaved progress and unnecessary API calls.")
            print("Existing saved results are intact. Please resolve disk/permission issues and restart.")
            return

        status_tag = "API ERROR -- excluded from accuracy" if is_api_error else f"Match: {is_correct}"
        print(f"  Predicted: '{predicted_answer}' | Correct: '{correct_answer}' | {status_tag}")

        # Pace Gemini requests to stay safely below 15 RPM free-tier quota (~5s between calls)
        if provider == "gemini":
            time.sleep(5.0)

    # ------------------------------------------------------------------
    # Final summary: read the complete on-disk CSV so totals always reflect
    # the full file, including any rows from prior resumed runs.
    # ------------------------------------------------------------------
    print(f"\nAll questions processed. Reading final results from: {results_csv_path}")
    try:
        final_df = pd.read_csv(results_csv_path)
    except Exception as e:
        print(f"Error reading final results for summary: {e}")
        return

    total_in_file = len(final_df)

    if "is_api_error" in final_df.columns:
        is_error_mask = final_df["is_api_error"].fillna(False).apply(
            lambda x: str(x).lower() in ["true", "1"]
        )
    else:
        is_error_mask = pd.Series([False] * total_in_file, index=final_df.index)

    api_errors      = int(is_error_mask.sum())
    valid_responses = total_in_file - api_errors

    is_correct_mask = final_df["is_correct"].apply(lambda x: str(x).lower() in ["true", "1"])
    correct_count   = int((is_correct_mask & ~is_error_mask).sum())
    incorrect_count = valid_responses - correct_count
    accuracy        = (correct_count / valid_responses * 100) if valid_responses > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY: {experiment_title} REASONING")
    print("=" * 60)
    print(f"Provider:                  {provider}")
    print(f"Model:                     {model_name}")
    print(f"Total questions in file:   {total_in_file}")
    print(f"API errors:                {api_errors}")
    print(f"Valid model responses:     {valid_responses}")
    print(f"Correct answers:           {correct_count}")
    print(f"Incorrect answers:         {incorrect_count}")
    print(f"Accuracy (valid only):     {accuracy:.2f}%")
    if api_errors > 0:
        print(f"NOTE: {api_errors} API error row(s) saved in CSV but excluded from accuracy.")
    print("=" * 60)


def test_single_question(provider: str = "gemini"):
    """
    Sends exactly ONE GSM8K question to the specified provider using the baseline prompt.
    Prints:
      - Selected provider
      - Selected model
      - Generated response
      - API success or readable error
    Does not save to or modify any results CSV.
    """
    load_dotenv()
    print("=" * 60)
    print("TEST MODE: SINGLE QUESTION RUN")
    print("=" * 60)
    print(f"Selected Provider: {provider}")

    try:
        client, model_name = get_llm_client(provider)
        print(f"Selected Model:    {model_name}")
    except Exception as e:
        print(f"API Status:        ERROR initializing client: {e}")
        return

    # Load 1st question from test split (fallback to gsm8k_data.csv if test split is not found)
    data_path = os.path.join(os.path.dirname(__file__), "data", "gsm8k_test_300.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "data", "gsm8k_data.csv")

    try:
        df = pd.read_csv(data_path)
        sample_q = str(df.iloc[0]["question"]).strip()
        sample_a = str(df.iloc[0]["answer"]).strip()
    except Exception as e:
        print(f"API Status:        ERROR loading sample question from dataset: {e}")
        return

    correct_answer = extract_ground_truth(sample_a)
    prompt = PROMPTS["baseline"].format(question=sample_q)

    print(f"\nQuestion: {sample_q}")
    print(f"Ground Truth Answer: {correct_answer}")
    print("\nSending request to API...")

    try:
        response_text = call_llm(client, provider, model_name, prompt)
        predicted_answer = extract_model_answer(response_text)
        is_correct = compare_answers(predicted_answer, correct_answer)

        print("\nAPI Status: SUCCESS")
        print("-" * 60)
        print("Generated Response:")
        print(response_text.strip())
        print("-" * 60)
        print(f"Extracted Answer:  '{predicted_answer}'")
        print(f"Ground Truth:      '{correct_answer}'")
        print(f"Correct:           {is_correct}")
    except Exception as e:
        print(f"\nAPI Status: FAILED")
        print(f"Readable Error: {e}")

    print("=" * 60)


def main():
    # Usage:
    #   python main.py test [gemini|groq]        -> Runs single-question test mode (no CSV output)
    #   python main.py <technique>               -> Runs experiment with provider from LLM_PROVIDER
    #   python main.py <technique> <provider>    -> Runs experiment with explicit provider
    load_dotenv()
    args = sys.argv[1:]

    if args and args[0].lower() == "test":
        provider = args[1].lower() if len(args) > 1 else os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        test_single_question(provider=provider)
        return

    technique = args[0] if len(args) > 0 else "step_by_step"
    provider = args[1].lower() if len(args) > 1 else None
    run_experiment(technique=technique, provider=provider)


if __name__ == "__main__":
    main()
