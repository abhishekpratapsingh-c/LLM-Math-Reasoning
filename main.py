import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError


def main():
    # 2. Load environment variables from .env
    load_dotenv()

    # 10. Retrieve API key securely from environment (never hardcoded)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: 'GROQ_API_KEY' not found in environment or .env file.")
        print("Please ensure your key is added to .env as: GROQ_API_KEY=<your_key>")
        return

    # 1 & 3. Configure the OpenAI client pointing to the Groq endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # 6. Read the first mathematical question from data/gsm8k_data.csv using pandas
    data_path = os.path.join(os.path.dirname(__file__), "data", "gsm8k_data.csv")
    try:
        df = pd.read_csv(data_path)
        if df.empty or "question" not in df.columns:
            print(f"Error: Dataset at {data_path} is empty or missing 'question' column.")
            return
        question = str(df.iloc[0]["question"])
    except FileNotFoundError:
        print(f"Error: Could not find dataset at '{data_path}'.")
        return
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    baseline_prompt = "Solve the following mathematical problem and provide the final answer."

    try:
        # 4. Use client.models.list() to print available model IDs
        print("Querying available models from Groq API...")
        models_response = client.models.list()
        model_ids = [model.id for model in models_response.data]
        print(f"\nAvailable Groq Model IDs ({len(model_ids)}):")
        for mid in model_ids:
            print(f" - {mid}")

        # 1, 2 & 3. Verify and use the exact model 'openai/gpt-oss-120b'
        target_model = "openai/gpt-oss-120b"
        if target_model not in model_ids:
            print(f"\nError: Required model '{target_model}' was not found in the available models list.")
            return

        selected_model = target_model
        print(f"\nVerified Model for Inference: {selected_model}")

        # 7. Send the question using Chat Completions API
        print(f"Sending first GSM8K question to Groq ({selected_model})...\n")
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": baseline_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )

        answer = response.choices[0].message.content

        # 8. Print the question and model response
        print("=" * 60)
        print("QUESTION:")
        print("=" * 60)
        print(question)
        print("\n" + "=" * 60)
        print("MODEL RESPONSE:")
        print("=" * 60)
        print(answer)

    # 9. Handle API errors with status code and response body
    except APIStatusError as e:
        print("\nGroq API Status Error:")
        print(f"HTTP Status Code: {e.status_code}")
        print(f"Response Body: {e.response.text}")
        request_id = getattr(e, "request_id", None) or e.response.headers.get("x-request-id")
        if request_id:
            print(f"Request ID: {request_id}")

    except Exception as e:
        print("\nUnexpected Error:")
        print("Error type:", type(e).__name__)
        print("Full error:", str(e))


if __name__ == "__main__":
    main()
