"""Runner for batch product description generation."""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from product_generator import generate_contextualized_descriptions_batched

# Defaults
DEFAULT_INPUT = "horseland_products.xlsx"
DEFAULT_OUTPUT = "horseland_products_described.xlsx"
DEFAULT_RULES = "Australian_Horse_Guide_Association_Rules_2025.xlsx"
DEFAULT_MODEL = "gpt-5"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run product description generation.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to products Excel/CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for output file")
    parser.add_argument("--rules", default=DEFAULT_RULES, help="Path to rulebook Excel/CSV (optional)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=12, help="Rows per API call")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent API calls")
    parser.add_argument("--preview-rows", type=int, default=None, help="Optional subset for testing")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Pass --api-key or set the environment variable.")
    os.environ["OPENAI_API_KEY"] = api_key

    input_path = str(Path(args.input))
    output_path = str(Path(args.output))
    rules_path = str(Path(args.rules)) if args.rules else None

    df_out = generate_contextualized_descriptions_batched(
    input_path=input_path,
    output_path=output_path,
    model=args.model,
    temperature=args.temperature,
    batch_size=max(1, args.batch_size),
    concurrency=max(1, args.concurrency),
    preview_rows=args.preview_rows,
    rules_path=rules_path,
)
    print("Wrote", len(df_out), "rows to", output_path)


if __name__ == "__main__":
    main()
