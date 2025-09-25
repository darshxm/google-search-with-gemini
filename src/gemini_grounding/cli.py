import argparse
from pathlib import Path
from .core import run_prompt_and_resolve

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Gemini grounded prompt and resolve citation redirect URLs."
    )
    p.add_argument(
        '-p', '--prompt',
        default="Who won the euro 2024?",
        help="Prompt to send to the model.",
    )
    p.add_argument('-o', '--output', type=str, help="Optional path to save JSON result.")
    p.add_argument('--model', default="gemini-2.5-flash", help="Model name.")
    p.add_argument('--no-verbose', action='store_true', help="Suppress printed output.")
    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else None
    run_prompt_and_resolve(
        args.prompt, model=args.model, output_json=output_path, verbose=not args.no_verbose
    )

