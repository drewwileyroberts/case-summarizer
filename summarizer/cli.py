from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .pdf_utils import extract_text_from_pdf
from .openai_summarizer import summarize_text, DEFAULT_MODEL

# Load environment variables from .env file
load_dotenv()


def _write_output(output_dir: Path, input_pdf: Path, summary_text: str, opinion_date: str | None = None, case_number: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from metadata if available
    if opinion_date and case_number:
        # Convert YYYY-MM-DD to YYYY.MM.DD
        formatted_date = opinion_date.replace("-", ".")
        filename = f"{formatted_date}_{case_number}.txt"
    else:
        # Fallback to original format
        filename = f"{input_pdf.stem}-summary.txt"
    
    out_path = output_dir / filename
    out_path.write_text(summary_text, encoding="utf-8")
    return out_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize one or more PDFs via OpenAI.")
    parser.add_argument("--pdf", nargs="+", required=True, help="Path(s) to PDF file(s)")
    parser.add_argument("--prompt-file", default=None, help="Path to a text prompt file")
    parser.add_argument("--prompt", default=None, help="Inline prompt string (overrides file)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--output-dir", default="summaries", help="Directory for output summaries")

    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    pdf_paths = [Path(p) for p in args.pdf]

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"[warn] Skipping missing file: {pdf_path}")
            continue
        print(f"[info] Extracting text from: {pdf_path}")
        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            print(f"[warn] No text extracted: {pdf_path}")
            continue

        print(f"[info] Summarizing: {pdf_path}")
        result = summarize_text(
            text,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            model=args.model,
        )

        out_file = _write_output(
            output_dir,
            pdf_path,
            result.combined_summary,
            result.opinion_date,
            result.case_number,
        )
        print(f"[ok] Wrote summary: {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



