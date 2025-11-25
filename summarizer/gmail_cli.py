from __future__ import annotations

import argparse
from datetime import datetime, date
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .gmail_checker import authenticate_gmail, process_court_emails

# Load environment variables from .env file
load_dotenv()


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Gmail for court opinions and generate summaries."
    )
    parser.add_argument(
        "--date",
        type=parse_date,
        default=None,
        help="Date to search for emails (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--sender",
        nargs="+",
        default=["uscourts@updates.uscourts.gov", "drewwileyroberts@gmail.com"],
        help="Email sender(s) to search for (default: uscourts@updates.uscourts.gov and drewwileyroberts@gmail.com)",
    )
    parser.add_argument(
        "--pdf-dir",
        default="pdfs",
        help="Directory to save downloaded PDFs (default: pdfs)",
    )
    parser.add_argument(
        "--summary-dir",
        default="summaries",
        help="Directory to save summaries (default: summaries)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a text prompt file for summarization",
    )
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to Gmail API credentials.json (default: credentials.json)",
    )
    parser.add_argument(
        "--token",
        default="token.json",
        help="Path to store/read token.json (default: token.json)",
    )
    parser.add_argument(
        "--email-to",
        nargs="+",
        default=None,
        help="Email address(es) to send summary to (optional, space-separated)",
    )
    parser.add_argument(
        "--email-bcc",
        nargs="+",
        default=None,
        help="Email address(es) to BCC (optional, space-separated)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if summaries already exist for this date",
    )

    args = parser.parse_args(argv)

    # Authenticate with Gmail
    print(f"[info] Authenticating with Gmail API...")
    try:
        service = authenticate_gmail(args.credentials, args.token)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return 1
    except Exception as e:
        print(f"[error] Authentication failed: {e}")
        return 1

    print(f"[ok] Successfully authenticated")

    # Process emails
    pdf_count = process_court_emails(
        service=service,
        sender=args.sender,
        search_date=args.date,
        pdf_dir=Path(args.pdf_dir),
        summary_dir=Path(args.summary_dir),
        prompt_file=args.prompt_file,
        email_to=args.email_to,
        email_bcc=args.email_bcc,
        force=args.force,
    )

    print(f"\n[ok] Processed {pdf_count} PDF(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

