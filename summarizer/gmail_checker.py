from __future__ import annotations

import base64
import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .uscourts_scraper import doc_id_from_url, extract_links_from_text, parse_email_entries, process_uscourts_link
from .pdf_utils import extract_text_from_pdf
from .openai_summarizer import summarize_text


# Gmail API scopes - readonly for checking emails, send for sending summaries
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send"
]


def authenticate_gmail(credentials_path: str = "credentials.json", token_path: str = "token.json"):
    """
    Authenticate with Gmail API using OAuth 2.0.
    
    Args:
        credentials_path: Path to credentials.json from Google Cloud Console
        token_path: Path to store the token.json (will be created on first run)
        
    Returns:
        Authenticated Gmail API service
    """
    creds = None
    
    # Check if we have a saved token
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found: {credentials_path}\n"
                    "Please download it from Google Cloud Console:\n"
                    "1. Go to https://console.cloud.google.com/\n"
                    "2. Enable Gmail API\n"
                    "3. Create OAuth 2.0 credentials\n"
                    "4. Download and save as credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    
    return build("gmail", "v1", credentials=creds)

# Sender address used as To: when sending BCC-only (so recipients appear only on BCC)
SENDER_EMAIL = "fed.cir.summaries@gmail.com"


def get_email_body(message: dict) -> str:
    """
    Extract the body text from a Gmail message.
    
    Args:
        message: Gmail message object with payload
        
    Returns:
        Email body as plain text
    """
    def extract_from_parts(parts):
        """Recursively extract text from parts."""
        text_body = ""
        html_body = ""
        
        for part in parts:
            mime_type = part.get("mimeType", "")
            
            # If this part has nested parts, recurse
            if "parts" in part:
                nested_text, nested_html = extract_from_parts(part["parts"])
                if nested_text:
                    text_body = nested_text
                if nested_html and not text_body:
                    html_body = nested_html
            
            # Extract text/plain
            elif mime_type == "text/plain" and "data" in part.get("body", {}):
                text_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
            
            # Extract text/html as fallback
            elif mime_type == "text/html" and "data" in part.get("body", {}) and not text_body:
                html_body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
        
        return text_body, html_body
    
    body = ""
    
    if "payload" not in message:
        return body
    
    payload = message["payload"]
    
    # Handle simple messages (single body)
    if "body" in payload and "data" in payload["body"]:
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
    
    # Handle multipart messages
    elif "parts" in payload:
        text_body, html_body = extract_from_parts(payload["parts"])
        
        if text_body:
            body = text_body
        elif html_body:
            # Simple HTML tag removal
            body = re.sub(r"<[^>]+>", "", html_body)
    
    # Truncate at the footer to avoid unsubscribe/junk links
    footer_marker = "To view or to search for other opinions and orders"
    if footer_marker in body:
        body = body.split(footer_marker)[0]
    
    return body


def search_emails(
    service,
    sender: str,
    search_date: date | None = None,
    max_results: int = 10,
) -> List[dict]:
    """
    Search for emails from a specific sender on a specific date.
    
    Args:
        service: Authenticated Gmail API service
        sender: Email address to search for (e.g., uscourts@updates.uscourts.gov)
        search_date: Date to search for (defaults to today)
        max_results: Maximum number of emails to return
        
    Returns:
        List of email message objects with full details
    """
    if search_date is None:
        search_date = date.today()
    
    # Format dates for Gmail query (YYYY/MM/DD)
    # after: is inclusive, before: is exclusive, so we need tomorrow's date for before:
    date_str = search_date.strftime("%Y/%m/%d")
    next_date_str = (search_date + timedelta(days=1)).strftime("%Y/%m/%d")
    
    # Build query
    query = f"from:{sender} after:{date_str} before:{next_date_str}"
    
    print(f"[info] Searching Gmail for: {query}")
    
    # Search for messages
    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results,
    ).execute()
    
    messages = results.get("messages", [])
    
    if not messages:
        print(f"[info] No emails found from {sender} on {date_str}")
        return []
    
    print(f"[info] Found {len(messages)} email(s)")
    
    # Fetch full message details
    full_messages = []
    for msg in messages:
        full_msg = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="full",
        ).execute()
        full_messages.append(full_msg)
    
    return full_messages


@dataclass
class CaseSummary:
    """Data class to hold case summary information."""
    case_name: str
    is_precedential: bool
    summary_text: str
    opinion_date: str | None
    case_number: str | None
    pdf_url: str | None = None
    # Structured fields from new decision tree approach
    is_patent_case: bool = False
    is_copyright_case: bool = False  # Copyright infringement, ownership, fair use, DMCA, etc.
    is_trade_secret_case: bool = False  # Trade secret misappropriation, DTSA, state trade secret laws
    is_trademark_case: bool = False  # Trademark infringement, dilution, Lanham Act, etc.
    panel_judges: List[str] = None
    author_judge: str | None = None
    case_summary: str | None = None
    major_holdings: str | None = None
    final_disposition: str | None = None
    is_rule_42b_dismissal: bool = False  # Fed. R. App. P. 42(b) dismissal (no opinion content)
    is_rule_36_affirmance: bool = False  # Fed. R. App. P. Rule 36 affirmance (minimal opinion content)
    patent_law_issues: List[str] = None  # List of patent law issues addressed (for patent cases only)
    landing_url: str | None = None
    error: str | None = None  # Set when the case could not be summarized; listed for manual review

    def __post_init__(self):
        # Initialize lists to empty if None
        if self.panel_judges is None:
            self.panel_judges = []
        if self.patent_law_issues is None:
            self.patent_law_issues = []


def _markdown_to_html(text: str) -> str:
    """
    Convert simple markdown formatting to HTML.
    Handles **bold**, *italic*, and newlines.
    """
    # Escape HTML characters first
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    # Convert **bold** to <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert *italic* to <em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert newlines to <br> for proper spacing
    text = text.replace('\n', '<br>')
    
    return text


def _format_judges_html(panel_judges: List[str], author_judge: str | None) -> str:
    """Format the judge panel with author underlined.

    Args:
        panel_judges: List of judge names
        author_judge: Name of the authoring judge (to be underlined)

    Returns:
        HTML string with judges formatted, author underlined
    """
    if not panel_judges:
        return ""

    # Handle special cases
    if panel_judges == ["Per Curiam"] or panel_judges == ["Unsigned"]:
        return panel_judges[0]

    # Format judges with author underlined
    formatted_judges = []
    for judge in panel_judges:
        if author_judge and judge == author_judge:
            formatted_judges.append(f"<u>{judge}</u>")
        else:
            formatted_judges.append(judge)

    return ", ".join(formatted_judges)


def _format_ip_types(summary: CaseSummary) -> str:
    """Return comma-separated list of IP types for this case."""
    types = []
    if summary.is_patent_case:
        types.append("Patent")
    if summary.is_copyright_case:
        types.append("Copyright")
    if summary.is_trade_secret_case:
        types.append("Trade Secret")
    if summary.is_trademark_case:
        types.append("Trademark")
    return ", ".join(types)


def _format_final_disposition_line(final_disposition: str | None) -> str:
    """Return the email metadata line for a case's final disposition."""
    if not final_disposition:
        return ""

    disposition_html = final_disposition.replace("<", "&lt;").replace(">", "&gt;")
    return f'<p style="margin: 5px 0 0 0;"><strong>Final Disposition:</strong> {disposition_html}</p>'


def build_summary_email_html(summaries: List[CaseSummary], email_date: date) -> str:
    """
    Build the HTML body of the summary email.

    Structure:
    - Cases that could not be summarized (links only, for manual review)
    - IP Cases
      - Precedential (with full details: judges, summary, holdings)
      - Non-Precedential (with full details: judges, summary, holdings)
    - Non-IP Cases (just case cite with link)
    - Summary Dispositions
    """
    # Cases that failed are listed separately so they are never silently dropped
    failed = [s for s in summaries if s.error]
    summaries = [s for s in summaries if not s.error]

    # Categorize cases - Summary dispositions are separate from IP/non-IP
    rule_42b_dismissals = [s for s in summaries if s.is_rule_42b_dismissal]
    rule_36_affirmances = [s for s in summaries if s.is_rule_36_affirmance]

    # Filter out summary dispositions for substantive categorization
    substantive = [s for s in summaries if not s.is_rule_42b_dismissal and not s.is_rule_36_affirmance]

    # IP cases = any case where at least one IP type is true
    ip_cases = [s for s in substantive if s.is_patent_case or s.is_copyright_case or s.is_trade_secret_case or s.is_trademark_case]
    ip_precedential = [s for s in ip_cases if s.is_precedential]
    ip_non_precedential = [s for s in ip_cases if not s.is_precedential]

    # Non-IP cases = no IP type is true
    non_ip = [s for s in substantive if not s.is_patent_case and not s.is_copyright_case and not s.is_trade_secret_case and not s.is_trademark_case]
    
    # Build HTML email body with inline styles (Outlook-friendly)
    date_str = email_date.strftime("%B %d, %Y")
    
    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; font-size: 14px; color: #333; max-width: 800px;">
    <h1 style="color: #1a1a1a; border-bottom: 3px solid #0066cc; padding-bottom: 10px;">{date_str} Federal Circuit Opinions</h1>
    <p style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 15px 0; font-size: 12px; color: #856404;">
        <strong>Note:</strong> The below summaries have been generated using AI and have not been reviewed by an attorney. Read and analyze any case before relying on it. Do not rely on the AI summary.
    </p>
"""

    # Failed Cases Section
    if failed:
        html_body += f"""
    <h2 style="color: #0066cc; margin-top: 30px; border-bottom: 2px solid #0066cc; padding-bottom: 5px;">NOT SUMMARIZED ({len(failed)})</h2>
    <p style="margin: 10px 0 10px 20px; color: #666;">These were released today but could not be fetched and summarized automatically. Links are provided for direct review.</p>
"""
        for summary in failed:
            case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
            link = summary.pdf_url or summary.landing_url
            case_link = f'<a href="{link}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>' if link else case_name
            html_body += f"""
    <p style="margin: 10px 0 10px 20px;">{case_link} <em style="color: #666;">({summary.error[0].lower() + summary.error[1:]})</em></p>
"""

    # IP Cases Section
    if ip_precedential or ip_non_precedential:
        html_body += """
    <h2 style="color: #0066cc; margin-top: 30px; border-bottom: 2px solid #0066cc; padding-bottom: 5px;">IP CASES</h2>
"""

        # Precedential IP Cases
        if ip_precedential:
            html_body += f"""
    <h3 style="color: #006600; margin-top: 20px; margin-left: 15px;">Precedential ({len(ip_precedential)})</h3>
"""
            for summary in ip_precedential:
                case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
                # Make case name a clickable link if PDF URL is available
                if summary.pdf_url:
                    case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
                else:
                    case_link = case_name

                # Format IP category line
                ip_types_text = _format_ip_types(summary)
                ip_types_line = f"<p style=\"margin: 5px 0;\"><strong>IP Category:</strong> {ip_types_text}</p>"

                # Format patent law issues (only if this is a patent case)
                issues_line = ""
                if summary.is_patent_case and summary.patent_law_issues:
                    issues_text = ", ".join(summary.patent_law_issues)
                    issues_line = f"<p style=\"margin: 5px 0;\"><strong>Issues Addressed:</strong> {issues_text}</p>"

                # Format judges with author underlined
                judges_html = _format_judges_html(summary.panel_judges, summary.author_judge)
                judges_line = f"<p style=\"margin: 5px 0;\"><strong>Judges:</strong> {judges_html}</p>" if judges_html else ""
                disposition_line = _format_final_disposition_line(summary.final_disposition)

                # Use structured summary if available, fallback to summary_text
                summary_content = summary.case_summary if summary.case_summary else summary.summary_text
                summary_html = _markdown_to_html(summary_content)

                # Format holdings
                holdings_html = ""
                if summary.major_holdings:
                    holdings_html = f"<p style=\"margin: 10px 0 5px 0;\"><strong>Major Holdings:</strong></p><div style=\"line-height: 1.6; margin-left: 15px;\">{_markdown_to_html(summary.major_holdings)}</div>"

                html_body += f"""
    <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0 15px 20px; border-left: 4px solid #006600;">
        <p style="font-size: 16px; font-weight: bold; margin: 0 0 10px 0;">{case_link}</p>
        {ip_types_line}
        {issues_line}
        {judges_line}
        {disposition_line}
        <p style="margin: 10px 0 5px 0;"><strong>Summary:</strong></p>
        <div style="line-height: 1.6; margin-left: 15px;">{summary_html}</div>
        {holdings_html}
    </div>
"""

        # Non-Precedential IP Cases
        if ip_non_precedential:
            html_body += f"""
    <h3 style="color: #666; margin-top: 20px; margin-left: 15px;">Non-Precedential ({len(ip_non_precedential)})</h3>
"""
            for summary in ip_non_precedential:
                case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
                # Make case name a clickable link if PDF URL is available
                if summary.pdf_url:
                    case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
                else:
                    case_link = case_name

                # Format IP category line
                ip_types_text = _format_ip_types(summary)
                ip_types_line = f"<p style=\"margin: 5px 0;\"><strong>IP Category:</strong> {ip_types_text}</p>"

                # Format patent law issues (only if this is a patent case)
                issues_line = ""
                if summary.is_patent_case and summary.patent_law_issues:
                    issues_text = ", ".join(summary.patent_law_issues)
                    issues_line = f"<p style=\"margin: 5px 0;\"><strong>Issues Addressed:</strong> {issues_text}</p>"

                # Format judges with author underlined
                judges_html = _format_judges_html(summary.panel_judges, summary.author_judge)
                judges_line = f"<p style=\"margin: 5px 0;\"><strong>Judges:</strong> {judges_html}</p>" if judges_html else ""
                disposition_line = _format_final_disposition_line(summary.final_disposition)

                # Use structured summary if available, fallback to summary_text
                summary_content = summary.case_summary if summary.case_summary else summary.summary_text
                summary_html = _markdown_to_html(summary_content)

                # Format holdings
                holdings_html = ""
                if summary.major_holdings:
                    holdings_html = f"<p style=\"margin: 10px 0 5px 0;\"><strong>Major Holdings:</strong></p><div style=\"line-height: 1.6; margin-left: 15px;\">{_markdown_to_html(summary.major_holdings)}</div>"

                html_body += f"""
    <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0 15px 20px; border-left: 4px solid #999;">
        <p style="font-size: 16px; font-weight: bold; margin: 0 0 10px 0;">{case_link}</p>
        {ip_types_line}
        {issues_line}
        {judges_line}
        {disposition_line}
        <p style="margin: 10px 0 5px 0;"><strong>Summary:</strong></p>
        <div style="line-height: 1.6; margin-left: 15px;">{summary_html}</div>
        {holdings_html}
    </div>
"""
    
    # Non-IP Cases Section
    if non_ip:
        html_body += f"""
    <h2 style="color: #0066cc; margin-top: 30px; border-bottom: 2px solid #0066cc; padding-bottom: 5px;">NON-IP CASES ({len(non_ip)})</h2>
"""
        for summary in non_ip:
            case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
            # Make case name a clickable link if PDF URL is available
            if summary.pdf_url:
                case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
            else:
                case_link = case_name
            html_body += f"""
    <p style="margin: 10px 0 10px 20px;">{case_link}</p>
"""
    
    # Summary Dispositions Section
    if rule_42b_dismissals or rule_36_affirmances:
        total_dispositions = len(rule_42b_dismissals) + len(rule_36_affirmances)
        html_body += f"""
    <h2 style="color: #999; margin-top: 30px; border-bottom: 2px solid #999; padding-bottom: 5px;">SUMMARY DISPOSITIONS ({total_dispositions})</h2>
"""
        
        # Rule 42(b) Dismissals sub-section
        if rule_42b_dismissals:
            html_body += f"""
    <h3 style="color: #999; margin-top: 15px; margin-left: 15px;">Rule 42(b) Dismissals ({len(rule_42b_dismissals)})</h3>
"""
            for summary in rule_42b_dismissals:
                case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
                # Make case name a clickable link if PDF URL is available
                if summary.pdf_url:
                    case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
                else:
                    case_link = case_name
                html_body += f"""
    <p style="margin: 10px 0 10px 35px; color: #666;">{case_link} <em>(dismissed)</em></p>
"""
        
        # Rule 36 Affirmances sub-section
        if rule_36_affirmances:
            html_body += f"""
    <h3 style="color: #999; margin-top: 15px; margin-left: 15px;">Rule 36 Affirmances ({len(rule_36_affirmances)})</h3>
"""
            for summary in rule_36_affirmances:
                case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
                # Make case name a clickable link if PDF URL is available
                if summary.pdf_url:
                    case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
                else:
                    case_link = case_name
                html_body += f"""
    <p style="margin: 10px 0 10px 35px; color: #666;">{case_link} <em>(affirmed)</em></p>
"""
    
    html_body += """
</body>
</html>
"""
    return html_body


def send_summary_email(
    service,
    to_email: str | List[str],
    summaries: List[CaseSummary],
    email_date: date,
    bcc_email: str | List[str] | None = None,
    subject_prefix: str = "",
) -> bool:
    """
    Send an email with all case summaries in structured format.

    Args:
        service: Authenticated Gmail API service
        to_email: Email address(es) to send to (string or list of strings)
        summaries: List of CaseSummary objects
        email_date: Date of the opinions
        bcc_email: Email address(es) to BCC (string or list of strings, optional)
        subject_prefix: Text prepended to the subject (e.g. "[TEST] ")

    Returns:
        True if email sent successfully, False otherwise
    """
    if not summaries:
        print("[warn] No summaries to send")
        return False

    # Normalize to_email to a list
    to_emails = [to_email] if isinstance(to_email, str) else to_email

    html_body = build_summary_email_html(summaries, email_date)
    date_str = email_date.strftime("%B %d, %Y")

    # Create email message with HTML
    message = MIMEText(html_body, "html")
    message["to"] = ", ".join(to_emails)
    if bcc_email:
        bcc_emails = [bcc_email] if isinstance(bcc_email, str) else bcc_email
        message["bcc"] = ", ".join(bcc_emails)
    message["subject"] = f"{subject_prefix}Federal Circuit Opinions - {date_str}"
    
    # Encode message
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    
    # Build log message
    log_parts = [f"to: {', '.join(to_emails)}"]
    if bcc_email:
        log_parts.append(f"bcc: {', '.join(bcc_emails)}")
    
    try:
        service.users().messages().send(
            userId="me",
            body={"raw": raw_message}
        ).execute()
        print(f"[ok] Summary email sent ({'; '.join(log_parts)})")
        return True
    except Exception as e:
        print(f"[error] Failed to send email: {e}")
        return False


# Written after the summary email for a date is successfully sent; its presence
# means cron runs for that date skip processing (use --force to override).
SENT_MARKER_NAME = ".sent"


@contextmanager
def _run_lock(lock_path: Path):
    """Yield True if we acquired an exclusive, non-blocking lock (so overlapping cron runs exit)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as fp:
        try:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def _summarize_case(doc, date_summary_dir: Path, prompt_file: str | None) -> CaseSummary:
    """Summarize one downloaded case. Failures are recorded on the returned CaseSummary's `error`."""
    summary = CaseSummary(
        case_name=doc.case_name,
        is_precedential=doc.is_precedential,
        summary_text="",
        opinion_date=doc.opinion_date,
        case_number=doc.case_number,
        pdf_url=doc.pdf_url,
        landing_url=doc.landing_url,
    )
    if doc.error:
        summary.error = doc.error
        return summary

    pdf_path = doc.pdf_path
    try:
        # Extract text from PDF
        print(f"[info] Extracting text from: {pdf_path.name}")
        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            summary.error = "Unable to read the text of the opinion"
            return summary

        # Summarize
        print(f"[info] Summarizing: {pdf_path.name}")
        result = summarize_text(text, prompt_file=prompt_file, opinion_date=doc.opinion_date, case_number=doc.case_number)
    except Exception as e:
        print(f"[error] Summarization failed for {doc.case_name}: {type(e).__name__}: {e}")
        summary.error = "Unable to generate the AI summary"
        return summary

    # Generate output filename
    if result.opinion_date and result.case_number:
        formatted_date = result.opinion_date.replace("-", ".")
        filename = f"{formatted_date}_{result.case_number}.txt"
    else:
        filename = f"{pdf_path.stem}-summary.txt"

    # Organize into date/precedential/non-precedential subdirectories
    subdir = date_summary_dir / ("precedential" if doc.is_precedential else "non-precedential")
    subdir.mkdir(parents=True, exist_ok=True)

    # Write summary
    summary_path = subdir / filename
    summary_path.write_text(result.combined_summary, encoding="utf-8")
    print(f"[ok] Wrote summary: {summary_path}")

    summary.summary_text = result.combined_summary
    summary.opinion_date = result.opinion_date
    summary.case_number = result.case_number
    summary.is_patent_case = result.is_patent_case
    summary.is_copyright_case = result.is_copyright_case
    summary.is_trade_secret_case = result.is_trade_secret_case
    summary.is_trademark_case = result.is_trademark_case
    summary.panel_judges = result.panel_judges
    summary.author_judge = result.author_judge
    summary.case_summary = result.case_summary
    summary.major_holdings = result.major_holdings
    summary.final_disposition = result.final_disposition
    summary.is_rule_42b_dismissal = result.is_rule_42b_dismissal
    summary.is_rule_36_affirmance = result.is_rule_36_affirmance
    summary.patent_law_issues = result.patent_law_issues
    return summary


def process_court_emails(
    service,
    sender: str | List[str] = "uscourts@updates.uscourts.gov",
    search_date: date | None = None,
    pdf_dir: Path | None = None,
    summary_dir: Path | None = None,
    prompt_file: str | None = None,
    email_to: str | List[str] | None = None,
    email_bcc: str | List[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    subject_prefix: str = "",
) -> int:
    """
    Complete workflow: search emails, download PDFs, generate summaries, and send email.
    
    Args:
        service: Authenticated Gmail API service
        sender: Email address(es) to search for (string or list of strings)
        search_date: Date to search for (defaults to today)
        pdf_dir: Directory to save downloaded PDFs (defaults to "pdfs")
        summary_dir: Directory to save summaries (defaults to "summaries")
        prompt_file: Optional path to prompt file for summarization
        email_to: Email address(es) to send summary to (string or list of strings, optional)
        email_bcc: Email address(es) to BCC (string or list of strings, optional)
        force: Reprocess and resend even if the email for this date was already sent (default: False)
        dry_run: Write the email HTML to a preview file instead of sending it (default: False)
        subject_prefix: Text prepended to the email subject (e.g. "[TEST] ")
        
    Returns:
        Number of PDFs successfully summarized
    """
    if pdf_dir is None:
        pdf_dir = Path("pdfs")
    if summary_dir is None:
        summary_dir = Path("summaries")
    if search_date is None:
        search_date = date.today()

    with _run_lock(summary_dir / ".run.lock") as acquired:
        if not acquired:
            print("[info] Another run is already in progress; exiting")
            return 0
        return _process_court_emails_locked(
            service, sender, search_date, pdf_dir, summary_dir, prompt_file,
            email_to, email_bcc, force, dry_run, subject_prefix,
        )


def _process_court_emails_locked(
    service,
    sender: str | List[str],
    search_date: date,
    pdf_dir: Path,
    summary_dir: Path,
    prompt_file: str | None,
    email_to: str | List[str] | None,
    email_bcc: str | List[str] | None,
    force: bool,
    dry_run: bool,
    subject_prefix: str,
) -> int:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Create date-specific summary directory (YYYY-MM-DD)
    date_str = search_date.strftime("%Y-%m-%d")
    date_summary_dir = summary_dir / date_str
    sent_marker = date_summary_dir / SENT_MARKER_NAME
    
    # Idempotency: skip if the summary email for this date has already gone out
    if not force and sent_marker.exists():
        print(f"[info] Summary email already sent for {date_str} ({sent_marker})")
        print(f"[info] Skipping processing to avoid duplicates (use --force to override)")
        return 0
    
    date_summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize sender to a list
    senders = [sender] if isinstance(sender, str) else sender
    
    # Search for emails from all senders
    emails = []
    for s in senders:
        emails.extend(search_emails(service, s, search_date))
    
    if not emails:
        return 0
    
    summaries = []  # Collect summaries (and failures) for email
    seen_doc_ids = set()  # The same case can appear in more than one email
    
    # Process each email
    for i, email in enumerate(emails, 1):
        print(f"\n[info] Processing email {i}/{len(emails)}")
        
        # Extract email body
        body = get_email_body(email)
        
        if not body:
            print("[warn] Empty email body, skipping")
            continue
        
        # Extract uscourts.gov links, plus each case's name/status as listed in the email
        links = extract_links_from_text(body)
        entries = parse_email_entries(body)
        
        if not links:
            print("[warn] No uscourts.gov links found in email")
            continue
        
        print(f"[info] Found {len(links)} link(s) in email")
        
        # Process each link
        for link in links:
            doc_id = doc_id_from_url(link)
            if doc_id and doc_id in seen_doc_ids:
                print(f"[info] Skipping duplicate link: {link}")
                continue
            seen_doc_ids.add(doc_id)
            
            doc = process_uscourts_link(link, pdf_dir, entries.get(doc_id))
            summary = _summarize_case(doc, date_summary_dir, prompt_file)
            if summary.error:
                print(f"[error] {summary.case_name}: {summary.error}")
            summaries.append(summary)
    
    failures = [s for s in summaries if s.error]
    pdf_count = len(summaries) - len(failures)
    if failures:
        print(f"\n[warn] {len(failures)} case(s) could not be summarized and will be listed for manual review:")
        for s in failures:
            print(f"[warn]   {s.case_name}: {s.error}")
    
    if not summaries:
        return pdf_count
    
    if not (email_to or email_bcc):
        if dry_run:
            email_to = []
        else:
            print("\n[info] No recipients given; not sending an email")
            return pdf_count
    
    if email_to:
        actual_to = email_to
        actual_bcc = email_bcc
    else:
        # BCC-only: use sending account as To so recipients appear only on BCC
        actual_to = SENDER_EMAIL
        actual_bcc = email_bcc
    email_list = [actual_to] if isinstance(actual_to, str) else actual_to
    bcc_list = [actual_bcc] if isinstance(actual_bcc, str) else (actual_bcc or [])
    log_parts = [f"to: {', '.join(email_list)}"]
    if bcc_list:
        log_parts.append(f"bcc: {', '.join(bcc_list)}")
    
    if dry_run:
        preview_path = date_summary_dir / "email_preview.html"
        preview_path.write_text(build_summary_email_html(summaries, search_date), encoding="utf-8")
        print(f"\n[dry-run] Email NOT sent (would have sent {'; '.join(log_parts)})")
        print(f"[dry-run] Preview written to: {preview_path}")
        return pdf_count
    
    print(f"\n[info] Sending summary email ({'; '.join(log_parts)})...")
    if send_summary_email(service, actual_to, summaries, search_date, bcc_email=actual_bcc, subject_prefix=subject_prefix):
        sent_marker.write_text(json.dumps({
            "sent_at": datetime.now().isoformat(timespec="seconds"),
            "to": email_list,
            "bcc": bcc_list,
            "cases": len(summaries),
            "failed_cases": [s.case_name for s in failures],
        }, indent=2), encoding="utf-8")
    
    return pdf_count
