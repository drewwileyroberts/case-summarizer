from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .uscourts_scraper import extract_links_from_text, process_uscourts_link
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


def send_summary_email(
    service,
    to_email: str | List[str],
    summaries: List[CaseSummary],
    email_date: date,
) -> bool:
    """
    Send an email with all case summaries.
    
    Args:
        service: Authenticated Gmail API service
        to_email: Email address(es) to send to (string or list of strings)
        summaries: List of CaseSummary objects
        email_date: Date of the opinions
        
    Returns:
        True if email sent successfully, False otherwise
    """
    if not summaries:
        print("[warn] No summaries to send")
        return False
    
    # Normalize to_email to a list
    to_emails = [to_email] if isinstance(to_email, str) else to_email
    
    # Sort: precedential first, then non-precedential
    precedential = [s for s in summaries if s.is_precedential]
    non_precedential = [s for s in summaries if not s.is_precedential]
    
    # Build HTML email body with inline styles (Outlook-friendly)
    date_str = email_date.strftime("%B %d, %Y")
    
    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; font-size: 14px; color: #333; max-width: 800px;">
    <h1 style="color: #1a1a1a; border-bottom: 3px solid #0066cc; padding-bottom: 10px;">{date_str} Federal Circuit Opinions</h1>
"""
    
    # Add precedential opinions
    if precedential:
        html_body += f"""
    <h2 style="color: #0066cc; margin-top: 30px; border-bottom: 2px solid #ccc; padding-bottom: 5px;">PRECEDENTIAL OPINIONS ({len(precedential)})</h2>
"""
        for summary in precedential:
            case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
            summary_html = _markdown_to_html(summary.summary_text)
            # Make case name a clickable link if PDF URL is available
            if summary.pdf_url:
                case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
            else:
                case_link = case_name
            html_body += f"""
    <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-left: 4px solid #0066cc;">
        <p style="font-size: 16px; font-weight: bold; margin: 0 0 10px 0;">{case_link} <span style="color: #006600;">(Precedential)</span></p>
        <div style="line-height: 1.6;">{summary_html}</div>
    </div>
"""
    
    # Add non-precedential opinions
    if non_precedential:
        html_body += f"""
    <h2 style="color: #0066cc; margin-top: 30px; border-bottom: 2px solid #ccc; padding-bottom: 5px;">NON-PRECEDENTIAL OPINIONS ({len(non_precedential)})</h2>
"""
        for summary in non_precedential:
            case_name = summary.case_name.replace("<", "&lt;").replace(">", "&gt;")
            summary_html = _markdown_to_html(summary.summary_text)
            # Make case name a clickable link if PDF URL is available
            if summary.pdf_url:
                case_link = f'<a href="{summary.pdf_url}" style="color: #0066cc; text-decoration: underline;">{case_name}</a>'
            else:
                case_link = case_name
            html_body += f"""
    <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0; border-left: 4px solid #999;">
        <p style="font-size: 16px; font-weight: bold; margin: 0 0 10px 0;">{case_link} <span style="color: #666;">(Non-Precedential)</span></p>
        <div style="line-height: 1.6;">{summary_html}</div>
    </div>
"""
    
    html_body += """
</body>
</html>
"""
    
    # Create email message with HTML
    message = MIMEText(html_body, "html")
    message["to"] = ", ".join(to_emails)
    message["subject"] = f"Federal Circuit Opinions - {date_str}"
    
    # Encode message
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    
    try:
        service.users().messages().send(
            userId="me",
            body={"raw": raw_message}
        ).execute()
        print(f"[ok] Summary email sent to: {', '.join(to_emails)}")
        return True
    except Exception as e:
        print(f"[error] Failed to send email: {e}")
        return False


def process_court_emails(
    service,
    sender: str | List[str] = "uscourts@updates.uscourts.gov",
    search_date: date | None = None,
    pdf_dir: Path | None = None,
    summary_dir: Path | None = None,
    prompt_file: str | None = None,
    email_to: str | List[str] | None = None,
    force: bool = False,
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
        force: Force reprocessing even if summaries already exist (default: False)
        
    Returns:
        Number of PDFs processed
    """
    if pdf_dir is None:
        pdf_dir = Path("pdfs")
    if summary_dir is None:
        summary_dir = Path("summaries")
    if search_date is None:
        search_date = date.today()
    
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Create date-specific summary directory (YYYY-MM-DD)
    date_str = search_date.strftime("%Y-%m-%d")
    date_summary_dir = summary_dir / date_str
    
    # Check if we've already processed opinions for this date (idempotency)
    if not force and date_summary_dir.exists():
        # Check if there are any summary files
        existing_summaries = list(date_summary_dir.rglob("*.txt"))
        if existing_summaries:
            print(f"[info] Summaries already exist for {date_str} ({len(existing_summaries)} files)")
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
    
    pdf_count = 0
    summaries = []  # Collect summaries for email
    
    # Process each email
    for i, email in enumerate(emails, 1):
        print(f"\n[info] Processing email {i}/{len(emails)}")
        
        # Extract email body
        body = get_email_body(email)
        
        if not body:
            print("[warn] Empty email body, skipping")
            continue
        
        # Extract uscourts.gov links
        links = extract_links_from_text(body)
        
        if not links:
            print("[warn] No uscourts.gov links found in email")
            continue
        
        print(f"[info] Found {len(links)} link(s) in email")
        
        # Process each link
        for link in links:
            # Download PDF from landing page and get metadata
            pdf_path, is_precedential, case_name, pdf_url = process_uscourts_link(link, pdf_dir)
            
            if not pdf_path:
                continue
            
            # Extract text from PDF
            print(f"[info] Extracting text from: {pdf_path.name}")
            text = extract_text_from_pdf(str(pdf_path))
            
            if not text.strip():
                print(f"[warn] No text extracted from: {pdf_path.name}")
                continue
            
            # Summarize
            print(f"[info] Summarizing: {pdf_path.name}")
            result = summarize_text(text, prompt_file=prompt_file)
            
            # Generate output filename
            if result.opinion_date and result.case_number:
                formatted_date = result.opinion_date.replace("-", ".")
                filename = f"{formatted_date}_{result.case_number}.txt"
            else:
                filename = f"{pdf_path.stem}-summary.txt"
            
            # Organize into date/precedential/non-precedential subdirectories
            if is_precedential:
                subdir = date_summary_dir / "precedential"
            else:
                subdir = date_summary_dir / "non-precedential"
            
            subdir.mkdir(parents=True, exist_ok=True)
            
            # Write summary
            summary_path = subdir / filename
            summary_path.write_text(result.combined_summary, encoding="utf-8")
            
            print(f"[ok] Wrote summary: {summary_path}")
            pdf_count += 1
            
            # Add to summaries list for email
            if case_name:
                summaries.append(CaseSummary(
                    case_name=case_name,
                    is_precedential=is_precedential,
                    summary_text=result.combined_summary,
                    opinion_date=result.opinion_date,
                    case_number=result.case_number,
                    pdf_url=pdf_url,
                ))
    
    # Send summary email if email_to is provided
    if email_to and summaries:
        email_list = [email_to] if isinstance(email_to, str) else email_to
        print(f"\n[info] Sending summary email to {', '.join(email_list)}...")
        send_summary_email(service, email_to, summaries, search_date)
    
    return pdf_count

