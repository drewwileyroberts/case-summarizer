from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup


# Seconds to wait between attempts for transient network failures (timeouts, 5xx).
RETRY_DELAYS = (5, 15)

# Landing-page URLs end in e.g. "-24-2353-opinion-9-21-2026_2758411/", which maps
# one-to-one onto the PDF ".../opinions-orders/24-2353.OPINION.9-21-2026_2758411.pdf".
LANDING_URL_PATTERN = re.compile(
    r'-(?P<case_number>\d{2,}-\d+)-(?P<doc_type>[a-z0-9_]+)-'
    r'(?P<month>\d{1,2})-(?P<day>\d{1,2})-(?P<year>\d{4})_(?P<doc_id>\d+)/?$',
    re.IGNORECASE,
)

# Court email entries look like:
# "24-2353: BERKELEY*IEOR v. W.W. GRAINGER INC. [OPINION], Nonprecedential [ https://... ]"
EMAIL_ENTRY_PATTERN = re.compile(
    r'(?P<case_number>\d{2,}-\d+):\s*(?P<title>[^\[\]]+?)\s*\[(?P<doc_type>[^\[\]]+)\],\s*'
    r'(?P<status>Precedential|Non-?precedential)\s*\[\s*(?P<url>https?://[^\s\]]+)\s*\]',
    re.IGNORECASE,
)


@dataclass
class EmailCaseEntry:
    """A case as listed in the court's notification email."""
    case_number: str
    case_name: str
    is_precedential: bool
    url: str


@dataclass
class CaseDocument:
    """Everything known about one case link after locating and downloading its PDF."""
    landing_url: str
    case_name: str | None = None
    case_number: str | None = None
    opinion_date: str | None = None
    is_precedential: bool | None = None
    pdf_url: str | None = None
    pdf_path: Path | None = None
    error: str | None = None


def _get_with_retries(url: str, timeout: int) -> requests.Response:
    """GET a URL, retrying timeouts/connection errors/5xx. 4xx errors are raised immediately."""
    for attempt, delay in enumerate((*RETRY_DELAYS, None), 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if delay is None or (status is not None and 400 <= status < 500):
                raise
            print(f"[warn] Attempt {attempt} failed for {url}: {e}; retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def doc_id_from_url(url: str) -> str | None:
    """Return the court's numeric document id (the digits after the final '_'), used to match links."""
    match = re.search(r'_(\d+)/?$', url)
    return match.group(1) if match else None


def parse_landing_url(landing_url: str) -> Tuple[str, str, str] | None:
    """Derive (pdf_url, case_number, opinion_date YYYY-MM-DD) from a landing-page URL, or None."""
    match = LANDING_URL_PATTERN.search(urlparse(landing_url).path)
    if not match:
        return None
    g = match.groupdict()
    pdf_name = f"{g['case_number']}.{g['doc_type'].upper()}.{g['month']}-{g['day']}-{g['year']}_{g['doc_id']}.pdf"
    pdf_url = urljoin(landing_url, f"/opinions-orders/{pdf_name}")
    opinion_date = f"{g['year']}-{g['month'].zfill(2)}-{g['day'].zfill(2)}"
    return pdf_url, g['case_number'], opinion_date


def parse_email_entries(text: str) -> Dict[str, EmailCaseEntry]:
    """Parse the court email's case listings, keyed by document id."""
    entries: Dict[str, EmailCaseEntry] = {}
    for match in EMAIL_ENTRY_PATTERN.finditer(text):
        doc_id = doc_id_from_url(match.group('url'))
        if not doc_id:
            continue
        case_number = match.group('case_number')
        entries[doc_id] = EmailCaseEntry(
            case_number=case_number,
            case_name=_clean_case_name(f"{case_number}: {match.group('title')}"),
            is_precedential=not match.group('status').lower().startswith('non'),
            url=match.group('url'),
        )
    return entries


def _clean_case_name(case_text: str) -> str:
    """Normalize CAFC case-title text for display."""
    case_name = re.sub(r',?\s*(Precedential|Non-Precedential|Nonprecedential)\s*$', '', case_text, flags=re.IGNORECASE)
    case_name = re.sub(r'\s*\[(OPINION|ORDER)\]\s*', ' ', case_name).strip()
    return re.sub(r'\s+', ' ', case_name).strip()


def _extract_appeal_number(page_text: str) -> str | None:
    appeal_num_match = re.search(r'Appeal Number:\s*(\d{2,}-\d+)', page_text)
    if appeal_num_match:
        return appeal_num_match.group(1)
    return None


def _extract_precedential_status(page_text: str) -> bool | None:
    appeal_block_match = re.search(
        r'Appeal Number:\s*\d{2,}-\d+.*?(?:To see more opinions|Innovation Center|$)',
        page_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not appeal_block_match:
        return None

    appeal_block = appeal_block_match.group(0)
    if re.search(r'\bNon-?precedential\b', appeal_block, flags=re.IGNORECASE):
        return False
    if re.search(r'\bPrecedential\b', appeal_block, flags=re.IGNORECASE):
        return True
    return None


def _find_opinions_pdf_link(soup: BeautifulSoup, landing_url: str) -> tuple[str | None, str | None]:
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/opinions-orders/" in href and href.endswith(".pdf"):
            pdf_url = urljoin(landing_url, href)
            pdf_label = link.get_text(" ", strip=True)
            pdf_label = re.sub(r'\s*\(?pdf\)?\s*$', '', pdf_label, flags=re.IGNORECASE).strip()
            return pdf_url, pdf_label or None
    return None, None


def _extract_case_name_from_posted_pdf_link(soup: BeautifulSoup, page_text: str, landing_url: str) -> tuple[str | None, str | None, str | None]:
    pdf_url, pdf_label = _find_opinions_pdf_link(soup, landing_url)
    case_number = _extract_appeal_number(page_text)
    if pdf_label and case_number:
        return _clean_case_name(f"{case_number}: {pdf_label}"), case_number, pdf_url
    return None, case_number, pdf_url


def _extract_case_name_from_valid_heading(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    for heading in soup.find_all("h1"):
        case_text = heading.get_text(" ", strip=True)
        case_num_match = re.match(r'^(\d{2,}-\d+):', case_text)
        if case_num_match:
            return _clean_case_name(case_text), case_num_match.group(1)
    return None, None


def extract_metadata_from_landing_page(landing_url: str) -> Tuple[str | None, bool, str | None, str | None, str | None]:
    """
    Extracts the PDF URL, precedential status, case name, opinion date, and case number from a uscourts.gov landing page.
    
    Args:
        landing_url: URL of the landing page (e.g., the link from the email)
        
    Returns:
        Tuple of (pdf_url, is_precedential, case_name, opinion_date, case_number) where:
        - pdf_url: Direct PDF URL, or None if not found
        - is_precedential: True if precedential, False otherwise
        - case_name: Case name from the page title, or None if not found
        - opinion_date: Opinion date in YYYY-MM-DD format, or None if not found
        - case_number: Case/appeal number, or None if not found
    """
    try:
        response = _get_with_retries(landing_url, timeout=30)
    except requests.RequestException as e:
        print(f"[error] Failed to fetch landing page {landing_url}: {e}")
        return None, False, None, None, None
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    page_text = soup.get_text(" ", strip=True)
    case_name, case_number, pdf_url = _extract_case_name_from_posted_pdf_link(soup, page_text, landing_url)
    
    # Fallback: look for any PDF link
    if not pdf_url:
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf"):
                pdf_url = urljoin(landing_url, href)
                print(f"[info] Found PDF link (fallback): {pdf_url}")
                break
    
    if not pdf_url:
        print(f"[warn] No PDF link found on landing page: {landing_url}")
    
    # Fallback: only trust headings that look like CAFC case titles.
    if not case_name:
        case_name, heading_case_number = _extract_case_name_from_valid_heading(soup)
        if heading_case_number:
            case_number = heading_case_number
    
    # Extract precedential status
    # Look for "Precedential" or "Non-Precedential" text on the page
    is_precedential = _extract_precedential_status(page_text)

    if is_precedential is None and "Precedential" in page_text:
        # Check if it's actually "Non-Precedential" or just "Precedential"
        if "Non-Precedential" not in page_text and "Nonprecedential" not in page_text:
            is_precedential = True
        else:
            is_precedential = False
    elif is_precedential is None:
        is_precedential = False
    
    # Extract opinion date
    opinion_date = None
    
    # Try to find "Appeal Number: XXX" pattern in page text (fallback for case number)
    if not case_number:
        case_number = _extract_appeal_number(page_text)
    
    # Look for date in various formats on the page
    # First try the URL itself (e.g., /11-05-2025-25-1750-...)
    url_date_match = re.search(r'/(\d{1,2})-(\d{1,2})-(\d{4})-', landing_url)
    if url_date_match:
        month, day, year = url_date_match.groups()
        try:
            opinion_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except ValueError:
            pass
    
    # If not in URL, try to find date text like "November 5, 2025" or similar
    if not opinion_date:
        # Look for common date patterns
        date_patterns = [
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, page_text)
            if date_match:
                try:
                    if pattern.startswith(r'(January'):
                        # Month name format
                        month_name, day, year = date_match.groups()
                        date_obj = datetime.strptime(f"{month_name} {day} {year}", "%B %d %Y")
                        opinion_date = date_obj.strftime("%Y-%m-%d")
                        break
                    elif '/' in pattern:
                        # MM/DD/YYYY format
                        month, day, year = date_match.groups()
                        opinion_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        break
                    else:
                        # YYYY-MM-DD format
                        year, month, day = date_match.groups()
                        opinion_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        break
                except (ValueError, AttributeError):
                    continue
    
    return pdf_url, is_precedential, case_name, opinion_date, case_number


def extract_pdf_url_from_landing_page(landing_url: str) -> str | None:
    """
    Extracts the PDF URL from a uscourts.gov landing page.
    
    DEPRECATED: Use extract_metadata_from_landing_page() instead.
    
    Args:
        landing_url: URL of the landing page (e.g., the link from the email)
        
    Returns:
        The direct PDF URL, or None if not found
    """
    pdf_url, _, _, _, _ = extract_metadata_from_landing_page(landing_url)
    return pdf_url


def download_pdf(pdf_url: str, output_dir: Path) -> Path | None:
    """
    Downloads a PDF from the given URL.
    
    Args:
        pdf_url: Direct URL to the PDF file
        output_dir: Directory to save the downloaded PDF
        
    Returns:
        Path to the downloaded PDF file, or None if download failed
    """
    try:
        response = _get_with_retries(pdf_url, timeout=60)
    except requests.RequestException as e:
        print(f"[error] Failed to download PDF {pdf_url}: {e}")
        return None
    
    # Extract filename from URL
    parsed = urlparse(pdf_url)
    filename = Path(parsed.path).name
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the PDF
    output_path = output_dir / filename
    output_path.write_bytes(response.content)
    
    print(f"[ok] Downloaded PDF: {output_path}")
    return output_path


def process_uscourts_link(landing_url: str, output_dir: Path, email_entry: EmailCaseEntry | None = None) -> CaseDocument:
    """
    Complete workflow: determine the case metadata and PDF URL, then download the PDF.

    Metadata comes from the court email entry and the landing-page URL itself; the
    landing page is only fetched when those are incomplete, since the court's site
    can refuse some landing pages (e.g. 403 for titles containing '*').

    Args:
        landing_url: URL of the landing page from the email
        output_dir: Directory to save the downloaded PDF
        email_entry: The case's listing parsed from the court email, if available

    Returns:
        CaseDocument; on failure, `error` describes what went wrong and `pdf_path` is None
    """
    print(f"[info] Processing uscourts link: {landing_url}")

    doc = CaseDocument(landing_url=landing_url)
    url_info = parse_landing_url(landing_url)
    if url_info:
        doc.pdf_url, doc.case_number, doc.opinion_date = url_info
    if email_entry:
        doc.case_name = email_entry.case_name
        doc.case_number = doc.case_number or email_entry.case_number
        doc.is_precedential = email_entry.is_precedential

    if not (doc.pdf_url and doc.case_name and doc.is_precedential is not None):
        print("[info] Email/URL metadata incomplete; reading landing page")
        pdf_url, is_precedential, case_name, opinion_date, case_number = extract_metadata_from_landing_page(landing_url)
        doc.pdf_url = doc.pdf_url or pdf_url
        doc.case_name = doc.case_name or case_name
        doc.case_number = doc.case_number or case_number
        doc.opinion_date = doc.opinion_date or opinion_date
        if doc.is_precedential is None:
            doc.is_precedential = is_precedential

    if not doc.case_name:
        doc.case_name = doc.case_number or landing_url
    if doc.is_precedential is None:
        doc.is_precedential = False

    if not doc.pdf_url:
        doc.error = "Unable to locate the opinion on the court website"
        return doc

    print(f"[info] Found PDF URL: {doc.pdf_url}")
    print(f"[info] Case: {doc.case_name}")
    print(f"[info] Case Number: {doc.case_number}")
    print(f"[info] Opinion Date: {doc.opinion_date}")
    print(f"[info] Precedential: {'Yes' if doc.is_precedential else 'No'}")

    doc.pdf_path = download_pdf(doc.pdf_url, output_dir)
    if not doc.pdf_path:
        doc.error = "Unable to fetch the opinion from the court website"
    return doc


def extract_links_from_text(text: str) -> List[str]:
    """
    Extracts uscourts.gov URLs from text (e.g., email body).
    Handles GovDelivery link tracking wrappers.
    
    Args:
        text: Text containing URLs (email body, etc.)
        
    Returns:
        List of uscourts.gov URLs found
    """
    urls = []
    
    # First, look for GovDelivery wrapped links
    # Pattern: https://links-X.govdelivery.com/CL0/https:%2F%2Fwww.cafc.uscourts.gov%2F...
    govdelivery_pattern = r'https?://links[^\s]*?\.govdelivery\.com/CL0/(https?[^\s<>"\')/]*)'
    for match in re.finditer(govdelivery_pattern, text, re.IGNORECASE):
        encoded_url = match.group(1)
        # URL decode to get the actual URL
        decoded_url = unquote(encoded_url)
        # Check if it's a uscourts.gov URL
        if 'uscourts.gov' in decoded_url.lower():
            urls.append(decoded_url)
    
    # Also look for direct uscourts.gov URLs (in case some emails don't use wrapper)
    direct_pattern = r'https?://(?:www\.)?[a-z0-9]+\.uscourts\.gov/[^\s<>"\')]*'
    direct_urls = re.findall(direct_pattern, text, re.IGNORECASE)
    urls.extend(direct_urls)
    
    # Remove duplicates (keeping email order) and filter to only landing page URLs (not direct PDF links)
    unique_urls = list(dict.fromkeys(urls))
    landing_urls = [url for url in unique_urls if not url.endswith('.pdf')]
    
    return landing_urls

