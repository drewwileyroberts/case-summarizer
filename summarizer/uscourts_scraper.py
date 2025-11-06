from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup


def extract_metadata_from_landing_page(landing_url: str) -> Tuple[str | None, bool, str | None]:
    """
    Extracts the PDF URL, precedential status, and case name from a uscourts.gov landing page.
    
    Args:
        landing_url: URL of the landing page (e.g., the link from the email)
        
    Returns:
        Tuple of (pdf_url, is_precedential, case_name) where:
        - pdf_url: Direct PDF URL, or None if not found
        - is_precedential: True if precedential, False otherwise
        - case_name: Case name from the page title, or None if not found
    """
    try:
        response = requests.get(landing_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[error] Failed to fetch landing page {landing_url}: {e}")
        return None, False, None
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract PDF URL
    pdf_url = None
    
    # Look for links that point to PDF files in the opinions-orders path
    for link in soup.find_all("a", href=True):
        href = link["href"]
        
        # Check if it's a PDF link in the opinions-orders directory
        if "/opinions-orders/" in href and href.endswith(".pdf"):
            # Convert relative URL to absolute if needed
            pdf_url = urljoin(landing_url, href)
            break
    
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
    
    # Extract case name from the page heading (h1)
    case_name = None
    h1 = soup.find("h1")
    if h1:
        # Example: "23-1446: FOCUS PRODUCTS GROUP INTERNATIONAL, LLC v. KARTRI SALES CO., INC. [OPINION], Precedential"
        case_text = h1.get_text().strip()
        # Remove the precedential/non-precedential suffix
        case_name = re.sub(r',?\s*(Precedential|Non-Precedential|Nonprecedential)\s*$', '', case_text, flags=re.IGNORECASE)
        # Clean up extra whitespace and remove "[OPINION]" or similar tags
        case_name = re.sub(r'\s*\[(OPINION|ORDER)\]\s*', ' ', case_name).strip()
    
    # Extract precedential status
    # Look for "Precedential" or "Non-Precedential" text on the page
    page_text = soup.get_text()
    is_precedential = False
    
    if "Precedential" in page_text:
        # Check if it's actually "Non-Precedential" or just "Precedential"
        if "Non-Precedential" not in page_text and "Nonprecedential" not in page_text:
            is_precedential = True
    
    return pdf_url, is_precedential, case_name


def extract_pdf_url_from_landing_page(landing_url: str) -> str | None:
    """
    Extracts the PDF URL from a uscourts.gov landing page.
    
    DEPRECATED: Use extract_metadata_from_landing_page() instead.
    
    Args:
        landing_url: URL of the landing page (e.g., the link from the email)
        
    Returns:
        The direct PDF URL, or None if not found
    """
    pdf_url, _, _ = extract_metadata_from_landing_page(landing_url)
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
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
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


def process_uscourts_link(landing_url: str, output_dir: Path) -> Tuple[Path | None, bool, str | None, str | None]:
    """
    Complete workflow: extract PDF URL from landing page and download it.
    
    Args:
        landing_url: URL of the landing page from the email
        output_dir: Directory to save the downloaded PDF
        
    Returns:
        Tuple of (pdf_path, is_precedential, case_name, pdf_url) where:
        - pdf_path: Path to the downloaded PDF file, or None if failed
        - is_precedential: True if precedential, False otherwise
        - case_name: Case name from the landing page, or None if not found
        - pdf_url: Direct URL to the PDF file, or None if not found
    """
    print(f"[info] Processing uscourts link: {landing_url}")
    
    pdf_url, is_precedential, case_name = extract_metadata_from_landing_page(landing_url)
    if not pdf_url:
        return None, False, None, None
    
    print(f"[info] Found PDF URL: {pdf_url}")
    print(f"[info] Case: {case_name}")
    print(f"[info] Precedential: {'Yes' if is_precedential else 'No'}")
    
    pdf_path = download_pdf(pdf_url, output_dir)
    return pdf_path, is_precedential, case_name, pdf_url


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
    
    # Remove duplicates and filter to only landing page URLs (not direct PDF links)
    unique_urls = list(set(urls))
    landing_urls = [url for url in unique_urls if not url.endswith('.pdf')]
    
    return landing_urls

