from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


DEFAULT_MODEL = "gpt-4o-mini" 
DEFAULT_MAX_CHARS_PER_CHUNK = 9000


@dataclass
class SummarizationResult:
    combined_summary: str
    chunk_summaries: List[str]
    opinion_date: Optional[str] = None  # Format: YYYY-MM-DD
    case_number: Optional[str] = None


def _chunk_text(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    # naive chunking by characters to keep implementation simple
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _load_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt is not None:
        return prompt
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    # Check for default_prompt.txt in the project root
    default_prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "default_prompt.txt")
    if os.path.exists(default_prompt_path):
        with open(default_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback to hardcoded prompt if default_prompt.txt doesn't exist
    return (
        "Summarize the following text clearly and concisely for a layperson. "
        "Include key points, dates, names, and outcomes."
    )


def _create_client() -> OpenAI:
    # relies on OPENAI_API_KEY env var
    return OpenAI()


def _call_model(client: OpenAI, model: str, system_prompt: str, user_text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _extract_metadata(client: OpenAI, model: str, text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract opinion date and case number from the PDF text.
    
    Returns:
        tuple of (opinion_date, case_number) where opinion_date is in YYYY-MM-DD format
    """
    # Use first 3000 chars which should contain the header info
    sample = text[:3000]
    
    metadata_prompt = """Extract the following information from this legal document:
1. Opinion date (the date the opinion was issued/filed, not argued)
2. Case number

Return ONLY in this exact format:
DATE: YYYY-MM-DD
CASE: [case number]

If you cannot find either field, use "UNKNOWN" for that field."""
    
    response = _call_model(client, model, metadata_prompt, sample)
    
    # Parse the response
    opinion_date = None
    case_number = None
    
    for line in response.strip().split("\n"):
        if line.startswith("DATE:"):
            date_val = line.split(":", 1)[1].strip()
            if date_val != "UNKNOWN":
                opinion_date = date_val
        elif line.startswith("CASE:"):
            case_val = line.split(":", 1)[1].strip()
            if case_val != "UNKNOWN":
                case_number = case_val
    
    return opinion_date, case_number


def summarize_text(
    text: str,
    *,
    prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> SummarizationResult:
    client = _create_client()
    system_prompt = _load_prompt(prompt, prompt_file)

    chunks = _chunk_text(text, max_chars_per_chunk)
    if not chunks:
        return SummarizationResult(combined_summary="", chunk_summaries=[])

    # Extract metadata first
    opinion_date, case_number = _extract_metadata(client, model, text)

    chunk_summaries: List[str] = []
    for chunk in chunks:
        chunk_summary = _call_model(client, model, system_prompt, chunk)
        chunk_summaries.append(chunk_summary.strip())

    # final combine pass
    combined_prompt = (
        system_prompt
        + "\n\nYou will now receive multiple partial summaries. Combine them into a single, concise summary without repeating yourself."
    )
    combined_input = "\n\n".join(f"Part {i+1}:\n{cs}" for i, cs in enumerate(chunk_summaries))
    combined_summary = _call_model(client, model, combined_prompt, combined_input).strip()

    return SummarizationResult(
        combined_summary=combined_summary,
        chunk_summaries=chunk_summaries,
        opinion_date=opinion_date,
        case_number=case_number,
    )



