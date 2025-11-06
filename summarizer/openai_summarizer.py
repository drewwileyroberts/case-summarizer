from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"


@dataclass
class SummarizationResult:
    combined_summary: str
    opinion_date: Optional[str] = None  # Format: YYYY-MM-DD
    case_number: Optional[str] = None
    # Structured fields from decision tree
    is_patent_case: bool = False
    panel_judges: List[str] = None  # List of judge names or ["Per Curiam"] or ["Unsigned"]
    author_judge: Optional[str] = None  # The judge who authored the opinion
    case_summary: Optional[str] = None  # 4-5 sentence summary
    major_holdings: Optional[str] = None  # Major holdings from the case
    
    def __post_init__(self):
        # Initialize panel_judges to empty list if None
        if self.panel_judges is None:
            self.panel_judges = []


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


def _extract_structured_info(client: OpenAI, model: str, text: str) -> dict:
    """Extract structured case information using JSON format.
    
    Asks the key questions in the decision tree:
    1. Is this a patent related case?
    2. Which judges were on the panel?
    3. Which judge authored the opinion?
    4. What is a 4-5 sentence summary?
    5. What are the major holdings?
    
    Returns:
        dict with keys: is_patent_case, panel_judges, author_judge,
        case_summary, major_holdings
    """
    # Use a larger sample for better context (up to 15000 chars)
    sample = text[:15000]
    
    structured_prompt = """You are analyzing a legal case document. Please answer the following questions and return your response in valid JSON format.

Questions:
1. Is this a patent-related case? (true/false)
2. Which judges were on the panel? Return as an array of judge last names. If it's Per Curiam, return ["Per Curiam"]. If unsigned, return ["Unsigned"].
3. Which judge authored the opinion? Return the last name of the authoring judge, or "Per Curiam" or "Unsigned" if applicable. Return null if you cannot determine.
4. Provide a 4-5 sentence summary of the case. Focus on the key facts, legal issues, and outcome.
5. What are the major holdings or rules from this case? Provide 1 to 4 (only the amount needed) concise bullet points highlighting only the most important legal principles. Be brief and to the point. Format each holding on a new line like: "1. [holding text]\\n2. [holding text]\\n3. [holding text]"

Return ONLY valid JSON in this exact format (no additional text):
{
  "is_patent_case": true or false,
  "panel_judges": ["Judge1", "Judge2", "Judge3"],
  "author_judge": "Judge1" or null,
  "case_summary": "4-5 sentence summary here",
  "major_holdings": "1. [holding text]\\n2. [holding text]\\n3. [holding text]"
}"""
    
    response = _call_model(client, model, structured_prompt, sample)
    
    # Parse JSON response
    try:
        # Try to extract JSON if there's extra text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
        else:
            # Fallback to parsing the whole response
            data = json.loads(response)
        
        # Validate and provide defaults
        return {
            'is_patent_case': bool(data.get('is_patent_case', False)),
            'panel_judges': data.get('panel_judges', []),
            'author_judge': data.get('author_judge'),
            'case_summary': data.get('case_summary', ''),
            'major_holdings': data.get('major_holdings', ''),
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[warn] Failed to parse structured JSON response: {e}")
        print(f"[warn] Raw response: {response[:200]}...")
        # Return default values
        return {
            'is_patent_case': False,
            'panel_judges': [],
            'author_judge': None,
            'case_summary': '',
            'major_holdings': '',
        }


def summarize_text(
    text: str,
    *,
    prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    opinion_date: Optional[str] = None,
    case_number: Optional[str] = None,
) -> SummarizationResult:
    client = _create_client()
    
    if not text.strip():
        return SummarizationResult(combined_summary="")

    # Extract metadata only if not provided
    if opinion_date is None or case_number is None:
        print("[info] Extracting metadata from PDF text using GPT...")
        extracted_date, extracted_number = _extract_metadata(client, model, text)
        if opinion_date is None:
            opinion_date = extracted_date
        if case_number is None:
            case_number = extracted_number
    else:
        print(f"[info] Using scraped metadata: date={opinion_date}, case={case_number}")
    
    # Extract structured info
    structured_info = _extract_structured_info(client, model, text)

    # Generate summary from full text in one call
    system_prompt = _load_prompt(prompt, prompt_file)
    combined_summary = _call_model(client, model, system_prompt, text).strip()

    return SummarizationResult(
        combined_summary=combined_summary,
        opinion_date=opinion_date,
        case_number=case_number,
        is_patent_case=structured_info['is_patent_case'],
        panel_judges=structured_info['panel_judges'],
        author_judge=structured_info['author_judge'],
        case_summary=structured_info['case_summary'],
        major_holdings=structured_info['major_holdings'],
    )



