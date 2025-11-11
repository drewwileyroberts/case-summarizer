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
    is_rule_42b_dismissal: bool = False  # Fed. R. App. P. 42(b) dismissal (no opinion content)
    patent_law_issues: List[str] = None  # List of patent law issues addressed (for patent cases only)
    
    def __post_init__(self):
        # Initialize lists to empty if None
        if self.panel_judges is None:
            self.panel_judges = []
        if self.patent_law_issues is None:
            self.patent_law_issues = []


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
1. Is this a Fed. R. App. P. 42(b) dismissal? These are very short dismissal orders with no substantive opinion content - just a notice that the case was dismissed. (true/false)
2. Is this a patent-related case? (true/false) - Skip if question 1 is true
3. What are the main patent law issues addressed in this case? Select up to 5 of the most important issues from the list below. Use ONLY the exact strings provided. Return empty array [] if not a patent case or if Rule 42(b) dismissal.

Possible patent law issues (use exact strings, select up to 5 most important):
- patent-eligible subject matter (§ 101)
- printed matter doctrine
- natural law or abstract idea (Alice/Mayo) (§ 101)
- anticipation (§ 102)
- obviousness (§ 103)
- obviousness-type double patenting
- priority or entitlement to priority (§ 119 or § 120)
- written description (§ 112(a))
- enablement (§ 112(a))
- definiteness (§ 112(b))
- utility (§ 101)
- best mode (§ 112(a))
- public use or on-sale bar (§ 102)
- experimental use exception
- derivation or inventorship (§ 116 or § 256)
- joint inventorship (§ 116)
- claim construction
- means-plus-function interpretation (§ 112(f))
- claim scope disavowal or disclaimer
- prosecution-history estoppel
- intrinsic vs extrinsic evidence
- claim preamble limitation
- claim differentiation
- literal infringement (§ 271(a))
- doctrine of equivalents
- induced infringement (§ 271(b))
- contributory infringement (§ 271(c))
- divided or joint infringement (§ 271(a))
- importation or product-by-process (§ 271(g))
- willful infringement
- indirect infringement knowledge or intent (§ 271(b) or (c))
- extraterritoriality (§ 271(f))
- inequitable conduct
- unclean hands or litigation misconduct
- prosecution laches
- equitable estoppel
- intervening rights (§ 252 or § 307(b))
- patent exhaustion or first-sale doctrine
- prior user rights (§ 273)
- lost profits
- reasonable royalty (§ 284)
- apportionment (§ 284)
- entire market value rule (§ 284)
- enhanced damages (§ 284)
- injunctions (§ 283)
- ongoing royalties (§ 283 or § 284)
- attorneys' fees (§ 285)
- pre- or post-judgment interest
- subject-matter jurisdiction
- personal jurisdiction or venue (§ 1400(b))
- standing
- real party in interest or privity
- post-judgment motions (Rule 54/59/60)
- cross-appeals or appellate jurisdiction
- standard of review
- inter partes review (IPR) (§ 311–§ 319)
- post-grant review (PGR) or covered business method (CBM) (§ 321–§ 329)
- estoppel (§ 315(e))
- institution decisions or SAS issues (§ 314)
- obviousness in PTAB context (§ 103)
- real-party-in-interest challenges (§ 312(a)(2))
- director review or rehearing (§ 6 or § 141)
- reexamination or reissue (§ 251–§ 257)
- design patent ornamentality or functionality (§ 171)
- design patent anticipation or obviousness (§ 102 or § 103)
- article of manufacture definition (§ 171)
- plant patent requirements (§ 161)
- ITC § 337 actions (19 U.S.C. § 1337)
- government-use (§ 1498)
- export or import infringement (§ 271(f) or (g))
- assignment or ownership disputes (§ 261)
- licenses or contractual interpretation
- covenant not to sue
- FRAND or standard-essential patents
- attorney-client privilege or waiver
- sanctions (Rule 11)
- claim preclusion or res judicata
- reissue/reexamination effect on litigation (§ 251–§ 257)
- constitutional issues

4. Which judges were on the panel? Return as an array of judge last names. If it's Per Curiam, return ["Per Curiam"]. If unsigned, return ["Unsigned"]. Return empty array [] if this is a Rule 42(b) dismissal.
5. Which judge authored the opinion? Return the last name of the authoring judge, or "Per Curiam" or "Unsigned" if applicable. Return null if you cannot determine or if this is a Rule 42(b) dismissal.
6. Provide a 4-5 sentence summary of the case. Focus on the key facts, legal issues, and outcome. Return empty string "" if this is a Rule 42(b) dismissal.
7. What are the major holdings from this case? A major holding can be either: (a) a broad legal principle or rule that could apply to future cases, or (b) the case's disposition with its substantive reasoning (e.g., "Affirmed that the patent was invalid for lack of written description"). Many cases have only 1-2 major holdings, and some have none. Return 0-3 holdings only. Be very selective—do not include bare outcomes without reasoning (e.g., "Plaintiff failed to prove infringement") or generic procedural statements (e.g., "The district court was correct"). Format each holding on a new line like: "1. [holding text]\\n2. [holding text]\\n3. [holding text]". Return empty string "" if this is a Rule 42(b) dismissal or if no major holdings.

Return ONLY valid JSON in this exact format (no additional text):
{
  "is_rule_42b_dismissal": true or false,
  "is_patent_case": true or false,
  "patent_law_issues": ["issue1", "issue2"] or [],
  "panel_judges": ["Judge1", "Judge2", "Judge3"] or [],
  "author_judge": "Judge1" or null,
  "case_summary": "4-5 sentence summary here" or "",
  "major_holdings": "1. [holding text]\\n2. [holding text]\\n3. [holding text]" or ""
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
            'is_rule_42b_dismissal': bool(data.get('is_rule_42b_dismissal', False)),
            'is_patent_case': bool(data.get('is_patent_case', False)),
            'patent_law_issues': data.get('patent_law_issues', []),
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
            'is_rule_42b_dismissal': False,
            'is_patent_case': False,
            'patent_law_issues': [],
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
        is_rule_42b_dismissal=structured_info['is_rule_42b_dismissal'],
        patent_law_issues=structured_info['patent_law_issues'],
    )



