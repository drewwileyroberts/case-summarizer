from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


DEFAULT_MODEL = "gpt-5"


@dataclass
class SummarizationResult:
    combined_summary: str
    opinion_date: Optional[str] = None  # Format: YYYY-MM-DD
    case_number: Optional[str] = None
    # Structured fields from decision tree
    is_patent_case: bool = False
    is_copyright_case: bool = False  # Copyright infringement, ownership, fair use, DMCA, etc.
    is_trade_secret_case: bool = False  # Trade secret misappropriation, DTSA, state trade secret laws
    is_trademark_case: bool = False  # Trademark infringement, dilution, Lanham Act, etc.
    panel_judges: List[str] = None  # List of judge names or ["Per Curiam"] or ["Unsigned"]
    author_judge: Optional[str] = None  # The judge who authored the opinion
    case_summary: Optional[str] = None  # 4-5 sentence summary
    major_holdings: Optional[str] = None  # Major holdings from the case
    final_disposition: Optional[str] = None  # Final appellate disposition
    is_rule_42b_dismissal: bool = False  # Fed. R. App. P. 42(b) dismissal (no opinion content)
    is_rule_36_affirmance: bool = False  # Fed. R. App. P. Rule 36 affirmance (minimal opinion content)
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
    # GPT-5 models don't support custom temperature
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0.2

    response = client.chat.completions.create(**kwargs)
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
    6. What is the final disposition?
    
    Returns:
        dict with keys: is_patent_case, panel_judges, author_judge,
        case_summary, major_holdings
    """
    sample = text
    
    structured_prompt = """You are analyzing a legal case document. Please answer the following questions and return your response in valid JSON format.

Questions:
1a. Is this a Fed. R. App. P. 42(b) dismissal? These are very short dismissal orders with no substantive opinion content - just a notice that the case was dismissed. (true/false)
1b. Is this a Fed. Cir. R. 36 summary affirmance? Answer true ONLY if ALL of the following are met: (1) The document explicitly cites "Fed. Cir. R. 36" or "Rule 36", (2) The entire substantive content is essentially just "AFFIRMED. See Fed. Cir. R. 36." (1-2 sentences max), (3) There is NO Background section, NO Discussion section, and NO substantive legal analysis. If the opinion contains any legal reasoning, case citations with analysis, or discussion of issues - even if it's per curiam and affirms - answer false. (true/false)
2. Is this a patent-related case? (true/false) - involves patent claims, infringement, validity, USPTO proceedings, etc. Skip if question 1a or 1b is true.
3. Is this a copyright case? (true/false) - involves copyright infringement, ownership, fair use, DMCA, etc. Skip if question 1a or 1b is true.
4. Is this a trade secret case? (true/false) - involves trade secret misappropriation, DTSA, state trade secret laws, etc. Skip if question 1a or 1b is true.
5. Is this a trademark case? (true/false) - involves trademark infringement, dilution, Lanham Act, etc. Skip if question 1a or 1b is true.
6. What are the main patent law issues addressed in this case? Select up to 5 of the most important issues from the list below. Use ONLY the exact strings provided. Return empty array [] if not a patent case or if question 1a or 1b is true.

Possible patent law issues (use exact strings, select up to 5 most important):
- patent-eligible subject matter (Section 101)
- printed matter doctrine
- natural law or abstract idea (Alice/Mayo) (Section 101)
- anticipation (Section 102)
- obviousness (Section 103)
- obviousness-type double patenting
- priority or entitlement to priority (Section 119 or Section 120)
- written description (Section 112(a))
- enablement (Section 112(a))
- definiteness (Section 112(b))
- utility (Section 101)
- best mode (Section 112(a))
- public use or on-sale bar (Section 102)
- experimental use exception
- derivation or inventorship (Section 116 or Section 256)
- joint inventorship (Section 116)
- claim construction
- means-plus-function interpretation (Section 112(f))
- claim scope disavowal or disclaimer
- prosecution-history estoppel
- intrinsic vs extrinsic evidence
- claim preamble limitation
- claim differentiation
- literal infringement (Section 271(a))
- doctrine of equivalents
- induced infringement (Section 271(b))
- contributory infringement (Section 271(c))
- divided or joint infringement (Section 271(a))
- importation or product-by-process (Section 271(g))
- willful infringement
- indirect infringement knowledge or intent (Section 271(b) or (c))
- extraterritoriality (Section 271(f))
- inequitable conduct
- unclean hands or litigation misconduct
- prosecution laches
- equitable estoppel
- intervening rights (Section 252 or Section 307(b))
- patent exhaustion or first-sale doctrine
- prior user rights (Section 273)
- lost profits
- reasonable royalty (Section 284)
- apportionment (Section 284)
- entire market value rule (Section 284)
- enhanced damages (Section 284)
- injunctions (Section 283)
- ongoing royalties (Section 283 or Section 284)
- attorneys' fees (Section 285)
- pre- or post-judgment interest
- subject-matter jurisdiction
- personal jurisdiction or venue (Section 1400(b))
- standing
- real party in interest or privity
- post-judgment motions (Rule 54/59/60)
- cross-appeals or appellate jurisdiction
- standard of review
- inter partes review (IPR) (Sections 311-319)
- post-grant review (PGR) or covered business method (CBM) (Sections 321-329)
- estoppel (Section 315(e))
- institution decisions or SAS issues (Section 314)
- obviousness in PTAB context (Section 103)
- real-party-in-interest challenges (Section 312(a)(2))
- director review or rehearing (Section 6 or Section 141)
- reexamination or reissue (Sections 251-257)
- design patent ornamentality or functionality (Section 171)
- design patent anticipation or obviousness (Section 102 or Section 103)
- article of manufacture definition (Section 171)
- plant patent requirements (Section 161)
- ITC Section 337 actions (19 U.S.C. Section 1337)
- government-use (Section 1498)
- export or import infringement (Section 271(f) or (g))
- assignment or ownership disputes (Section 261)
- licenses or contractual interpretation
- covenant not to sue
- FRAND or standard-essential patents
- attorney-client privilege or waiver
- sanctions (Rule 11)
- claim preclusion or res judicata
- reissue/reexamination effect on litigation (Sections 251-257)
- constitutional issues

7. Which judges were on the panel? Return as an array of judge last names. If it's Per Curiam, return ["Per Curiam"]. If unsigned, return ["Unsigned"]. Return empty array [] if question 1a or 1b is true.
8. Which judge authored the opinion? Return the last name of the authoring judge, or "Per Curiam" or "Unsigned" if applicable. Return null if you cannot determine or if question 1a or 1b is true.
9. Provide a 4-5 sentence summary of the case. Focus on the key facts, legal issues, and outcome. Return empty string "" if question 1a or 1b is true.
10. Write 0-3 headnote-style summaries of the court's key rulings (1-2 is typical; 0 and 3 are rare). Each headnote should capture a specific legal conclusion the court reached on a disputed issue - the kind of point a practitioner would highlight when telling a colleague about this case. Keep each under 25 words. Only include affirmative rulings. Do NOT include: routine costs/fees allocations, standard procedural language, or issues the court declined to decide. Format on new lines: "1. [text]\\n2. [text]\\n3. [text]". Return empty string "" if question 1a or 1b is true or if none.
11. What is the court's final disposition? Use one of the exact strings below when possible. If none fit, use "Other: [short disposition]". Return empty string "" if question 1a or 1b is true.

Final disposition options:
- Affirmed
- Reversed
- Vacated
- Remanded
- Dismissed
- Affirmed in Part and Reversed in Part
- Affirmed in Part, Reversed in Part, and Remanded
- Affirmed in Part, Vacated in Part, and Remanded
- Reversed and Remanded
- Vacated and Remanded
- Dismissed in Part
- Other: [short disposition]

Return ONLY valid JSON in this exact format (no additional text):
{
  "is_rule_42b_dismissal": true or false,
  "is_rule_36_affirmance": true or false,
  "is_patent_case": true or false,
  "is_copyright_case": true or false,
  "is_trade_secret_case": true or false,
  "is_trademark_case": true or false,
  "patent_law_issues": ["issue1", "issue2"] or [],
  "panel_judges": ["Judge1", "Judge2", "Judge3"] or [],
  "author_judge": "Judge1" or null,
  "case_summary": "4-5 sentence summary here" or "",
  "major_holdings": "1. [holding text]\\n2. [holding text]\\n3. [holding text]" or "",
  "final_disposition": "Affirmed" or "Reversed and Remanded" or "Other: [short disposition]" or ""
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
            'is_rule_36_affirmance': bool(data.get('is_rule_36_affirmance', False)),
            'is_patent_case': bool(data.get('is_patent_case', False)),
            'is_copyright_case': bool(data.get('is_copyright_case', False)),
            'is_trade_secret_case': bool(data.get('is_trade_secret_case', False)),
            'is_trademark_case': bool(data.get('is_trademark_case', False)),
            'patent_law_issues': data.get('patent_law_issues', []),
            'panel_judges': data.get('panel_judges', []),
            'author_judge': data.get('author_judge'),
            'case_summary': data.get('case_summary', ''),
            'major_holdings': data.get('major_holdings', ''),
            'final_disposition': str(data.get('final_disposition') or '').strip(),
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[warn] Raw response: {response[:200]}...")
        # Raise rather than returning blank defaults, which would misfile the case as non-IP
        raise ValueError(f"Failed to parse structured JSON response: {e}") from e


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
    
    # Extract structured info (includes case_summary, so no separate summarization call needed)
    structured_info = _extract_structured_info(client, model, text)

    return SummarizationResult(
        combined_summary=structured_info['case_summary'],  # Use structured summary as fallback
        opinion_date=opinion_date,
        case_number=case_number,
        is_patent_case=structured_info['is_patent_case'],
        is_copyright_case=structured_info['is_copyright_case'],
        is_trade_secret_case=structured_info['is_trade_secret_case'],
        is_trademark_case=structured_info['is_trademark_case'],
        panel_judges=structured_info['panel_judges'],
        author_judge=structured_info['author_judge'],
        case_summary=structured_info['case_summary'],
        major_holdings=structured_info['major_holdings'],
        final_disposition=structured_info['final_disposition'],
        is_rule_42b_dismissal=structured_info['is_rule_42b_dismissal'],
        is_rule_36_affirmance=structured_info['is_rule_36_affirmance'],
        patent_law_issues=structured_info['patent_law_issues'],
    )



