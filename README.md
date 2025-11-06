## Case Summarizer

An automated system for monitoring court opinions via Gmail and generating AI-powered summaries.

### Prerequisites
- Python 3.9+
- An OpenAI API key (`OPENAI_API_KEY` environment variable)
- Gmail API credentials (for automated email checking)

### Setup
1) Create and activate a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```
4) Set up Gmail API credentials (for automated checking):
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Gmail API
   - Create OAuth 2.0 credentials (Desktop app)
   - Download the credentials and save as `credentials.json` in the project root
   - On first run, you'll be prompted to authorize the app

### Usage

#### Option 1: Automated Gmail Checking (Recommended)

Check your Gmail for court opinions, automatically download and summarize them, and send a summary email:

```bash
python3 -m summarizer.gmail_cli --date 2025-10-02 --prompt-file prompt.example.txt --email-to drewwileyroberts@gmail.com
```

This will:
1. Search for emails from `uscourts@updates.uscourts.gov` on the specified date (defaults to today)
2. Extract court opinion links from the emails
3. Download the PDFs from the court website
4. Detect case names and whether each opinion is precedential or non-precedential
5. Generate AI summaries for each opinion
6. Save summaries organized by date and type:
   - `summaries/2025-10-01/precedential/2025.10.01_23-1446.txt`
   - `summaries/2025-10-01/non-precedential/2025.10.01_24-1121.txt`
   - `summaries/2025-10-02/precedential/2025.10.02_23-1447.txt`
7. Send a formatted email with all summaries (precedential first, then non-precedential)

Options:
- `--date YYYY-MM-DD` - Date to search (defaults to today)
- `--sender EMAIL` - Email sender to search for
- `--email-to EMAIL` - Email address to send summary to (optional)
- `--pdf-dir DIR` - Where to save PDFs (default: `pdfs/`)
- `--summary-dir DIR` - Where to save summaries (default: `summaries/`)
- `--prompt-file FILE` - Custom prompt for summarization
- `--credentials FILE` - Path to credentials.json (default: `credentials.json`)
- `--token FILE` - Path to token.json (default: `token.json`)

#### Option 2: Direct PDF Summarization

Summarize PDFs directly without Gmail:

```bash
python3 -m summarizer.cli --pdf /path/to/file.pdf --prompt-file prompt.example.txt
```

Options:
- `--pdf FILE [FILE ...]` - One or more PDF files to summarize
- `--prompt-file FILE` - Custom prompt file
- `--prompt TEXT` - Inline prompt (overrides file)
- `--model MODEL` - OpenAI model (default: `gpt-4o`)
- `--output-dir DIR` - Output directory (default: `summaries/`)

### Features

- **Automatic Metadata Extraction**: Extracts opinion date, case number, and case names directly from landing pages (no GPT calls needed)
- **Structured Information Extraction**: Automatically extracts:
  - Patent case status
  - Precedential/non-precedential status
  - Panel judges and authoring judge
  - 4-5 sentence case summary
  - Major legal holdings
- **Smart Filename Generation**: Saves summaries as `YYYY.MM.DD_CASE-NUMBER.txt`
- **Precedential Organization**: Automatically detects and organizes opinions into `precedential/` and `non-precedential/` folders
- **Email Summaries**: Sends formatted email digest with all opinions (precedential first)
- **Gmail Integration**: Monitors email for new court opinions and sends summaries via Gmail API
- **Web Scraping**: Automatically downloads PDFs from uscourts.gov landing pages and extracts metadata
- **Efficient Processing**: Processes full documents in single API calls for speed
- **Customizable Prompts**: Use your own summarization prompts

### Notes
- Default model is `gpt-4o` for optimal balance of speed and legal accuracy.
- The system processes full documents in single API calls (no chunking) for faster results.
- Metadata (date, case number) is scraped from landing pages to minimize API calls.
- The Gmail integration uses OAuth 2.0 with readonly scope for checking emails and send scope for sending summaries.
- If you change the Gmail API scopes, delete `token.json` to re-authenticate with new permissions.

