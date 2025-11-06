# Gmail API Setup Guide

## Step-by-Step Setup

### 1. Enable Gmail API in Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. In the left sidebar, go to **APIs & Services** > **Library**
4. Search for "Gmail API"
5. Click on it and click **Enable**

### 2. Create OAuth 2.0 Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **+ CREATE CREDENTIALS** > **OAuth client ID**
3. If prompted, configure the OAuth consent screen:
   - Choose **External** (unless you have a Google Workspace)
   - Fill in the app name (e.g., "Case Summarizer")
   - Add your email as a test user
   - Skip optional fields and save
4. Back to **Create OAuth client ID**:
   - Application type: **Desktop app**
   - Name: "Case Summarizer Desktop"
   - Click **Create**
5. Download the credentials file
6. Rename it to `credentials.json` and place it in your project root

### 3. First Run - Authorization

On the first run, the app will:
1. Open your browser
2. Ask you to sign in to your Google account
3. Show a warning that the app isn't verified (this is normal for personal apps)
   - Click **Advanced** > **Go to [Your App Name] (unsafe)**
4. Grant permissions (readonly access to Gmail)
5. The app will save `token.json` for future use

### 4. Test It Out

```bash
# Check today's emails
python3 -m summarizer.gmail_cli

# Check a specific date
python3 -m summarizer.gmail_cli --date 2025-10-01

# Use a custom prompt
python3 -m summarizer.gmail_cli --prompt-file my_prompt.txt
```

## Troubleshooting

**"Credentials file not found"**
- Make sure `credentials.json` is in the project root directory
- You can specify a different path with `--credentials /path/to/file.json`

**"Access blocked: This app's request is invalid"**
- Make sure you added your email as a test user in the OAuth consent screen
- For the OAuth consent screen, choose "External" user type

**"Token has been expired or revoked"**
- Delete `token.json` and re-authenticate

**No emails found**
- Verify the date format is `YYYY-MM-DD`
- Check that you actually received an email from `uscourts@updates.uscourts.gov` on that date
- Try searching in Gmail directly to confirm the email exists

## Security Notes

- The app only requests **readonly** access to Gmail (no sending, deleting, or modifying emails)
- Your credentials are stored locally in `token.json` (not shared with anyone)
- Both `credentials.json` and `token.json` are excluded from git via `.gitignore`
- You can revoke access anytime at https://myaccount.google.com/permissions

