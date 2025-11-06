# Federal Circuit Cron Job Setup

## Overview
Automated daily checking for Federal Circuit decision emails from 12:00-12:14 PM on weekdays.

## What Was Set Up

### 1. Multi-Email Support
Modified the code to send summaries to multiple email addresses:
- `drewwileyroberts@gmail.com`
- `rich.lowry@lw.com`

Files modified:
- `summarizer/gmail_cli.py` - Now accepts multiple `--email-to` addresses
- `summarizer/gmail_checker.py` - Sends emails to all recipients

### 2. Shell Wrapper Script
Created: `/home/pi/case-summarizer/run_daily_check.sh`

This script:
- Activates the Python virtual environment
- Runs the gmail checker with email notifications
- Logs all output to `logs/cron.log`

### 3. Cron Schedule
**When it runs:** Every 2 minutes from 12:00-12:14 PM, Monday-Friday only

**Times:** 12:00, 12:02, 12:04, 12:06, 12:08, 12:10, 12:12, 12:14 PM

**Why multiple runs?** Federal Circuit emails sometimes arrive a few minutes late. The built-in idempotency check ensures that once summaries are generated for a day, subsequent runs that same day will skip processing.

## Checking Status

### View cron schedule
```bash
crontab -l
```

### View execution logs
```bash
tail -f /home/pi/case-summarizer/logs/cron.log
```

### Test the script manually
```bash
cd /home/pi/case-summarizer
./run_daily_check.sh
```

### Force reprocessing (if needed)
```bash
cd /home/pi/case-summarizer
source .venv/bin/activate
python3 -m summarizer.gmail_cli --email-to drewwileyroberts@gmail.com rich.lowry@lw.com --force
```

## How It Works

1. **12:00 PM** - Cron runs the first check
2. **If email found** - Downloads PDFs, generates summaries, sends email to both recipients
3. **If no email yet** - Logs that no new opinions were found
4. **12:02, 12:04, etc.** - Subsequent runs check again
5. **Once processed** - All remaining runs that day will skip (idempotency)
6. **Next weekday** - Process starts fresh at 12:00 PM

## Troubleshooting

### Emails not being sent
- Check Gmail API authentication: `ls -la token.json`
- Check logs: `cat logs/cron.log`
- Test manually: `./run_daily_check.sh`

### Cron not running
- Verify cron service: `systemctl status cron`
- Check crontab: `crontab -l`
- Check system logs: `grep CRON /var/log/syslog`

### Want to change email recipients
Edit the script: `nano run_daily_check.sh`
Change the `--email-to` line to add/remove addresses

### Want to change schedule
Edit crontab: `crontab -e`
Modify the time values (currently `0,2,4,6,8,10,12,14 12 * * 1-5`)

