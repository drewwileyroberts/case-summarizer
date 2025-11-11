#!/bin/bash

# Federal Circuit Daily Check Script
# This script is designed to be run by cron to check for new Federal Circuit decisions

# Exit on error
set -e

# Change to project directory
cd /home/pi/case-summarizer

# Activate virtual environment
source .venv/bin/activate

# Log the start time
echo "========================================" >> logs/cron.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Federal Circuit check..." >> logs/cron.log

# Run the gmail checker with multiple email recipients
python3 -m summarizer.gmail_cli \
    --email-to richard.lowry@lw.com drew.roberts@lw.com andrew.kerrick@lw.com sarah.propst@lw.com\
    2>&1 | tee -a logs/cron.log

# Log completion
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check completed" >> logs/cron.log
echo "" >> logs/cron.log

