#!/usr/bin/env python3
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time

# Config
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
PAIRS = ["BTCUSDT", "ETHUSDT"]
DAYS_BACK = 30
DATA_DIR = Path("/home/nosferatu/freqtrade/user_data/data/binance/aggTrades")

def download_file(url, target_path):
    if target_path.exists():
        # print(f"  Skipping (exists): {target_path.name}")
        return True
    
    print(f"  Downloading: {url}")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            target_path.write_bytes(response.content)
            return True
        else:
            print(f"  Failed ({response.status_code}): {url}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=DAYS_BACK)
    
    print(f"=== Binance aggTrade Downloader ===")
    print(f"Targeting: {PAIRS}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    
    for pair in PAIRS:
        print(f"\nProcessing {pair}...")
        current = start_date
        success_count = 0
        
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            filename = f"{pair}-aggTrades-{date_str}.zip"
            url = f"{BASE_URL}/{pair}/{filename}"
            target_path = DATA_DIR / filename
            
            if download_file(url, target_path):
                success_count += 1
            
            current += timedelta(days=1)
            time.sleep(0.5) # rate limit
            
        print(f"  Finished {pair}: {success_count}/{DAYS_BACK+1} files active.")

if __name__ == "__main__":
    main()
