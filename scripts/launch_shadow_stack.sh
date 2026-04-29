#!/bin/bash
# scripts/launch_shadow_stack.sh
# Kill existing
pkill -9 -f freqtrade || true
pkill -9 -f market_data_daemon || true
pkill -9 -f monitor_shadow_selector || true

# Start Daemon
nohup /home/nosferatu/anaconda3/envs/freqtrade/bin/python services/market_data_daemon.py > logs/daemon.log 2>&1 &
echo "Market Data Daemon started (PID $!)."

sleep 5

# Start Strategy
nohup /home/nosferatu/anaconda3/envs/freqtrade/bin/freqtrade trade -c /home/nosferatu/freqtrade/user_data/config.json --userdir /home/nosferatu/freqtrade/user_data/ -s InstitutionalDollarStrategy --strategy-path . --dry-run > logs/freqtrade_shadow.log 2>&1 &
echo "Freqtrade Strategy started (PID $!)."

echo "Waiting for stabilization..."
sleep 10
ps -ef | grep -E "freqtrade|market_data_daemon" | grep -v grep
