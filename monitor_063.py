#!/usr/bin/env python3
import time
import re
import sys

log_file = "nohup_063.out"
last_pos = 0
milestones = {
    "burn_in_50k": False,
    "actor_unfreeze": False,
    "eval_65k": False,
}

print("📊 Monitoring Run 063 training...")

while True:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Check for milestones
        if "Actor UNFROZEN" in content and not milestones["actor_unfreeze"]:
            print("\n✅ [MILESTONE] Actor UNFROZEN — Policy gradient now active!")
            milestones["actor_unfreeze"] = True
        
        # Extract latest metrics
        lines = content.split('\n')
        for line in reversed(lines[-50:]):  # Last 50 lines
            if "[burn-in] step=" in line or "[active] step=" in line:
                # Parse key metrics
                import re
                matches = {
                    'step': re.search(r'step=(\d+)k', line),
                    'entropy': re.search(r'entropy=(-?\d\.\d{3})', line),
                    'trades': re.search(r'Trades\s+(\d+)', line),
                    'trades_eval': re.search(r'T\s+(\d+)', line),
                }
                
                msg = f"\n📈 Latest: "
                if matches['step']: 
                    msg += f"Step={matches['step'].group(1)}k  "
                if matches['entropy']: 
                    msg += f"Entropy={matches['entropy'].group(1)}  "
                
                # Look for trades in eval line
                for eval_line in reversed(lines[-50:]):
                    if eval_line.startswith('  │'):
                        if 'Trades' in eval_line:
                            trades = re.search(r'T\s+(\d+)', eval_line)
                            if trades:
                                msg += f"Eval_Trades={trades.group(1)}"
                        break
                
                if msg != "\n📈 Latest: ":
                    print(msg)
                break
        
        time.sleep(30)  # Check every 30s
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
