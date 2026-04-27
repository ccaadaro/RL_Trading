#!/bin/bash
# start_institutional_bot.sh
# Levanta el daemon ZMQ y Freqtrade en sesiones tmux independientes.
# Pueden cerrarse las terminales sin que los procesos mueran.
#
# Uso:
#   bash start_institutional_bot.sh          # arrancar
#   bash start_institutional_bot.sh stop     # parar todo
#   tmux attach -t daemon                    # ver daemon
#   tmux attach -t freqtrade                 # ver bot

CONDA_ENV="freqtrade"
FREQTRADE_DIR="/home/nosferatu/freqtrade"
RL_DIR="$FREQTRADE_DIR/user_data/strategies/RL_Trading"
CONFIG="$FREQTRADE_DIR/user_data/config.json"
STRATEGY_PATH="$FREQTRADE_DIR/user_data/strategies/institutional/"
CONDA_SH="/home/nosferatu/anaconda3/etc/profile.d/conda.sh"

SESSION_DAEMON="institutional-daemon"
SESSION_BOT="institutional-bot"

case "${1:-start}" in

  start)
    # ── Daemon ────────────────────────────────────────────────────────────────
    if tmux has-session -t "$SESSION_DAEMON" 2>/dev/null; then
      echo "[INFO] Sesión '$SESSION_DAEMON' ya existe. Usa: tmux attach -t $SESSION_DAEMON"
    else
      tmux new-session -d -s "$SESSION_DAEMON" -x 220 -y 50
      tmux send-keys -t "$SESSION_DAEMON" "source '$CONDA_SH' && conda activate '$CONDA_ENV'" Enter
      tmux send-keys -t "$SESSION_DAEMON" "cd '$RL_DIR'" Enter
      tmux send-keys -t "$SESSION_DAEMON" "python services/market_data_daemon.py --symbol BTCUSDT 2>&1 | tee logs/daemon.log" Enter
      echo "[OK] Daemon arrancado en tmux session '$SESSION_DAEMON'"
    fi

    # Esperar un momento para que el daemon inicialice el ZMQ socket
    sleep 2

    # ── Freqtrade Bot ─────────────────────────────────────────────────────────
    if tmux has-session -t "$SESSION_BOT" 2>/dev/null; then
      echo "[INFO] Sesión '$SESSION_BOT' ya existe. Usa: tmux attach -t $SESSION_BOT"
    else
      tmux new-session -d -s "$SESSION_BOT" -x 220 -y 50
      tmux send-keys -t "$SESSION_BOT" "source '$CONDA_SH' && conda activate '$CONDA_ENV'" Enter
      tmux send-keys -t "$SESSION_BOT" "cd '$FREQTRADE_DIR'" Enter
      tmux send-keys -t "$SESSION_BOT" "freqtrade trade --userdir '$FREQTRADE_DIR/user_data' --config '$CONFIG' --strategy-path '$STRATEGY_PATH' -s InstitutionalDollarStrategy 2>&1 | tee user_data/logs/freqtrade.log" Enter
      echo "[OK] Bot arrancado en tmux session '$SESSION_BOT'"
    fi

    echo ""
    echo "Comandos útiles:"
    echo "  tmux attach -t $SESSION_DAEMON    # ver log del daemon"
    echo "  tmux attach -t $SESSION_BOT       # ver log del bot"
    echo "  tmux ls                           # listar todas las sesiones"
    echo "  Ctrl+B, D                         # detach (salir sin matar)"
    echo "  bash $0 stop                      # parar todo"
    ;;

  stop)
    echo "[INFO] Parando sesiones tmux..."
    tmux kill-session -t "$SESSION_DAEMON" 2>/dev/null && echo "[OK] Daemon parado." || echo "[WARN] Daemon no estaba corriendo."
    tmux kill-session -t "$SESSION_BOT"    2>/dev/null && echo "[OK] Bot parado."    || echo "[WARN] Bot no estaba corriendo."
    ;;

  status)
    echo "=== Sesiones tmux activas ==="
    tmux ls 2>/dev/null || echo "Ninguna sesión activa."
    ;;

  *)
    echo "Uso: $0 [start|stop|status]"
    ;;
esac
