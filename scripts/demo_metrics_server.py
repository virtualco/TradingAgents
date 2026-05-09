"""
demo_metrics_server.py
Serves realistic 60-day simulated trading data on the same /metrics endpoint
that performance.html and index.html consume. Use for demos and development.

Usage:
    python3 scripts/demo_metrics_server.py [--port 8765]
"""

import json
import random
import math
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse

random.seed(42)

# ── Data generation ────────────────────────────────────────────────────────────

def generate_demo_payload():
    now = datetime.now(timezone.utc)
    symbols = ["BTCUSDT", "ETHUSDT"]

    # ── 60-day regime history (hourly) ──
    regime_history = []
    for h in range(60 * 24, -1, -1):
        ts = (now - timedelta(hours=h)).isoformat()
        for sym in symbols:
            hurst = 0.45 + random.gauss(0.07, 0.06)
            hurst = max(0.3, min(0.75, hurst))
            adx = max(10, random.gauss(22, 8))
            if hurst > 0.55 and adx > 25:
                regime = "TRENDING"
            elif hurst < 0.45 and adx < 20:
                regime = "RANGING"
            else:
                regime = "TRANSITION"
            regime_history.append({
                "symbol": sym, "regime": regime,
                "hurst": round(hurst, 4), "adx": round(adx, 2),
                "timestamp": ts,
            })

    # ── 120 trades over 60 days ──
    trades = []
    for i in range(120):
        sym = random.choice(symbols)
        side = random.choice(["LONG", "SHORT"])
        regime = random.choices(
            ["TRENDING", "RANGING", "TRANSITION"],
            weights=[0.45, 0.35, 0.20]
        )[0]
        strategy = "MOMENTUM" if regime == "TRENDING" else \
                   "MEAN_REVERSION" if regime == "RANGING" else "NONE"
        base_price = 80000 if sym == "BTCUSDT" else 3200
        price = base_price * (1 + random.gauss(0, 0.02))
        qty = round(random.uniform(0.001, 0.003) if sym == "BTCUSDT" else random.uniform(0.05, 0.15), 4)
        win_prob = 0.60 if strategy == "MOMENTUM" else 0.55 if strategy == "MEAN_REVERSION" else 0.45
        won = random.random() < win_prob
        pnl = round(random.uniform(2, 18) if won else -random.uniform(1, 12), 4)
        days_ago = 60 - i * 0.5
        ts = (now - timedelta(days=days_ago, seconds=random.randint(0, 86400))).isoformat()
        trades.append({
            "id": i + 1, "symbol": sym, "side": side,
            "qty": str(qty), "price": str(round(price, 2)),
            "pnl": str(pnl), "regime": regime, "strategy": strategy,
            "timestamp": ts,
        })

    trades.sort(key=lambda t: t["timestamp"], reverse=True)

    # ── Current live signals ──
    reports = []
    for sym in symbols:
        hurst = round(random.uniform(0.44, 0.62), 4)
        adx = round(random.uniform(16, 32), 2)
        rsi = round(random.uniform(38, 68), 2)
        if hurst > 0.55 and adx > 25:
            regime = "TRENDING"
        elif hurst < 0.45 and adx < 20:
            regime = "RANGING"
        else:
            regime = "TRANSITION"
        sub_strategy = "MOMENTUM" if regime == "TRENDING" else \
                       "MEAN_REVERSION" if regime == "RANGING" else "NONE"
        signal = random.choice([1, -1]) if regime != "TRANSITION" else 0
        price = (80000 + random.gauss(0, 500)) if sym == "BTCUSDT" else (3200 + random.gauss(0, 50))
        reports.append({
            "symbol": sym, "signal": signal, "regime": regime,
            "hurst": hurst, "adx": adx, "rsi": rsi,
            "sub_strategy": sub_strategy,
            "close_price": round(price, 2),
            "action": "OPEN" if signal != 0 else "FLAT",
            "side": "LONG" if signal == 1 else ("SHORT" if signal == -1 else None),
            "qty": 0.001 if sym == "BTCUSDT" else 0.05,
            "stop_loss": round(price * 0.985, 2),
            "take_profit": round(price * 1.025, 2),
            "strategy": sub_strategy,
        })

    # ── Aggregate stats ──
    all_pnl = [float(t["pnl"]) for t in trades]
    realized = round(sum(all_pnl), 4)
    unrealized = round(realized * 0.08, 4)
    n_wins = sum(1 for p in all_pnl if p > 0)
    gross_win = sum(p for p in all_pnl if p > 0)
    gross_loss = abs(sum(p for p in all_pnl if p < 0))
    dd_today = round(random.uniform(0.5, 2.5), 2)

    return {
        "cycle_time": now.isoformat(),
        "symbols": symbols,
        "reports": reports,
        "recent_trades": trades[:50],
        "_all_trades": trades,
        "_regime_history": regime_history,
        "daily_stats": {
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "n_trades": len(trades),
            "peak_nav": round(100000 + realized * 1.1, 2),
        },
        "risk_state": {
            "kill_switch_active": False,
            "circuit_breaker_tripped": False,
            "daily_drawdown_pct": dd_today,
            "max_daily_drawdown_pct": 5.0,
            "open_positions": len([r for r in reports if r["signal"] != 0]),
            "max_positions": 2,
            "realized_pnl_today": round(realized * 0.05, 4),
        },
    }


# ── HTTP Server ────────────────────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path in ("/metrics", "/metrics/"):
            payload = generate_demo_payload()
            body = json.dumps(payload, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress default access log noise
        pass


def main():
    parser = argparse.ArgumentParser(description="TradingAgents demo metrics server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"Demo metrics server running on http://localhost:{args.port}/metrics")
    print("Open dashboard/performance.html in your browser")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
