import argparse
import json
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_CHECKPOINT_DIR = os.environ.get("SWFT_CHECKPOINT_DIR", "./checkpoints")
DEFAULT_LOG_PATH = os.environ.get(
    "SWFT_METRICS_LOG",
    os.path.join(DEFAULT_CHECKPOINT_DIR, "train_metrics.jsonl")
)


def read_jsonl_tail(path: str, limit: int = 500) -> list[dict]:
    if not os.path.exists(path):
        return []
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    events = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def disk_usage(path: str) -> dict:
    target = path if os.path.exists(path) else os.path.dirname(path) or "."
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return {}
    return {
        "total_gb": usage.total / 1e9,
        "used_gb": usage.used / 1e9,
        "free_gb": usage.free / 1e9,
        "used_pct": usage.used / max(usage.total, 1) * 100.0,
    }


def list_checkpoints(checkpoint_dir: str) -> list[dict]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return []
    result = []
    for path in sorted(root.glob("*.pt")):
        stat = path.stat()
        result.append({
            "name": path.name,
            "path": str(path),
            "size_mb": stat.st_size / 1e6,
            "mtime": stat.st_mtime,
        })
    return result


def build_status(log_path: str, checkpoint_dir: str) -> dict:
    events = read_jsonl_tail(log_path, limit=2000)
    latest_by_stage: dict[str, dict] = {}
    latest_progress = None
    latest_epoch = None
    for event in events:
        stage = event.get("stage")
        if stage:
            latest_by_stage[str(stage)] = event
        if event.get("event") == "progress":
            latest_progress = event
        if event.get("event") == "epoch_end":
            latest_epoch = event

    return {
        "log_path": log_path,
        "checkpoint_dir": checkpoint_dir,
        "event_count_tail": len(events),
        "latest_event": events[-1] if events else None,
        "latest_progress": latest_progress,
        "latest_epoch": latest_epoch,
        "latest_by_stage": latest_by_stage,
        "checkpoints": list_checkpoints(checkpoint_dir),
        "disk": disk_usage(checkpoint_dir),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SWFT Training Monitor</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #0f1115; color: #e8eaf0; }
    header { position: sticky; top: 0; background: #151923; border-bottom: 1px solid #2b3140; padding: 12px 14px; z-index: 2; }
    h1 { font-size: 18px; margin: 0 0 4px; }
    main { padding: 12px; max-width: 1100px; margin: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
    .card { background: #171b25; border: 1px solid #2b3140; border-radius: 8px; padding: 12px; }
    .label { color: #9aa3b2; font-size: 12px; }
    .value { font-size: 20px; font-weight: 650; margin-top: 4px; overflow-wrap: anywhere; }
    .small { color: #aeb6c5; font-size: 12px; overflow-wrap: anywhere; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #2b3140; padding: 8px 4px; text-align: left; }
    th { color: #aeb6c5; font-weight: 600; }
    canvas { width: 100%; height: 260px; background: #10141d; border-radius: 8px; border: 1px solid #2b3140; }
    .ok { color: #6ee7a8; }
    .warn { color: #fbbf24; }
    .bad { color: #fb7185; }
  </style>
</head>
<body>
  <header>
    <h1>SWFT Training Monitor</h1>
    <div class="small" id="subtitle">Đang tải...</div>
  </header>
  <main>
    <section class="grid">
      <div class="card"><div class="label">Stage</div><div class="value" id="stage">-</div></div>
      <div class="card"><div class="label">Step</div><div class="value" id="step">-</div></div>
      <div class="card"><div class="label">Loss</div><div class="value" id="loss">-</div></div>
      <div class="card"><div class="label">STS-B</div><div class="value" id="stsb">-</div></div>
      <div class="card"><div class="label">LR</div><div class="value" id="lr">-</div></div>
      <div class="card"><div class="label">ETA</div><div class="value" id="eta">-</div></div>
      <div class="card"><div class="label">Disk Free</div><div class="value" id="disk">-</div></div>
      <div class="card"><div class="label">Last Event</div><div class="value" id="event">-</div></div>
    </section>

    <h2>Loss / STS-B</h2>
    <canvas id="chart" width="1000" height="300"></canvas>

    <h2>Checkpoints</h2>
    <div class="card"><table id="ckpts"><thead><tr><th>Name</th><th>Size</th><th>Updated</th></tr></thead><tbody></tbody></table></div>

    <h2>Recent Events</h2>
    <div class="card"><table id="events"><thead><tr><th>Time</th><th>Event</th><th>Stage</th><th>Loss</th><th>Metric</th></tr></thead><tbody></tbody></table></div>
  </main>
<script>
function fmt(n, d=4) { return Number.isFinite(n) ? Number(n).toFixed(d) : "-"; }
function dur(s) {
  if (!Number.isFinite(s)) return "-";
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h${String(m).padStart(2,"0")}m` : `${m}m${String(s%60).padStart(2,"0")}s`;
}
function date(t) { return t ? new Date(t * 1000).toLocaleString() : "-"; }

function drawChart(events) {
  const c = document.getElementById("chart"), ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const points = events.filter(e => e.avg_loss !== undefined || e.stsb_spearman !== undefined);
  if (!points.length) return;
  const xs = points.map((_, i) => i);
  const losses = points.map(e => Number(e.avg_loss)).filter(Number.isFinite);
  const maxLoss = Math.max(...losses, 1), minLoss = Math.min(...losses, 0);
  function x(i) { return 40 + i * (c.width - 70) / Math.max(points.length - 1, 1); }
  function yLoss(v) { return 260 - (v - minLoss) * 220 / Math.max(maxLoss - minLoss, 1e-9); }
  function yMetric(v) { return 260 - Number(v) * 220; }
  ctx.strokeStyle = "#30384a"; ctx.lineWidth = 1;
  for (let i=0;i<5;i++) { const y=40+i*55; ctx.beginPath(); ctx.moveTo(35,y); ctx.lineTo(c.width-20,y); ctx.stroke(); }
  ctx.fillStyle = "#aeb6c5"; ctx.fillText("loss", 44, 24); ctx.fillText("STS-B", 105, 24);
  ctx.strokeStyle = "#60a5fa"; ctx.lineWidth = 2; ctx.beginPath();
  let started = false;
  points.forEach((p,i) => { if (Number.isFinite(Number(p.avg_loss))) { const xx=x(i), yy=yLoss(Number(p.avg_loss)); if (!started) { ctx.moveTo(xx,yy); started=true; } else ctx.lineTo(xx,yy); }});
  ctx.stroke();
  ctx.strokeStyle = "#6ee7a8"; ctx.beginPath(); started = false;
  points.forEach((p,i) => { if (Number.isFinite(Number(p.stsb_spearman))) { const xx=x(i), yy=yMetric(Number(p.stsb_spearman)); if (!started) { ctx.moveTo(xx,yy); started=true; } else ctx.lineTo(xx,yy); }});
  ctx.stroke();
}

async function refresh() {
  const [status, events] = await Promise.all([
    fetch("/api/status").then(r => r.json()),
    fetch("/api/events?limit=200").then(r => r.json())
  ]);
  const latest = status.latest_progress || status.latest_epoch || status.latest_event || {};
  document.getElementById("subtitle").textContent = `Log: ${status.log_path}`;
  document.getElementById("stage").textContent = latest.stage || "-";
  document.getElementById("step").textContent = latest.global_step !== undefined ? `${latest.global_step}/${latest.total_batch_steps || "-"}` : "-";
  document.getElementById("loss").textContent = fmt(latest.avg_loss);
  document.getElementById("stsb").textContent = fmt((status.latest_epoch || {}).stsb_spearman);
  document.getElementById("lr").textContent = latest.lr ? Number(latest.lr).toExponential(2) : "-";
  document.getElementById("eta").textContent = dur(latest.eta_sec);
  const disk = status.disk || {};
  const diskEl = document.getElementById("disk");
  diskEl.textContent = disk.free_gb ? `${disk.free_gb.toFixed(1)}GB` : "-";
  diskEl.className = "value " + (disk.free_gb < 10 ? "bad" : disk.free_gb < 25 ? "warn" : "ok");
  document.getElementById("event").textContent = (status.latest_event || {}).event || "-";

  const ckptBody = document.querySelector("#ckpts tbody");
  ckptBody.innerHTML = "";
  (status.checkpoints || []).slice().reverse().forEach(c => {
    ckptBody.insertAdjacentHTML("beforeend", `<tr><td>${c.name}</td><td>${c.size_mb.toFixed(1)}MB</td><td>${date(c.mtime)}</td></tr>`);
  });

  const evBody = document.querySelector("#events tbody");
  evBody.innerHTML = "";
  events.slice(-20).reverse().forEach(e => {
    evBody.insertAdjacentHTML("beforeend", `<tr><td>${date(e.time_unix)}</td><td>${e.event}</td><td>${e.stage || ""}</td><td>${fmt(e.avg_loss)}</td><td>${fmt(e.stsb_spearman)}</td></tr>`);
  });
  drawChart(events);
}
refresh().catch(console.error);
setInterval(() => refresh().catch(console.error), 10000);
</script>
</body>
</html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG_PATH
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html()
            return
        if parsed.path == "/api/status":
            self._send_json(build_status(self.log_path, self.checkpoint_dir))
            return
        if parsed.path == "/api/events":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["500"])[0])
            self._send_json(read_jsonl_tail(self.log_path, limit=limit))
            return
        if parsed.path == "/api/checkpoints":
            self._send_json(list_checkpoints(self.checkpoint_dir))
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="SWFT training monitor gateway")
    parser.add_argument("--host", default=os.environ.get("SWFT_MONITOR_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SWFT_MONITOR_PORT", "7860")))
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    MonitorHandler.log_path = args.log
    MonitorHandler.checkpoint_dir = args.checkpoint_dir
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"SWFT monitor: http://{args.host}:{args.port}")
    print(f"Log: {args.log}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
