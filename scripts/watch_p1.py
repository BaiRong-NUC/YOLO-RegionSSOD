"""Read-only local P1 progress viewer; start with the existing yolo Python."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
from statistics import mean
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs/gate3_20261027"
RUN = ROOT / "runs/partial10_p1_pilot30_s0"
LOG = ROOT / "outputs/p1_pilot30_20261027.stdout.log"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def tail(path: Path, count=160_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - count))
        return ANSI.sub("", handle.read().decode("utf-8", errors="replace"))


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def collect():
    status = read_json(EVIDENCE / "status.json")
    csv_file = RUN / "results.csv"
    rows = []
    if csv_file.is_file():
        for row in csv.DictReader(io.StringIO(csv_file.read_text(encoding="utf-8-sig"))):
            try:
                if all(value is not None and value.strip() for value in row.values()):
                    int(row["epoch"])
                    float(row["time"])
                    rows.append(row)
            except (ValueError, TypeError):
                continue
    complete = int(rows[-1]["epoch"]) if rows else 0
    log = tail(LOG)
    matches = re.findall(r"(\d+)/30\s+[^\r\n]*?(\d+)/(\d+)", log)
    epoch, batch, batches = map(int, matches[-1]) if matches else (complete, 0, 0)
    fraction = batch / batches if batches and epoch > complete else 0
    progress = min(100, (complete + fraction) / 30 * 100)
    steps = [float(b["time"]) - float(a["time"]) for a, b in zip(rows[-6:], rows[-5:])] if len(rows) >= 6 else []
    # For short runs use the cumulative CSV time; estimates exclude later evaluation.
    seconds_per_epoch = mean(steps) if steps else float(rows[-1]["time"]) / complete if rows else None
    eta = round(max(0, 30 - complete - fraction) * seconds_per_epoch / 60) if seconds_per_epoch else None
    return {"stage": status.get("stage", "unknown"), "complete": complete,
            "epoch": max(epoch, complete), "batch": batch, "batches": batches,
            "progress": progress, "eta_minutes": eta,
            "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
            "log_updated_at": datetime.fromtimestamp(LOG.stat().st_mtime, ZoneInfo("Asia/Shanghai")).strftime("%H:%M:%S") if LOG.exists() else "—",
            "lines": log.splitlines()[-8:],
            "error": tail(ROOT / "outputs/p1_pilot30_20261027.stderr.log", 8000),
            "result": read_json(EVIDENCE / "comparison.json")}


def render(state):
    stage = state["stage"]
    labels = {"training": "训练进行中", "validation960": "训练已结束 · 完整 val960 验证中",
              "size_recall": "训练已结束 · 小目标 recall 评估中", "completed": "训练和自动评估均已完成",
              "failed": "流程异常 · 请通知我检查", "unknown": "等待状态记录"}
    title = labels.get(stage, stage)
    if stage == "training" and state["complete"] >= 30:
        title = "30 epoch 训练已结束 · 训练内最终验证中"
    done = stage == "completed"
    message = "现在可以通知我：P1 已完成，请验收 Gate 3 结果。" if done else "训练结束后还会自动进行 val960 和小目标 recall；显示全部完成后即可通知我验收。"
    current = f"第 {state['epoch']} / 30 epoch · batch {state['batch']} / {state['batches']}" if state["batches"] else "等待批次日志"
    eta = f"训练剩余约 {state['eta_minutes']} 分钟（按近期速度估计，另需自动评估时间）" if state["eta_minutes"] is not None and stage == "training" and state["complete"] < 30 else ""
    result = state["result"]
    verdict = ("Gate 3 自动比较：" + ("通过" if result["gate"]["passed"] else "未通过") + "，等待验收核对。") if done and "gate" in result else "Gate 3 尚未验收"
    logs = html.escape("\n".join(state["lines"]))
    errors = f"<details open><summary>错误输出</summary><pre>{html.escape(state['error'])}</pre></details>" if state["error"] else ""
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="3">
<title>P1 · {html.escape(title)}</title><style>
:root{{color-scheme:light dark;font-family:system-ui,'Microsoft YaHei',sans-serif;background:light-dark(#f5f7fa,#10141c);color:light-dark(#172538,#e8eef7)}}
body{{max-width:1000px;margin:auto;padding:28px 20px;line-height:1.7}}h1{{font-size:26px;margin:8px 0}}h2{{font-size:19px}}small,.muted{{color:light-dark(#536276,#a9b7cb)}}
.status{{padding:20px;background:light-dark(#fff,#1b2432);border-radius:16px;margin:20px 0}}.number{{font-size:34px;font-weight:600;font-variant-numeric:tabular-nums}}
progress{{width:100%;height:18px;accent-color:#398eeb}}pre{{font:12px/1.7 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:light-dark(#e8edf4,#192331);padding:16px;border-radius:10px}}
.done{{color:light-dark(#157042,#73dfa3)}}.stage{{font-size:20px;font-weight:600}}a{{color:light-dark(#175cb2,#8bc4ff)}}
</style><body><small>YOLO11s · VisDrone · seed 0 · 30 epoch pilot</small><h1>P1 实时训练进度</h1>
<p class="muted">每 3 秒自动刷新 · 页面更新 {state['checked_at']}（北京时间）· 日志更新 {state['log_updated_at']}</p>
<div class="status"><div class="stage {'done' if done else ''}">{html.escape(title)}</div>
<div class="number">{state['progress']:.1f}%</div><progress value="{state['progress']}" max="100" aria-label="训练进度"></progress>
<p>已完成 {state['complete']} / 30 epoch<br>{html.escape(current)}</p><p class="muted">{eta}</p></div>
<p class="{'done' if done else ''}">{message}</p><p>{html.escape(verdict)}</p>
<h2>最近日志</h2><pre>{logs}</pre>{errors}
<small>进度条仅表示训练进度。显示“训练和自动评估均已完成”表示本轮流程结束；训练中的 640 指标不用于判断 Gate 3。</small>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route not in {"/", "/state"}:
            self.send_error(404)
            return
        state = collect()
        body = (json.dumps(state, ensure_ascii=False) if route == "/state" else render(state)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8" if route == "/state" else "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    server = ThreadingHTTPServer(("127.0.0.1", parser.parse_args().port), Handler)
    output = ROOT / "outputs/p1_monitor"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"url": f"http://127.0.0.1:{server.server_port}/", "pid": os.getpid()}
    (output / "server.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    server.serve_forever()
