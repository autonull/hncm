"""One resumable overnight experiment per invocation.

Run from a systemd timer; a host reboot leaves `running` state behind, the
next invocation marks it interrupted, and the pointer checkpoint resumes.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
STATE = REPORTS / "overnight.json"
LOCK = REPORTS / ".overnight.lock"
OUT = REPORTS / "overnight"
MAX_ATTEMPTS = 3


def scale(size: str, seed: int, cap: int | None = None) -> dict:
    seed_tag = f"-n{seed}" if seed else ""
    cap_tag = f"-c{cap // 1_000_000}m" if cap else ""
    sizes = [s for s in size.split(",") if s]
    arms = [f"codec-scale-code-756752-p{s}{seed_tag}{cap_tag}" for s in sizes]
    cmd = ["scale", "--sizes", size, "--steps", "8000", "--seed", str(seed)]
    if cap:
        cmd += ["--scale-cap", str(cap)]
    return {
        "id": "scale-" + "+".join(sizes) + seed_tag + cap_tag,
        "cmd": cmd,
        "expect": [str(RUNS / arm / "metrics.json") for arm in arms],
        "timeout": (7200 if "40m" in sizes else 5400 if "20m" in sizes else 3600),
    }


TASKS = [
    scale("10m", 1),
    scale("10m,20m", 0, 48_000_000),
    scale("10m", 2),
    scale("10m", 3),
    scale("5m", 2),
    scale("5m", 3),
    scale("10m", 4),
    scale("10m", 5),
    scale("5m", 4),
    scale("10m,20m", 1, 48_000_000),
    scale("10m,20m", 2, 48_000_000),
    scale("20m,40m", 0, 64_000_000),
    {
        "id": "portfolio-round4",
        "cmd": ["portfolio", "--round", "4", "--task", "t2", "--budget", "6h"],
        "expect": [str(REPORTS / "PORTFOLIO.md")],
        "timeout": 300,
    },
]


def load() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {"tasks": {}}


def save(state: dict) -> None:
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)
    write_report(state)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def metrics(paths: list[str]) -> dict:
    out = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out[p.parent.name] = {
            k: m.get(k)
            for k in (
                "seed",
                "target_params",
                "train_bytes",
                "params",
                "payload_bpb",
                "shared_bpb",
                "bits_per_byte",
                "seconds",
            )
        }
    return out


def write_report(state: dict) -> None:
    rows = []
    for task in TASKS:
        entry = state["tasks"].get(task["id"], {})
        status = entry.get("status", "queued")
        attempts = entry.get("attempts", 0)
        got = entry.get("metrics", {})
        detail = "; ".join(
            f"{name} {m['payload_bpb']:.3f}/{m['shared_bpb']:.3f}"
            for name, m in got.items()
            if m.get("payload_bpb") is not None
        )
        rows.append(f"| `{task['id']}` | {status} | {attempts} | {detail or '-'} |")
    done = sum(1 for v in state["tasks"].values() if v.get("status") == "done")
    lines = [
        "# Overnight queue",
        "",
        f"Updated {state.get('updated', '-')}. One task per 5-minute timer.",
        f"Done {done}/{len(TASKS)}; a reboot is retried from the pointer checkpoint.",
        "",
        "| task | status | attempts | payload/shared bpb |",
        "|---|---|---:|---|",
        *rows,
        "",
        "Logs: `reports/overnight/<task>.log`.",
    ]
    tmp = (REPORTS / "OVERNIGHT.md").with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, REPORTS / "OVERNIGHT.md")


def stale(state: dict) -> None:
    for entry in state["tasks"].values():
        if entry.get("status") != "running":
            continue
        pid = int(entry.get("pid", 0))
        if pid and alive(pid):
            continue
        entry["status"] = "interrupted"
        entry["error"] = "process died; checkpoint will resume"


def complete(task: dict) -> bool:
    if not all(Path(p).exists() for p in task["expect"]):
        return False
    if task["id"] == "portfolio-round4":
        try:
            data = json.loads((REPORTS / "portfolio.json").read_text())
        except json.JSONDecodeError:
            return False
        return any(r.get("round") == 4 for r in data.get("rounds", []))
    return True


def pick(state: dict) -> dict | None:
    stale(state)
    for task in TASKS:
        entry = state["tasks"].setdefault(task["id"], {})
        if entry.get("status") == "done":
            continue
        if entry.get("status") == "running" and alive(int(entry.get("pid", 0))):
            return None
        if entry.get("attempts", 0) >= MAX_ATTEMPTS:
            continue
        if complete(task):
            entry["status"] = "done"
            entry["metrics"] = metrics(task["expect"])
            continue
        return task
    return None


def run(task: dict, state: dict) -> None:
    entry = state["tasks"][task["id"]]
    attempt = entry.get("attempts", 0) + 1
    entry.update(
        {
            "status": "running",
            "attempts": attempt,
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    save(state)
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{task['id']}-a{attempt}.log"
    entry["log"] = str(log)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["MALLOC_ARENA_MAX"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [sys.executable, str(ROOT / "cml.py"), *task["cmd"]]
    started = time.perf_counter()
    with log.open("w") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        entry["pid"] = proc.pid
        save(state)
        try:
            code = proc.wait(timeout=task["timeout"])
            error = None
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
            code = -1
            error = f"timeout {task['timeout']}s"
    entry["seconds"] = round(time.perf_counter() - started, 1)
    got = metrics(task["expect"])
    if code == 0 and complete(task):
        entry.update({"status": "done", "metrics": got, "error": None})
    else:
        entry.update(
            {
                "status": "failed",
                "error": error or f"exit {code}; expected metrics missing",
                "metrics": got,
            }
        )
    save(state)


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    lock = LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("overnight: another runner holds the lock")
        return 0
    state = load()
    stale(state)
    save(state)
    if "--dry" in sys.argv:
        task = pick(state)
        save(state)
        print("next:", task["id"] if task else "none")
        return 0
    task = pick(state)
    save(state)
    if task is None:
        print("overnight: queue exhausted")
        return 0
    print("overnight: running", task["id"], flush=True)
    run(task, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
