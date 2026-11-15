import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "outputs", "runs")
INDEX_NAME = "index.csv"
MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.jsonl"
CONFIG_NAME = "config.json"
LOG_NAME = "run.log"


def fail(message):
    sys.stderr.write("[error] " + str(message) + "\n")
    sys.exit(1)


def info(message):
    sys.stdout.write("[info] " + str(message) + "\n")
    sys.stdout.flush()


def warn(message):
    sys.stderr.write("[warn] " + str(message) + "\n")
    sys.stderr.flush()


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_id(prefix="run"):
    return "%s_%s" % (prefix, now_stamp())


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def sha256_of(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def load_config(path):
    if path is None or len(str(path).strip()) == 0:
        return {}
    if not os.path.isfile(path):
        fail("配置文件不存在：%s" % path)
    if path.endswith(".json"):
        return load_json(path, {})
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
        except Exception:
            fail("读取 YAML 需要 pyyaml")
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    fail("不支持的配置文件格式：%s" % path)


def merge_config(base, overrides):
    merged = dict(base)
    for key, value in (overrides or {}).items():
        merged[key] = value
    return merged


def set_global_seed(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_logger(log_path, verbose=True):
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


class MetricsLedger:
    def __init__(self, path):
        self.path = path

    def append(self, record):
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = dict(record)
        record.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read(self):
        rows = []
        if not os.path.isfile(self.path):
            return rows
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def summary(self):
        rows = self.read()
        numeric = defaultdict(list)
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric[key].append(float(value))
        result = {}
        for key, values in numeric.items():
            count = len(values)
            total = sum(values)
            center = total / float(count)
            variance = 0.0
            for value in values:
                variance += (value - center) ** 2
            sample_sd = 0.0
            if count > 1:
                sample_sd = (variance / float(count - 1)) ** 0.5
            result[key] = {
                "n": count,
                "mean": center,
                "sd": sample_sd,
                "min": min(values),
                "max": max(values),
                "last": values[-1],
            }
        return result

    def export_csv(self, path):
        rows = self.read()
        if len(rows) == 0:
            return path
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path


class ArtifactRegistry:
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        self.payload = load_json(manifest_path, {"artifacts": []})
        if "artifacts" not in self.payload:
            self.payload["artifacts"] = []

    def register(self, path, name=None, kind="file"):
        if not os.path.isfile(path):
            fail("产物不存在：%s" % path)
        entry = {
            "name": name if name else os.path.basename(path),
            "path": os.path.abspath(path),
            "kind": kind,
            "size": os.path.getsize(path),
            "sha256": sha256_of(path),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.payload["artifacts"].append(entry)
        dump_json(self.manifest_path, self.payload)
        return entry

    def copy_and_register(self, path, target_dir, name=None, kind="file"):
        ensure_dir(target_dir)
        target = os.path.join(target_dir, name if name else os.path.basename(path))
        shutil.copy2(path, target)
        return self.register(target, name=name, kind=kind)

    def entries(self):
        return list(self.payload.get("artifacts", []))

    def verify(self):
        results = []
        for entry in self.entries():
            path = entry.get("path", "")
            exists = os.path.isfile(path)
            actual = sha256_of(path) if exists else ""
            results.append({
                "name": entry.get("name", ""),
                "path": path,
                "exists": exists,
                "expected": entry.get("sha256", ""),
                "actual": actual,
                "match": exists and actual == entry.get("sha256", ""),
            })
        return results


class RunContext:
    def __init__(self, root, run_id, config, seed):
        self.root = root
        self.run_id = run_id
        self.dir = os.path.join(root, run_id)
        self.config_dir = ensure_dir(os.path.join(self.dir, "config"))
        self.log_dir = ensure_dir(os.path.join(self.dir, "logs"))
        self.metrics_dir = ensure_dir(os.path.join(self.dir, "metrics"))
        self.artifacts_dir = ensure_dir(os.path.join(self.dir, "artifacts"))
        self.config = dict(config)
        self.config.setdefault("run_id", run_id)
        self.config.setdefault("seed", seed)
        self.config.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.config_path = os.path.join(self.config_dir, CONFIG_NAME)
        dump_json(self.config_path, self.config)
        self.logger = build_logger(os.path.join(self.log_dir, LOG_NAME))
        self.ledger = MetricsLedger(os.path.join(self.metrics_dir, METRICS_NAME))
        self.registry = ArtifactRegistry(os.path.join(self.dir, MANIFEST_NAME))

    def log(self, message, level="info"):
        writer = getattr(self.logger, level, self.logger.info)
        writer(message)

    def record(self, record):
        self.ledger.append(record)
        self.log("记录指标：%s" % json.dumps(record, ensure_ascii=False))

    def artifact(self, path, name=None, kind="file"):
        entry = self.registry.register(path, name=name, kind=kind)
        self.log("登记产物：%s（%d 字节）" % (entry["name"], entry["size"]))
        return entry

    def summary(self):
        return {
            "run_id": self.run_id,
            "dir": self.dir,
            "config": self.config,
            "metrics": self.ledger.summary(),
            "artifacts": len(self.registry.entries()),
        }

    def close(self):
        self.ledger.export_csv(os.path.join(self.metrics_dir, "metrics.csv"))
        dump_json(os.path.join(self.dir, "summary.json"), self.summary())
        self.log("运行 %s 结束" % self.run_id)


def index_path(root):
    return os.path.join(root, INDEX_NAME)


def read_index(root):
    path = index_path(root)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_index(root, rows):
    path = index_path(root)
    ensure_dir(root)
    if len(rows) == 0:
        return path
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def append_index(root, entry):
    rows = read_index(root)
    rows.append(entry)
    return write_index(root, rows)


def load_run(root, run_id):
    path = os.path.join(root, run_id)
    if not os.path.isdir(path):
        fail("运行目录不存在：%s" % path)
    return path


def collect_runs(root):
    runs = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        summary = load_json(os.path.join(path, "summary.json"), {})
        config = load_json(os.path.join(path, "config", CONFIG_NAME), {})
        runs.append({
            "run_id": name,
            "dir": path,
            "created_at": config.get("created_at", ""),
            "seed": config.get("seed", ""),
            "metrics": summary.get("metrics", {}),
            "artifacts": summary.get("artifacts", 0),
        })
    return runs


def format_table(rows, columns):
    if len(rows) == 0:
        return ""
    widths = []
    for column in columns:
        width = len(str(column))
        for row in rows:
            width = max(width, len(str(row.get(column, ""))))
        widths.append(width)
    lines = []
    header = "  ".join(str(column).ljust(width) for column, width in zip(columns, widths))
    lines.append(header)
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(row.get(column, "")).ljust(width)
                               for column, width in zip(columns, widths)))
    return "\n".join(lines)


def command_new(args):
    config = load_config(args.config) if args.config else {}
    config = merge_config(config, parse_overrides(args.set))
    root = ensure_dir(args.root)
    run_id = args.run_id if args.run_id else make_run_id(args.prefix)
    seed = int(config.get("seed", args.seed))
    set_global_seed(seed)
    context = RunContext(root, run_id, config, seed)
    append_index(root, {
        "run_id": run_id,
        "created_at": context.config["created_at"],
        "seed": seed,
        "dir": context.dir,
        "status": "created",
    })
    context.log("创建运行 %s，种子 %d" % (run_id, seed))
    context.close()
    info("运行目录：%s" % context.dir)
    return context.dir


def command_record(args):
    path = load_run(args.root, args.run_id)
    ledger = MetricsLedger(os.path.join(path, "metrics", METRICS_NAME))
    payload = json.loads(args.json)
    if not isinstance(payload, dict):
        fail("--json 必须是 JSON 对象")
    payload["step"] = args.step
    ledger.append(payload)
    info("已记录：%s" % json.dumps(payload, ensure_ascii=False))


def command_artifact(args):
    path = load_run(args.root, args.run_id)
    registry = ArtifactRegistry(os.path.join(path, MANIFEST_NAME))
    if args.copy:
        target_dir = os.path.join(path, "artifacts")
        entry = registry.copy_and_register(args.path, target_dir, name=args.name)
    else:
        entry = registry.register(args.path, name=args.name)
    info("登记产物 %s，sha256=%s" % (entry["name"], entry["sha256"][:16]))


def command_verify(args):
    path = load_run(args.root, args.run_id)
    registry = ArtifactRegistry(os.path.join(path, MANIFEST_NAME))
    results = registry.verify()
    if len(results) == 0:
        info("清单为空")
        return
    ok = 0
    for item in results:
        status = "OK" if item["match"] else "FAIL"
        if item["match"]:
            ok += 1
        info("%-32s %s %s" % (item["name"], status, item["path"]))
    info("校验完成：%d / %d 通过" % (ok, len(results)))
    if ok != len(results):
        sys.exit(1)


def command_list(args):
    runs = collect_runs(args.root)
    if len(runs) == 0:
        info("没有运行记录")
        return
    rows = []
    for run in runs:
        row = {"run_id": run["run_id"], "created_at": run["created_at"], "seed": run["seed"]}
        for key, value in run["metrics"].items():
            row[key] = "%.3f" % value["mean"]
        rows.append(row)
    columns = ["run_id", "created_at", "seed"]
    for run in runs:
        for key in run["metrics"].keys():
            if key not in columns:
                columns.append(key)
    print(format_table(rows, columns))


def command_summary(args):
    path = load_run(args.root, args.run_id)
    payload = load_json(os.path.join(path, "summary.json"), None)
    ledger = MetricsLedger(os.path.join(path, "metrics", METRICS_NAME))
    if payload is None:
        payload = {"run_id": args.run_id, "dir": path, "metrics": ledger.summary()}
    payload["ledger_rows"] = len(ledger.read())
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            fail("--set 需要 key=value 形式：%s" % item)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(key) == 0:
            fail("--set 的键不能为空")
        if value.lower() in ("true", "false"):
            overrides[key] = value.lower() == "true"
            continue
        try:
            overrides[key] = int(value)
            continue
        except Exception:
            pass
        try:
            overrides[key] = float(value)
            continue
        except Exception:
            pass
        overrides[key] = value
    return overrides


def build_parser():
    parser = argparse.ArgumentParser(description="实验运行管理与产物登记")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command")
    new_parser = subparsers.add_parser("new")
    new_parser.add_argument("--config", type=str, default="")
    new_parser.add_argument("--set", nargs="*", default=[])
    new_parser.add_argument("--prefix", type=str, default="run")
    new_parser.add_argument("--run_id", type=str, default="")
    new_parser.add_argument("--seed", type=int, default=42)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--run_id", type=str, required=True)
    record_parser.add_argument("--json", type=str, required=True)
    record_parser.add_argument("--step", type=int, default=0)
    artifact_parser = subparsers.add_parser("artifact")
    artifact_parser.add_argument("--run_id", type=str, required=True)
    artifact_parser.add_argument("--path", type=str, required=True)
    artifact_parser.add_argument("--name", type=str, default=None)
    artifact_parser.add_argument("--copy", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run_id", type=str, required=True)
    subparsers.add_parser("list")
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--run_id", type=str, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "new":
        command_new(args)
        return
    if args.command == "record":
        command_record(args)
        return
    if args.command == "artifact":
        command_artifact(args)
        return
    if args.command == "verify":
        command_verify(args)
        return
    if args.command == "list":
        command_list(args)
        return
    if args.command == "summary":
        command_summary(args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
