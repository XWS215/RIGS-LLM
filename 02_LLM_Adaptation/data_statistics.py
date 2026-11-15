import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "Data", "metrics.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "statistics")
T_CRITICAL = {0}


def fail(message):
    sys.stderr.write("[error] " + str(message) + "\n")
    sys.exit(1)


def info(message):
    sys.stdout.write("[info] " + str(message) + "\n")
    sys.stdout.flush()


def warn(message):
    sys.stderr.write("[warn] " + str(message) + "\n")
    sys.stderr.flush()


def t_critical(df):
    if df in T_CRITICAL:
        return T_CRITICAL[df]
    if df < 1:
        return 12.706
    if df > 120:
        return 1.960
    return 1.960 + 2.576 / math.sqrt(float(df))


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if len(text) == 0:
        return None
    lowered = text.lower()
    if lowered in ("na", "nan", "none", "null", "-"):
        return None
    try:
        return float(text.replace("%", ""))
    except Exception:
        return None


def load_table(path):
    if not os.path.isfile(path):
        fail("文件不存在：%s" % path)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as err:
                    fail("%s:%d 解析失败：%s" % (path, line_no, err))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "records" in payload:
            return payload["records"]
        if isinstance(payload, list):
            return payload
        fail("JSON 文件需要是记录列表：%s" % path)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_columns(rows):
    if len(rows) == 0:
        return []
    keys = list(rows[0].keys())
    columns = []
    for key in keys:
        values = [to_float(row.get(key)) for row in rows]
        valid = [value for value in values if value is not None]
        if len(valid) >= max(2, int(0.5 * len(rows))):
            columns.append(key)
    return columns


def group_rows(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        key = row.get(group_key, "")
        if isinstance(key, (dict, list)):
            key = json.dumps(key, ensure_ascii=False)
        groups[str(key)].append(row)
    return groups


def values_of(rows, column):
    values = []
    for row in rows:
        value = to_float(row.get(column))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def mean(values):
    if len(values) == 0:
        return float("nan")
    return float(sum(values)) / float(len(values))


def stdev(values, ddof=1):
    count = len(values)
    if count <= ddof:
        return float("nan")
    center = mean(values)
    total = 0.0
    for value in values:
        total += (value - center) ** 2
    return math.sqrt(total / float(count - ddof))


def quantile(values, probability):
    if len(values) == 0:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def mean_ci(values, confidence=0.95):
    count = len(values)
    if count < 2:
        return mean(values), float("nan"), float("nan")
    sample_mean = mean(values)
    sample_sd = stdev(values)
    half = t_critical(count - 1) * sample_sd / math.sqrt(float(count))
    return sample_mean, sample_mean - half, sample_mean + half


def bootstrap_ci(values, statistic=mean, iterations=1000, seed=0):
    if len(values) < 2:
        value = statistic(values) if values else float("nan")
        return value, float("nan"), float("nan")
    state = random.Random(seed)
    estimates = []
    count = len(values)
    for _ in range(iterations):
        sample = [values[state.randrange(count)] for _ in range(count)]
        estimates.append(statistic(sample))
    return statistic(values), quantile(estimates, 0.025), quantile(estimates, 0.975)


def rank_values(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = float(index + end + 2) / 2.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 3:
        return float("nan")
    rank_left = rank_values(left)
    rank_right = rank_values(right)
    mean_left = mean(rank_left)
    mean_right = mean(rank_right)
    numerator = 0.0
    left_var = 0.0
    right_var = 0.0
    for index in range(len(left)):
        dl = rank_left[index] - mean_left
        dr = rank_right[index] - mean_right
        numerator += dl * dr
        left_var += dl * dl
        right_var += dr * dr
    if left_var == 0.0 or right_var == 0.0:
        return float("nan")
    return numerator / math.sqrt(left_var * right_var)


def kendall_tau(left, right):
    if len(left) != len(right) or len(left) < 3:
        return float("nan")
    concordant = 0
    discordant = 0
    count = len(left)
    for i in range(count):
        for j in range(i + 1, count):
            sign_left = (left[i] - left[j])
            sign_right = (right[i] - right[j])
            product = sign_left * sign_right
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = float(concordant + discordant)
    if total == 0.0:
        return float("nan")
    return float(concordant - discordant) / total


def paired_permutation_test(left, right, iterations=5000, seed=0):
    if len(left) != len(right) or len(left) < 3:
        return float("nan"), float("nan")
    differences = []
    for a, b in zip(left, right):
        differences.append(a - b)
    observed = abs(mean(differences))
    state = random.Random(seed)
    count = 0
    for _ in range(iterations):
        total = 0.0
        for value in differences:
            if state.random() < 0.5:
                total += value
            else:
                total -= value
        if abs(total / float(len(differences))) >= observed:
            count += 1
    return observed, float(count + 1) / float(iterations + 1)


def cliffs_delta(left, right):
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    greater = 0
    less = 0
    for a in left:
        for b in right:
            if a > b:
                greater += 1
            elif a < b:
                less += 1
    total = float(len(left) * len(right))
    return float(greater - less) / total


def levene_permutation(groups, iterations=5000, seed=0):
    def statistic(samples):
        deviations = []
        for sample in samples:
            center = quantile(sample, 0.5)
            deviations.append([abs(value - center) for value in sample])
        grand = mean([value for item in deviations for value in item])
        numerator = 0.0
        denominator = 0.0
        for item in deviations:
            numerator += len(item) * (mean(item) - grand) ** 2
            for value in item:
                denominator += (value - mean(item)) ** 2
        if denominator == 0.0:
            return 0.0
        k = len(samples)
        n = sum(len(item) for item in samples)
        return float(n - k) / float(k - 1) * numerator / denominator

    observed = statistic(groups)
    state = random.Random(seed)
    pooled = []
    for group in groups:
        pooled.extend(group)
    sizes = [len(group) for group in groups]
    count = 0
    for _ in range(iterations):
        state.shuffle(pooled)
        samples = []
        start = 0
        for size in sizes:
            samples.append(pooled[start:start + size])
            start += size
        if statistic(samples) >= observed:
            count += 1
    return observed, float(count + 1) / float(iterations + 1)


def krippendorff_alpha(units, level="interval"):
    if len(units) == 0:
        return float("nan")
    categories = []
    for unit in units:
        for value in unit:
            if value not in categories:
                categories.append(value)
    categories.sort()
    index = {value: position for position, value in enumerate(categories)}
    coincidence = [[0.0] * len(categories) for _ in categories]
    for unit in units:
        counts = [0] * len(categories)
        for value in unit:
            counts[index[value]] += 1
        total = float(sum(counts))
        if total < 2.0:
            continue
        for a in range(len(categories)):
            for b in range(len(categories)):
                if a == b:
                    coincidence[a][b] += float(counts[a]) * (counts[a] - 1) / (total - 1.0)
                else:
                    coincidence[a][b] += float(counts[a]) * float(counts[b]) / (total - 1.0)
    totals = []
    for row in coincidence:
        totals.append(sum(row))
    grand = sum(totals)
    if grand == 0.0:
        return float("nan")
    observed = 0.0
    for a in range(len(categories)):
        for b in range(len(categories)):
            observed += coincidence[a][b] * metric_distance(categories[a], categories[b], level)
    expected = 0.0
    for a in range(len(categories)):
        for b in range(len(categories)):
            expected += totals[a] * totals[b] * metric_distance(categories[a], categories[b], level)
    if expected == 0.0:
        return float("nan")
    return 1.0 - observed / expected


def metric_distance(left, right, level):
    if level == "nominal":
        return 0.0 if left == right else 1.0
    difference = float(left) - float(right)
    return difference * difference


def holm_adjust(pvalues):
    indexed = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(indexed):
        value = pvalues[index] * float(count - rank)
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def describe(rows, columns):
    report = []
    for column in columns:
        values = values_of(rows, column)
        if len(values) == 0:
            continue
        sample_mean, low, high = mean_ci(values)
        report.append({
            "column": column,
            "n": len(values),
            "mean": sample_mean,
            "sd": stdev(values),
            "median": quantile(values, 0.5),
            "q1": quantile(values, 0.25),
            "q3": quantile(values, 0.75),
            "min": min(values),
            "max": max(values),
            "ci_low": low,
            "ci_high": high,
        })
    return report


def group_summary(rows, group_key, column):
    groups = group_rows(rows, group_key)
    report = []
    for key in sorted(groups.keys()):
        values = values_of(groups[key], column)
        if len(values) == 0:
            continue
        sample_mean, low, high = mean_ci(values)
        report.append({
            "group": key,
            "n": len(values),
            "mean": sample_mean,
            "sd": stdev(values),
            "ci_low": low,
            "ci_high": high,
        })
    return report


def to_markdown(rows, columns=None):
    if len(rows) == 0:
        return ""
    if columns is None:
        columns = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("—")
                else:
                    cells.append("%.3f" % value)
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_report(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=float)
    info("统计结果写入 %s" % path)


def build_parser():
    parser = argparse.ArgumentParser(description="数据与统计分析")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", type=str, default="describe")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--column", type=str, default="")
    parser.add_argument("--left", type=str, default="")
    parser.add_argument("--right", type=str, default="")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    rows = load_table(args.input)
    info("读入 %d 行" % len(rows))
    columns = numeric_columns(rows)
    info("数值列：%s" % ", ".join(columns))
    if args.task == "describe":
        report = describe(rows, columns)
        print(to_markdown(report))
        save_report(os.path.join(args.output_dir, "describe.json"), report)
        return
    if args.task == "group":
        if len(args.group) == 0 or len(args.column) == 0:
            fail("group 任务需要 --group 与 --column")
        report = group_summary(rows, args.group, args.column)
        print(to_markdown(report))
        save_report(os.path.join(args.output_dir, "group_%s.json" % args.column), report)
        return
    if args.task == "correlation":
        if len(columns) < 2:
            fail("不足两列数值，无法计算相关")
        results = []
        for left_index in range(len(columns)):
            for right_index in range(left_index + 1, len(columns)):
                left_values = []
                right_values = []
                for row in rows:
                    left_value = to_float(row.get(columns[left_index]))
                    right_value = to_float(row.get(columns[right_index]))
                    if left_value is None or right_value is None:
                        continue
                    left_values.append(left_value)
                    right_values.append(right_value)
                results.append({
                    "left": columns[left_index],
                    "right": columns[right_index],
                    "n": len(left_values),
                    "spearman": spearman(left_values, right_values),
                    "kendall": kendall_tau(left_values, right_values),
                })
        print(to_markdown(results))
        save_report(os.path.join(args.output_dir, "correlation.json"), results)
        return
    if args.task == "compare":
        if len(args.group) == 0 or len(args.column) == 0:
            fail("compare 任务需要 --group 与 --column")
        groups = group_rows(rows, args.group)
        keys = sorted(groups.keys())
        if len(keys) < 2:
            fail("分组少于两个，无法比较")
        samples = [values_of(groups[key], args.column) for key in keys]
        statistic, pvalue = levene_permutation(samples, iterations=args.iterations, seed=args.seed)
        payload = {
            "column": args.column,
            "groups": keys,
            "levene_statistic": statistic,
            "levene_pvalue": pvalue,
            "summary": group_summary(rows, args.group, args.column),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
        save_report(os.path.join(args.output_dir, "compare_%s.json" % args.column), payload)
        return
    if args.task == "paired":
        if len(args.left) == 0 or len(args.right) == 0:
            fail("paired 任务需要 --left 与 --right")
        left_values = values_of(rows, args.left)
        right_values = values_of(rows, args.right)
        if len(left_values) != len(right_values):
            size = min(len(left_values), len(right_values))
            left_values = left_values[:size]
            right_values = right_values[:size]
        difference, pvalue = paired_permutation_test(
            left_values, right_values, iterations=args.iterations, seed=args.seed)
        payload = {
            "left": args.left,
            "right": args.right,
            "n": len(left_values),
            "mean_difference": mean([a - b for a, b in zip(left_values, right_values)]),
            "abs_mean_difference": difference,
            "pvalue": pvalue,
            "cliffs_delta": cliffs_delta(left_values, right_values),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=float))
        save_report(os.path.join(args.output_dir, "paired.json"), payload)
        return
    fail("未知任务：%s" % args.task)


if __name__ == "__main__":
    main()
