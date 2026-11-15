import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from PIL import Image
except Exception as err:
    plt = None
    np = None
    Image = None
    IMPORT_ERROR = err

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "Data", "metrics.csv")
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "figures")
CM = 1.0 / 2.54
FIG_W = 9.0 * CM
FIG_H = 2.7 * CM
DPI = 600
NAVY = "#1F3864"
BLUE = "#2E75B6"
MID = "#5B9BD5"
LIGHT = "#9DC3E6"
PALE = "#DEEBF7"
INK = "#262626"
GRAY = "#808080"
FIVE = ["#BDD7EE", "#9DC3E6", "#5B9BD5", "#2E75B6", "#1F3864"]
FOUR = ["#9DC3E6", "#5B9BD5", "#2E75B6", "#1F3864"]
MARKERS = ["o", "s", "^", "D", "v"]


def fail(message):
    sys.stderr.write("[error] " + str(message) + "\n")
    sys.exit(1)


def info(message):
    sys.stdout.write("[info] " + str(message) + "\n")
    sys.stdout.flush()


def require_matplotlib():
    if plt is None:
        fail("未安装 matplotlib 或 PIL：%s" % IMPORT_ERROR)


def setup_style():
    require_matplotlib()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })


def to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if len(text) == 0:
        return None
    try:
        return float(text)
    except Exception:
        return None


def load_table(path):
    if not os.path.isfile(path):
        fail("文件不存在：%s" % path)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_in_order(rows, key):
    values = []
    for row in rows:
        value = str(row.get(key, ""))
        if value not in values:
            values.append(value)
    return values


def series(rows, group_key, x_key, y_key):
    table = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row.get(group_key, ""))
        x = str(row.get(x_key, ""))
        y = to_float(row.get(y_key))
        if y is None or not math.isfinite(y):
            continue
        table[group][x].append(y)
    return table


def mean_ci(values):
    count = len(values)
    if count == 0:
        return float("nan"), float("nan")
    center = sum(values) / float(count)
    if count < 2:
        return center, 0.0
    variance = 0.0
    for value in values:
        variance += (value - center) ** 2
    sample_sd = math.sqrt(variance / float(count - 1))
    half = 2.776 * sample_sd / math.sqrt(float(count))
    return center, half


def new_ax(width=FIG_W, height=FIG_H, left=0.135, right=0.975, bottom=0.30, top=0.93):
    setup_style()
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.0, pad=1.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5)
    return fig, ax


def legend_top(ax, handles=None, ncol=3, extra=0.20):
    position = ax.get_position()
    fig = ax.get_figure()
    fig.subplots_adjust(top=position.y1 - extra * (position.y1 - position.y0))
    kwargs = dict(frameon=False, ncol=ncol, loc="lower center",
                  bbox_to_anchor=(0.5, 1.02), handlelength=1.5,
                  columnspacing=1.2, handletextpad=0.4, borderpad=0.0,
                  labelspacing=0.2, fontsize=8)
    if handles is not None:
        ax.legend(handles=handles, **kwargs)
    else:
        ax.legend(**kwargs)


def save_figure(fig, name, output_dir, dpi=DPI):
    require_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, name + ".png")
    tif = os.path.join(output_dir, name + ".tif")
    fig.savefig(png, dpi=dpi, format="png")
    Image.open(png).convert("RGB").save(
        tif, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    plt.close(fig)
    info("图件已保存：%s，%s" % (png, tif))
    return png, tif


def plot_bar_with_ci(rows, group_key, x_key, y_key, title, output_dir, name):
    require_matplotlib()
    table = series(rows, group_key, x_key, y_key)
    groups = sorted(table.keys())
    xs = unique_in_order(rows, x_key)
    width = 0.8 / max(1, len(groups))
    fig, ax = new_ax()
    for index, group in enumerate(groups):
        centers = []
        halves = []
        offsets = []
        for x_index, x in enumerate(xs):
            values = table[group].get(x, [])
            center, half = mean_ci(values)
            centers.append(center)
            halves.append(half)
            offsets.append(x_index + (index - (len(groups) - 1) / 2.0) * width)
        color = FIVE[(index + 1) % len(FIVE)]
        ax.bar(offsets, centers, width=width * 0.9, color=color,
               edgecolor="white", linewidth=0.3, label=group, zorder=2)
        ax.errorbar(offsets, centers, yerr=halves, fmt="none",
                    ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    legend_top(ax, ncol=min(4, len(groups)))
    return save_figure(fig, name, output_dir)


def plot_line_with_ci(rows, group_key, x_key, y_key, title, output_dir, name):
    require_matplotlib()
    table = series(rows, group_key, x_key, y_key)
    groups = sorted(table.keys())
    xs = unique_in_order(rows, x_key)
    fig, ax = new_ax()
    for index, group in enumerate(groups):
        centers = []
        halves = []
        for x in xs:
            values = table[group].get(x, [])
            center, half = mean_ci(values)
            centers.append(center)
            halves.append(half)
        color = FIVE[(index + 1) % len(FIVE)]
        ax.errorbar(range(len(xs)), centers, yerr=halves, fmt="-",
                    lw=0.9, color=color, marker=MARKERS[index % len(MARKERS)],
                    ms=2.4, mec="white", mew=0.3, ecolor=color,
                    elinewidth=0.5, capsize=1.2, capthick=0.5,
                    label=group, zorder=3)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    legend_top(ax, ncol=min(5, len(groups)))
    return save_figure(fig, name, output_dir)


def plot_heatmap(rows, row_key, col_key, value_key, title, output_dir, name):
    require_matplotlib()
    rows_keys = unique_in_order(rows, row_key)
    cols_keys = unique_in_order(rows, col_key)
    matrix = np.full((len(rows_keys), len(cols_keys)), np.nan)
    for row in rows:
        row_index = rows_keys.index(str(row.get(row_key, "")))
        col_index = cols_keys.index(str(row.get(col_key, "")))
        value = to_float(row.get(value_key))
        if value is not None:
            matrix[row_index, col_index] = value
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.24, top=0.88)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols_keys)))
    ax.set_xticklabels(cols_keys)
    ax.set_yticks(range(len(rows_keys)))
    ax.set_yticklabels(rows_keys)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isnan(value):
                continue
            ax.text(j, i, "%.2f" % value, ha="center", va="center",
                    fontsize=8, color="white" if value > 0.55 * np.nanmax(matrix) else INK)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    return save_figure(fig, name, output_dir)


def plot_box(rows, group_key, x_key, y_key, title, output_dir, name):
    require_matplotlib()
    table = series(rows, group_key, x_key, y_key)
    groups = sorted(table.keys())
    xs = unique_in_order(rows, x_key)
    positions = []
    data = []
    colors = []
    width = 0.8 / max(1, len(groups))
    for group_index, group in enumerate(groups):
        for x_index, x in enumerate(xs):
            values = table[group].get(x, [])
            if len(values) == 0:
                continue
            positions.append(x_index + (group_index - (len(groups) - 1) / 2.0) * width)
            data.append(values)
            colors.append(FIVE[(group_index + 1) % len(FIVE)])
    fig, ax = new_ax()
    box = ax.boxplot(data, positions=positions, widths=width * 0.8,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color=INK, lw=0.7),
                     whiskerprops=dict(color=INK, lw=0.5),
                     capprops=dict(color=INK, lw=0.5),
                     boxprops=dict(lw=0.5))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    handles = [Patch(facecolor=FIVE[(index + 1) % len(FIVE)], edgecolor="none", label=group)
               for index, group in enumerate(groups)]
    legend_top(ax, handles=handles, ncol=min(5, len(groups)))
    return save_figure(fig, name, output_dir)


def plot_stacked_area(rows, x_key, group_key, y_key, title, output_dir, name):
    require_matplotlib()
    table = series(rows, group_key, x_key, y_key)
    groups = sorted(table.keys())
    xs = unique_in_order(rows, x_key)
    values = []
    for group in groups:
        row_values = []
        for x in xs:
            items = table[group].get(x, [])
            row_values.append(sum(items) / float(len(items)) if items else 0.0)
        values.append(row_values)
    fig, ax = new_ax()
    ax.stackplot(range(len(xs)), np.array(values), labels=groups,
                 colors=FIVE[:len(groups)], alpha=0.95, linewidth=0.4, edgecolor="white")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlim(0, len(xs) - 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    legend_top(ax, ncol=min(4, len(groups)))
    return save_figure(fig, name, output_dir)


def plot_scatter_fit(rows, x_key, y_key, group_key, title, output_dir, name):
    require_matplotlib()
    fig, ax = new_ax()
    groups = unique_in_order(rows, group_key) if group_key else [""]
    for index, group in enumerate(groups):
        xs = []
        ys = []
        for row in rows:
            if group_key and str(row.get(group_key, "")) != group:
                continue
            x = to_float(row.get(x_key))
            y = to_float(row.get(y_key))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if len(xs) == 0:
            continue
        color = FIVE[(index + 1) % len(FIVE)]
        ax.scatter(xs, ys, s=6, color=color, alpha=0.75,
                   edgecolors="white", linewidths=0.2, label=group or None, zorder=3)
        if len(xs) >= 3:
            mean_x = sum(xs) / float(len(xs))
            mean_y = sum(ys) / float(len(ys))
            numerator = 0.0
            denominator = 0.0
            for x, y in zip(xs, ys):
                numerator += (x - mean_x) * (y - mean_y)
                denominator += (x - mean_x) ** 2
            if denominator > 0:
                slope = numerator / denominator
                intercept = mean_y - slope * mean_x
                line_x = [min(xs), max(xs)]
                line_y = [slope * x + intercept for x in line_x]
                ax.plot(line_x, line_y, lw=0.8, color=color, zorder=2)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    if group_key and len(groups) > 1:
        legend_top(ax, ncol=min(5, len(groups)))
    return save_figure(fig, name, output_dir)


def plot_raster(rows, row_key, col_key, group_key, title, output_dir, name,
                width=FIG_W, height=7.0 * CM):
    require_matplotlib()
    groups = unique_in_order(rows, group_key)
    columns = unique_in_order(rows, col_key)
    fig, axes = plt.subplots(len(groups), 1, figsize=(width, height), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    palette = {value: color for value, color in zip(
        sorted(set(str(row.get("status", "")) for row in rows)),
        [LIGHT, MID, BLUE, NAVY, "#C0504D", "#9BBB59"])}
    for index, group in enumerate(groups):
        ax = axes[index]
        subset = [row for row in rows if str(row.get(group_key, "")) == group]
        lookup = {}
        for row in subset:
            lookup[str(row.get(col_key, ""))] = str(row.get("status", ""))
        colors = [palette.get(lookup.get(column, ""), "#EEEEEE") for column in columns]
        ax.bar(range(len(columns)), [1] * len(columns), width=1.0, color=colors, linewidth=0)
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(group, rotation=0, labelpad=12, va="center")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel(col_key)
    handles = [Patch(facecolor=color, label=value) for value, color in palette.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.suptitle(title, fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.94, hspace=0.25)
    return save_figure(fig, name, output_dir)


def build_demo_rows():
    rows = []
    for group_index in range(3):
        group = "G%d" % (group_index + 1)
        for x in range(1, 6):
            for run in range(3):
                value = 0.3 + 0.1 * group_index + 0.05 * x + 0.01 * run
                rows.append({"group": group, "x": str(x), "value": value})
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="可视化与图件输出")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--kind", type=str, default="bar")
    parser.add_argument("--name", type=str, default="figure")
    parser.add_argument("--group", type=str, default="group")
    parser.add_argument("--x", type=str, default="x")
    parser.add_argument("--y", type=str, default="value")
    parser.add_argument("--row", type=str, default="row")
    parser.add_argument("--col", type=str, default="col")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--demo", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.demo:
        info("使用内置演示数据")
        rows = build_demo_rows()
    else:
        rows = load_table(args.input)
    info("读入 %d 行" % len(rows))
    title = args.title if args.title else args.name
    if args.kind == "bar":
        plot_bar_with_ci(rows, args.group, args.x, args.y, title, args.output_dir, args.name)
    elif args.kind == "line":
        plot_line_with_ci(rows, args.group, args.x, args.y, title, args.output_dir, args.name)
    elif args.kind == "heatmap":
        plot_heatmap(rows, args.row, args.col, args.y, title, args.output_dir, args.name)
    elif args.kind == "box":
        plot_box(rows, args.group, args.x, args.y, title, args.output_dir, args.name)
    elif args.kind == "area":
        plot_stacked_area(rows, args.x, args.group, args.y, title, args.output_dir, args.name)
    elif args.kind == "scatter":
        plot_scatter_fit(rows, args.x, args.y, args.group, title, args.output_dir, args.name)
    elif args.kind == "raster":
        plot_raster(rows, args.row, args.col, args.group, title, args.output_dir, args.name)
    else:
        fail("未知图类型：%s" % args.kind)


if __name__ == "__main__":
    main()
