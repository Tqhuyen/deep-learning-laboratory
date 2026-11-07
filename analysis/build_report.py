"""Build the self-contained HTML report of the recurrent comparison.

    python analysis/build_report.py

Reads every artifact under ``results/recurrent_comparison/`` and writes
``results/recurrent_comparison/report.html``. Numbers in the prose are computed
here from those artifacts, never typed by hand. Formulas are converted to MathML
at build time (``latex2mathml``) and charts are drawn by inline JavaScript from
embedded JSON, so the file opens offline in any current browser.

Requires Python 3.12+ (the prose embeds LaTeX in f-strings, PEP 701) and the
``analysis`` extra: ``pip install -e ".[analysis]"``.
"""

from __future__ import annotations

import html
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from latex2mathml.converter import convert
from scipy import stats

from laboratory.models.capacity import language_model_parameters, sequence_classifier_parameters

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "recurrent_comparison"
ASSETS = Path(__file__).resolve().parent / "report_assets"
OUTPUT = RESULTS / "report.html"

CELLS = ("rnn", "gru", "lstm")
LABEL = {"rnn": "RNN", "gru": "GRU", "lstm": "LSTM"}
MULTIPLIER = {"rnn": 1, "gru": 3, "lstm": 4}
CARRY = {"rnn": "tanh′ = 1 − h²", "gru": "cổng update z", "lstm": "cổng quên f"}
TRAINING_COMMIT = "4bc2e00"
NOTEBOOK = "https://www.kaggle.com/code/trnquanghuyn/rnn-gru-lstm-comparison"
REPOSITORY = "https://github.com/Tqhuyen/deep-learning-laboratory"


# ------------------------------------------------------------------ helpers
def load(relative: str):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def m(latex: str) -> str:
    """Inline MathML."""
    return convert(latex, display="inline")


def mb(latex: str) -> str:
    """Display MathML in an equation box."""
    return f'<div class="equation">{convert(latex, display="block")}</div>'


def f(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace("-", "−")


def s(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def sci(value: float, digits: int = 2) -> str:
    """``1.3·10⁻¹`` style, with ``digits`` significant digits."""
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    power = int(exponent)
    return mantissa if power == 0 else f"{mantissa}·10<sup>{str(power).replace('-', '−')}</sup>"


def percent(value: float) -> str:
    """Two significant digits below 10 %, one decimal above."""
    return f"{100 * value:.2g} %" if value < 0.1 else f"{100 * value:.1f} %"


def p_text(value: float, repetitions: int | None = None) -> str:
    """A bootstrap p of exactly 0 only says p < 2 / repetitions."""
    if repetitions is not None and value == 0.0:
        return f"&lt; {2 / repetitions:.4f}"
    return sci(value, 3) if value < 1e-3 else f"{value:.4f}"


def table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    head = "".join(
        f'<th class="num">{h}</th>' if i in numeric else f"<th>{h}</th>" for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f'<td class="num">{value}</td>' if c in numeric else f"<td>{value}</td>" for c, value in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def callout(kind: str, title: str, body: str) -> str:
    return f'<div class="callout {kind}"><p class="callout-title">{title}</p>{body}</div>'


def figure(title: str, chart: str, task: str | None = None, caption: str = "") -> str:
    data_task = f' data-task="{task}"' if task else ""
    note = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="figure"><h4>{title}</h4><div data-chart="{chart}"{data_task}></div>{note}</div>'


def section(identifier: str, number: str, title: str, body: str, lede: str = "") -> str:
    intro = f'<p class="section-lede">{lede}</p>' if lede else ""
    return f'<section id="{identifier}"><h2><span class="number">{number}</span>{title}</h2>{intro}{body}</section>'


def seed_interval(first: list[float], second: list[float]) -> tuple[float, float, float]:
    differences = np.asarray(first) - np.asarray(second)
    mean = float(differences.mean())
    half = float(stats.t.ppf(0.975, differences.size - 1) * differences.std(ddof=1) / math.sqrt(differences.size))
    return mean, mean - half, mean + half


def mean_sd(values: list[list[float]]) -> tuple[list[float], list[float]]:
    array = np.asarray(values, dtype=float)
    return array.mean(axis=0).tolist(), array.std(axis=0, ddof=1).tolist()


def fitted_relative(norms: np.ndarray, slope: float) -> list[float]:
    """``exp(a + λk) / g(0)`` for the least-squares line behind the stored ``λ``.

    ``norms`` are ordered by lag. The fit window follows ``decay_slope``: it stops
    at the first lag below ``1e-10`` times the largest norm.
    """
    above = norms > norms.max() * 1e-10
    used = norms if above.all() else norms[: int(np.argmin(above))]
    check, intercept = np.polyfit(np.arange(used.size, dtype=float), np.log(used), 1)
    assert abs(check - slope) < 1e-9, (check, slope)
    return (np.exp(intercept + slope * np.arange(norms.size)) / norms[0]).tolist()


def hidden_root(cell: str, vocabulary: int, embedding: int, target: int, task: str) -> dict:
    """Real root of P(H) = target for the quadratic parameter count."""
    multiplier = MULTIPLIER[cell]
    head_linear, head_constant = (2, 2) if task == "sentiment" else (vocabulary, vocabulary)
    a = multiplier
    b = multiplier * (embedding + 2) + head_linear
    c = vocabulary * embedding + head_constant - target
    root = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return {"a": a, "b": b, "c": c, "root": root}


# ------------------------------------------------------------------ data
def collect() -> dict:
    sentiment = load("sentiment/metrics.json")
    generation = load("generation/metrics.json")
    data = {
        "sentiment": {
            "summary": sentiment,
            "config": load("sentiment/config.json"),
            "curves": load("sentiment/curves.json"),
            "buckets": load("sentiment/length_buckets.json"),
        },
        "generation": {
            "summary": generation,
            "config": load("generation/config.json"),
            "curves": load("generation/curves.json"),
            "samples": (RESULTS / "generation/samples.txt").read_text(encoding="utf-8"),
        },
        "landscapes": {task: load(f"landscapes/landscape_{task}.json") for task in ("sentiment", "generation")},
        "context": load("contexts/context_sweep.json"),
        "reach": {task: load(f"reach/reach_{task}.json") for task in ("sentiment", "generation")},
    }
    return data


def derive(data: dict) -> dict:
    """Every number quoted in the prose, computed once."""
    d: dict = {}
    for task in ("sentiment", "generation"):
        summary = data[task]["summary"]
        config = data[task]["config"]
        key = "accuracy" if task == "sentiment" else "bpc"
        metrics = {
            cell: {
                "mean": summary["metrics"][cell][f"{key}_mean"],
                "sd": summary["metrics"][cell][f"{key}_std"],
                "seeds": summary["metrics"][cell]["values"],
            }
            for cell in CELLS
        }
        pairs = []
        for entry in summary["pairwise"]:
            first, second = entry["first"], entry["second"]
            estimate, low, high = seed_interval(metrics[first]["seeds"], metrics[second]["seeds"])
            difference = entry["accuracy_difference"] if task == "sentiment" else entry["bpc_difference"]
            # Mean of per-seed differences equals the difference of seed-averaged
            # scores; the stored per-window losses are float32, hence the tolerance.
            assert abs(difference - estimate) < 1e-6, (task, first, second, difference, estimate)
            pairs.append({
                "first": first,
                "second": second,
                "estimate": estimate,
                "seed": [low, high],
                "bootstrap": entry["bootstrap_interval"],
                "p": entry.get("mcnemar_p", entry.get("bootstrap_p")),
                "holm": entry["holm_p"],
            })

        vocabulary = config["vocabulary_size"]
        embedding = config["embedding_dim"]
        target = config["target_parameters"]
        matching = {}
        for cell in CELLS:
            hidden = summary["parameters"][cell]["hidden_size"]
            parameters = summary["parameters"][cell]["parameters"]
            formula = (
                sequence_classifier_parameters(cell, vocabulary, embedding, hidden)
                if task == "sentiment"
                else language_model_parameters(cell, vocabulary, embedding, hidden)
            )
            assert formula == parameters, (task, cell, formula, parameters)
            matching[cell] = {"hidden": hidden, "parameters": parameters, **hidden_root(cell, vocabulary, embedding, target, task)}
        d[task] = {"metrics": metrics, "pairs": pairs, "matching": matching}

    buckets = data["sentiment"]["buckets"]
    edges = buckets["edges"]
    labels = [f"&lt; {edges[1]}"] + [f"{edges[i]}–{edges[i + 1] - 1}" for i in range(1, len(edges) - 2)] + [f"{edges[-2]}–{edges[-1]}"]
    plain_labels = [label.replace("&lt;", "<") for label in labels]
    assert sum(buckets["counts"]) == data["sentiment"]["config"]["test_examples"]
    d["buckets"] = {"labels": labels, "plain": plain_labels, "counts": buckets["counts"], "series": buckets["series"]}

    surfaces = {}
    for task, landscape in data["landscapes"].items():
        grid = landscape["grid"]
        centre = (len(grid) - 1) // 2
        surfaces[task] = {}
        for cell in CELLS:
            entry = landscape["cells"][cell]
            array = np.asarray(entry["surface"])
            index = np.unravel_index(int(array.argmin()), array.shape)
            surfaces[task][cell] = {
                "minimum": entry["minimum"],
                "centre": entry["centre"],
                "maximum": entry["maximum"],
                "rise": entry["maximum"] - entry["centre"],
                "centre_is_minimum": index == (centre, centre),
                "argmin": (grid[index[0]], grid[index[1]]),
            }
    d["surfaces"] = surfaces

    reach = {}
    for task, payload in data["reach"].items():
        reach[task] = {"probe": payload["probe"], "cells": {}}
        for cell in CELLS:
            entry = payload["cells"][cell]
            reach[task]["cells"][cell] = entry
    d["reach"] = reach
    return d


def chart_data(data: dict, d: dict) -> dict:
    out: dict = {}
    for task, label, metric_label, difference_label, curve_key in (
        ("sentiment", "cross-entropy trên validation", "accuracy trên SST-2 dev", "chênh lệch accuracy (mô hình trái − mô hình phải)", "validation_loss"),
        ("generation", "BPC trên validation", "bits per character trên validation", "chênh lệch BPC (âm = mô hình trái tốt hơn)", "validation_bpc"),
    ):
        curves = data[task]["curves"]
        steps = [entry["step"] for entry in curves["rnn"][0]["history"]]
        curve_block = {"steps": steps, "label": label}
        for cell in CELLS:
            mean, sd = mean_sd([[entry[curve_key] for entry in run["history"]] for run in curves[cell]])
            curve_block[cell] = {"mean": mean, "sd": sd}
        out[task] = {
            "metricLabel": metric_label,
            "differenceLabel": difference_label,
            "metrics": d[task]["metrics"],
            "curves": curve_block,
            "pairs": [
                {
                    "label": f"{LABEL[pair['first']]} − {LABEL[pair['second']]}",
                    "estimate": pair["estimate"],
                    "seed": pair["seed"],
                    "bootstrap": pair["bootstrap"],
                    "detail": (
                        f"McNemar p = {pair['p']:.3g}, Holm p = {pair['holm']:.3g}"
                        if task == "sentiment"
                        else f"p bootstrap < {2 / data[task]['config']['bootstrap_repetitions']:.4f}"
                        if pair["p"] == 0.0
                        else f"p bootstrap = {pair['p']:.3g}"
                    ),
                }
                for pair in d[task]["pairs"]
            ],
        }
    out["sentiment"]["lengthBuckets"] = {
        "labels": d["buckets"]["plain"],
        "counts": d["buckets"]["counts"],
        "series": d["buckets"]["series"],
    }
    out["landscapes"] = {
        task: {
            "grid": landscape["grid"],
            "cells": {
                cell: {key: landscape["cells"][cell][key] for key in ("surface", "minimum", "maximum", "centre")}
                for cell in CELLS
            },
        }
        for task, landscape in data["landscapes"].items()
    }
    out["context"] = {
        "contexts": data["context"]["contexts"],
        "trained": data["context"]["trained_context"],
        "results": data["context"]["results"],
    }
    out["reach"] = {}
    for task, payload in data["reach"].items():
        cells = {}
        for cell in CELLS:
            norms = np.asarray(payload["cells"][cell]["norms"], dtype=float)[::-1]
            relative = norms / norms[0]
            slope = payload["cells"][cell]["slope"]
            cells[cell] = {
                "lags": list(range(norms.size)),
                "relative": [float(v) if v > 1e-12 else None for v in relative],
                "fitted": fitted_relative(norms, slope),
                "slope": slope,
            }
        out["reach"][task] = {"cells": cells}
    return out


# ------------------------------------------------------------------ content
def build_body(data: dict, d: dict) -> str:
    sc, gc = data["sentiment"]["config"], data["generation"]["config"]
    sm, gm = d["sentiment"]["metrics"], d["generation"]["metrics"]
    sp = {(p["first"], p["second"]): p for p in d["sentiment"]["pairs"]}
    gp = {(p["first"], p["second"]): p for p in d["generation"]["pairs"]}
    sr, gr = d["reach"]["sentiment"]["cells"], d["reach"]["generation"]["cells"]
    surfaces = d["surfaces"]
    buckets = d["buckets"]
    tokens_per_run = gc["max_steps"] * gc["batch_size"] * gc["block_size"]
    repetitions_g = gc["bootstrap_repetitions"]
    t_critical = float(stats.t.ppf(0.975, len(sc["seeds"]) - 1))

    def half_life(entry: dict) -> str:
        return f"{entry['half_life_steps']:.1f}" if entry["half_life_steps"] else "—"

    drop = {
        cell: buckets["series"][cell][1] - buckets["series"][cell][-1] for cell in CELLS
    }

    parts: list[str] = []

    # Hero ---------------------------------------------------------------
    parts.append(f"""
<header class="hero" id="top">
  <p class="eyebrow">Deep Learning Laboratory · báo cáo thí nghiệm</p>
  <h1>RNN, GRU và LSTM ở cùng ngân sách tham số</h1>
  <p class="lede">Hai bài toán — phân loại cảm xúc trên SST-2 và sinh văn bản mức ký tự trên Tiny Shakespeare —
  ba kiến trúc, ba seed, cùng dữ liệu, cùng optimizer, cùng số bước. Báo cáo tách rõ ba loại khẳng định:
  điều được chứng minh bằng toán học, điều được thống kê ủng hộ <em>có điều kiện</em>, và điều chưa thể kết luận.</p>
  <dl class="meta">
    <div><dt>Phần cứng</dt><dd>Kaggle · Tesla T4 (15.6 GB) · PyTorch 2.11 · Python 3.13</dd></div>
    <div><dt>Mã huấn luyện</dt><dd><a href="{REPOSITORY}/commit/{TRAINING_COMMIT}">commit {TRAINING_COMMIT}</a></dd></div>
    <div><dt>Notebook</dt><dd><a href="{NOTEBOOK}">kaggle.com/…/rnn-gru-lstm-comparison</a></dd></div>
    <div><dt>Ngày tạo báo cáo</dt><dd>{date.today().isoformat()}</dd></div>
  </dl>
</header>""")

    # Summary ------------------------------------------------------------
    parts.append(section("tom-tat", "0", "Tóm tắt", f"""
<div class="cards">
  <div class="card"><h3>Phân loại cảm xúc</h3>
    <p class="headline">GRU {f(sm['gru']['mean'], 3)} ≈ LSTM {f(sm['lstm']['mean'], 3)} &gt; RNN {f(sm['rnn']['mean'], 3)}</p>
    <p>Hai cell có cổng hơn RNN {f(-100 * sp[('rnn', 'gru')]['estimate'], 1)} điểm accuracy; CI t theo seed và CI bootstrap đều không chứa 0.
    GRU − LSTM = {s(sp[('gru', 'lstm')]['estimate'])}: không phân biệt được.</p></div>
  <div class="card"><h3>Sinh văn bản (BPC, thấp hơn là tốt)</h3>
    <p class="headline">RNN {f(gm['rnn']['mean'], 3)} &lt; GRU {f(gm['gru']['mean'], 3)} &lt; LSTM {f(gm['lstm']['mean'], 3)}</p>
    <p>LSTM kém hơn ở cả ba seed (CI theo seed không chứa 0). RNN − GRU có CI theo seed
    [{s(gp[('rnn', 'gru')]['seed'][0], 3)}, {s(gp[('rnn', 'gru')]['seed'][1], 3)}] chứa 0: chưa đủ để nói RNN hơn GRU.</p></div>
  <div class="card"><h3>Cơ chế: tầm với của gradient</h3>
    <p class="headline">g(8)/g(0): {" / ".join(f(sr[c]['retention']['8'], 2) for c in CELLS)}</p>
    <p>Sau 8 token, độ nhạy của trạng thái cuối còn {" / ".join(percent(sr[c]['retention']['8']) for c in CELLS)} (RNN / GRU / LSTM): cell có cổng với xa hơn,
    khớp với kết quả sentiment. Ở sinh văn bản, sau 64 ký tự còn {" / ".join(percent(gr[c]['retention']['64']) for c in CELLS)}:
    LSTM với xa nhất nhưng BPC tệ nhất — tầm với không giải thích được bài này.</p></div>
  <div class="card"><h3>Giới hạn</h3>
    <p class="headline">Không có kết luận phổ quát</p>
    <p>Mọi kết quả thực nghiệm đều gắn với dữ liệu, ngân sách và seed cụ thể. Mặt loss 3D chỉ là chẩn đoán cục bộ quanh một nghiệm.</p></div>
</div>
{callout("correction", '<span class="badge">Đính chính</span> Ba điểm sai trong báo cáo trước', f'''
<ol>
<li><b>Phép đo tầm với gradient bị lỗi.</b> Lần chạy Kaggle lấy đạo hàm theo một <em>bản sao</em> các output đã được stack;
bản sao đó không nối với {m(r"h_T")} ở các bước trước, nên mọi timestep trừ bước cuối có gradient bằng 0, và “gradient slope” cũ
chỉ phản ánh độ dài chuỗi. Đã sửa để đạo hàm theo input {m(r"x_t")} trên chuỗi không padding, thêm test so với nghiệm giải tích của
RNN tuyến tính, và đo lại từ checkpoint seed 0. Kết luận cũ “tầm với không giải thích thứ tự” là <b>sai với sentiment</b>.
Accuracy, BPC và mọi kiểm định không bị ảnh hưởng vì phép đo chạy sau huấn luyện.</li>
<li><b>Mô tả mặt loss.</b> Báo cáo trước viết LSTM có mặt phẳng nhất ở sentiment và mọi checkpoint nằm ở điểm thấp nhất của lát cắt.
Số liệu cho thấy ngược lại: RNN có độ dâng thấp nhất ({f(surfaces['sentiment']['rnn']['rise'], 3)}), và ở sentiment điểm thấp nhất của lát cắt
RNN và GRU không nằm tại checkpoint (loss được đo trên tập đánh giá, không phải tập huấn luyện). Ở sentiment, mặt phẳng nhất còn đổi
theo cặp hướng ngẫu nhiên (mục 6), nên không nên đọc độ phẳng từ một lát cắt.</li>
<li><b>Accuracy theo độ dài câu.</b> Báo cáo trước viết RNN mất khoảng 4 điểm còn GRU/LSTM chỉ 1–2 điểm. Thực tế, từ nhóm 8–15 token
xuống nhóm {buckets['plain'][-1]}, RNN mất {f(100 * drop['rnn'], 1)} điểm, GRU mất {f(100 * drop['gru'], 1)}, LSTM mất {f(100 * drop['lstm'], 1)}.</li>
</ol>''')}
"""))

    # 1. Question --------------------------------------------------------
    parts.append(section("cau-hoi", "1", "“Tốt hơn” nghĩa là gì", f"""
<p>Câu “kiến trúc A tốt hơn B” chỉ có nghĩa khi đã cố định bốn thứ. Thiếu một trong bốn, câu đó không thể đúng hay sai.</p>
{table(["Trục", "Câu hỏi", "Lựa chọn trong thí nghiệm này"], [
    ["Lớp bài toán", "Dữ liệu nào, phân phối nào?", "SST-2 (câu ngắn, nhãn nhị phân) và Tiny Shakespeare (ký tự)"],
    ["Ngân sách", "Tham số, số bước, số token?", f"Cùng số tham số: {sc['target_parameters']:,} (sentiment), {gc['target_parameters']:,} (sinh văn bản); cùng số bước optimizer"],
    ["Tiêu chí", "Đo bằng gì?", "Accuracy trên SST-2 dev; bits per character trên phần validation"],
    ["Giả định", "Seed, siêu tham số, khởi tạo?", "3 seed; AdamW, cosine warmup, lr 3·10⁻⁴, weight decay 0.01, clip 1.0; khởi tạo mặc định"],
])}
<p>Định lý <em>No free lunch</em> (Wolpert &amp; Macready, 1997) nói rằng lấy trung bình trên mọi phân phối, không thuật toán nào tốt hơn thuật toán nào.
Vì vậy mọi kết luận thực nghiệm dưới đây đều là mệnh đề có điều kiện: “với dữ liệu này, ngân sách này, tiêu chí này”.</p>
"""))

    # 2. Design ----------------------------------------------------------
    sm_match, gm_match = d["sentiment"]["matching"], d["generation"]["matching"]

    def matching_rows(matching: dict, target: int) -> list[list[str]]:
        rows = []
        for cell in CELLS:
            entry = matching[cell]
            rows.append([
                LABEL[cell],
                str(MULTIPLIER[cell]),
                f"{entry['root']:.2f}",
                str(entry["hidden"]),
                f"{entry['parameters']:,}",
                f"{s(100 * (entry['parameters'] - target) / target, 3)} %",
                f"{entry['hidden'] / matching['rnn']['hidden']:.3f}",
                f"{1 / math.sqrt(MULTIPLIER[cell]):.3f}",
            ])
        return rows

    matching_headers = ["Cell", "m", "H* (nghiệm thực)", "H chọn", "P(H)", "lệch so với P*", "H / H<sub>RNN</sub>", "tiệm cận 1/√m"]
    parts.append(section("thiet-ke", "2", "Thiết kế thí nghiệm", f"""
<h3>Dữ liệu và cấu hình</h3>
{table(["", "Phân loại cảm xúc", "Sinh văn bản"], [
    ["Dữ liệu", "SST-2 (GLUE), câu tiếng Anh, nhãn 0/1", "Tiny Shakespeare, mức ký tự"],
    ["Tách tập", f"{sc['train_examples']:,} train · {sc['validation_examples']:,} validation · {sc['test_examples']} dev (đánh giá)", "90 % đầu train · 10 % cuối validation (liên tục)"],
    ["Từ vựng V", f"{sc['vocabulary_size']:,} từ (tần suất ≥ {sc['min_frequency']})", f"{gc['vocabulary_size']} ký tự"],
    ["Embedding E", str(sc["embedding_dim"]), str(gc["embedding_dim"])],
    ["Ngân sách P*", f"{sc['target_parameters']:,}", f"{gc['target_parameters']:,}"],
    ["Độ dài", f"cắt tại {sc['max_len']} token", f"cửa sổ {gc['block_size']} ký tự"],
    ["Bước × batch", f"{sc['max_steps']:,} × {sc['batch_size']}", f"{gc['max_steps']:,} × {gc['batch_size']} = {tokens_per_run:,} token mỗi lần chạy"],
    ["Seed", ", ".join(map(str, sc["seeds"])), ", ".join(map(str, gc["seeds"]))],
    ["Đầu ra", "trạng thái ẩn tại token thật cuối cùng → 2 lớp", "trạng thái ẩn mọi bước → phân phối ký tự kế tiếp"],
])}
<h3>Khớp số tham số — đây là phần chứng minh được</h3>
<p>Một lớp hồi quy với input {m("I")} và trạng thái ẩn {m("H")} có số tham số</p>
{mb(r"P_{\text{cell}}(H) = m\,(IH + H^{2} + 2H),\qquad m = \begin{cases} 1 & \text{RNN} \\ 3 & \text{GRU} \\ 4 & \text{LSTM} \end{cases}")}
<p>vì một đơn vị GRU sở hữu 3 bộ ma trận cổng, một đơn vị LSTM 4. Cộng embedding và đầu ra:</p>
{mb(r"P_{\text{sentiment}}(H) = VE + m\,(EH + H^{2} + 2H) + 2H + 2,\qquad P_{\text{LM}}(H) = VE + m\,(EH + H^{2} + 2H) + VH + V")}
<p>Đặt {m(r"P(H) = P^{\star}")} cho ra một phương trình bậc hai theo {m("H")}, có đúng một nghiệm dương:</p>
{mb(r"a H^{2} + b H + c_{0} = 0,\quad a = m,\quad b = m(E + 2) + c,\quad c_{0} = VE + d - P^{\star}\;\Longrightarrow\; H^{\star} = \frac{-b + \sqrt{b^{2} - 4 a c_{0}}}{2a}")}
<p>với {m("(c, d) = (2, 2)")} cho bộ phân loại hai lớp và {m("(c, d) = (V, V)")} cho mô hình ngôn ngữ. Khi {m(r"H^{2}")} chi phối,
{m(r"H_m \approx \sqrt{P^{\star}/m}")}, nên tỉ lệ số đơn vị tiến tới {m(r"1/\sqrt{3} \approx 0.577")} (GRU) và {m(r"1/2")} (LSTM).
Đây là hệ quả đại số, không phải kết quả thực nghiệm.</p>
<h4>Sentiment (V = {sc['vocabulary_size']:,}, E = {sc['embedding_dim']}, P* = {sc['target_parameters']:,})</h4>
{table(matching_headers, matching_rows(sm_match, sc["target_parameters"]), numeric={1, 2, 3, 4, 5, 6, 7})}
<h4>Sinh văn bản (V = {gc['vocabulary_size']}, E = {gc['embedding_dim']}, P* = {gc['target_parameters']:,})</h4>
{table(matching_headers, matching_rows(gm_match, gc["target_parameters"]), numeric={1, 2, 3, 4, 5, 6, 7})}
<p class="caption">H chọn là số nguyên cho P(H) gần P* nhất; P(H) được kiểm tra khớp chính xác với số tham số đếm trên module thật
(<code>laboratory/models/capacity.py</code>, có unit test). Ở sentiment, embedding {sc['vocabulary_size']:,}×{sc['embedding_dim']} chiếm
{100 * sc['vocabulary_size'] * sc['embedding_dim'] / sc['target_parameters']:.0f} % ngân sách nên tỉ lệ lệch xa tiệm cận hơn.</p>
{callout("theorem", '<span class="badge">Hệ quả</span> Cùng P nghĩa là khác H', f'''
<p>So sánh “cùng số đơn vị ẩn” sẽ cho LSTM gần 4 lần tham số hồi quy so với RNN. Khớp P buộc LSTM chỉ có khoảng một nửa số đơn vị
của RNN ({gm_match['lstm']['hidden']} so với {gm_match['rnn']['hidden']} ở sinh văn bản). Đó là đánh đổi được thiết kế có chủ ý, và cũng là
giả thuyết chính cho kết quả sinh văn bản ở mục 4.</p>''')}
<h3>Thống kê</h3>
<p><b>Trung bình ± độ lệch chuẩn theo seed</b> (mẫu, {m("n - 1")}). <b>Bootstrap ghép cặp</b> lấy mẫu lại có hoàn lại các ví dụ của tập đánh giá.
Với sentiment, {m("a_i")} là độ đúng của câu {m("i")} lấy trung bình qua ba seed (một trong 0, ⅓, ⅔, 1); với sinh văn bản, {m("a_i")} là BPC của cửa sổ
{m("i")} lấy trung bình qua ba seed. {m("B")} = {sc['bootstrap_repetitions']:,} (sentiment) và {repetitions_g:,} (sinh văn bản) lần lấy mẫu:</p>
{mb(r"\hat{\Delta} = \frac{1}{n}\sum_{i=1}^{n} (a_i - b_i),\qquad \mathrm{CI}_{95\%} = \left[\,q_{0.025}(\hat{\Delta}^{*}),\; q_{0.975}(\hat{\Delta}^{*})\,\right],\qquad p_{\text{boot}} = 2\min\left(\Pr{}^{*}(\hat{\Delta}^{*} \le 0),\; \Pr{}^{*}(\hat{\Delta}^{*} \ge 0)\right)")}
<p>Với sentiment, <b>McNemar chính xác</b> dùng dự đoán của seed 0 ({m("b")}: số câu chỉ A đúng, {m("c")}: số câu chỉ B đúng); với sinh văn bản,
p là {m(r"p_{\text{boot}}")} ở trên. <b>Hiệu chỉnh Holm</b> áp dụng cho ba so sánh của mỗi bài toán ({m("K = 3")}, {m(r"p_{(1)} \le p_{(2)} \le p_{(3)}")}):</p>
{mb(r"p_{\text{McNemar}} = \min\left(1,\; 2\sum_{k=0}^{\min(b,c)} \binom{b+c}{k}\, 2^{-(b+c)}\right),\qquad p^{\text{Holm}}_{(j)} = \max_{i \le j}\, \min\left(1,\; (K - i + 1)\, p_{(i)}\right)")}
<p><b>Khoảng t theo seed</b> — kiểm định chặt nhất ở đây, vì mỗi seed là một lần lặp lại toàn bộ quá trình huấn luyện:</p>
{mb(r"\bar{d} \pm t_{0.975,\,n-1}\,\frac{s_d}{\sqrt{n}},\qquad d_s = \mathrm{metric}_A^{(s)} - \mathrm{metric}_B^{(s)},\qquad t_{0.975,\,2} = " + f"{t_critical:.3f}")}
{callout("open", '<span class="badge">Giới hạn của kiểm định</span> Bootstrap và McNemar không tính đến seed', f'''
<p>Bootstrap và McNemar chỉ đo dao động giữa các <em>ví dụ</em> của một tập đánh giá cố định, với mô hình đã huấn luyện xong; chúng không đo
dao động giữa các lần huấn luyện. Khoảng t theo seed đo đúng dao động đó, nhưng với n = 3 thì hệ số {t_critical:.3f} làm khoảng rất rộng.
Một chênh lệch vượt qua cả hai loại kiểm định là bằng chứng mạnh; một chênh lệch chỉ vượt bootstrap thì chưa đủ.</p>''')}
"""))

    # 3. Sentiment --------------------------------------------------------
    sentiment_rows = []
    for cell in CELLS:
        metric = sm[cell]
        sentiment_rows.append([
            LABEL[cell],
            str(sm_match[cell]["hidden"]),
            f"{sm_match[cell]['parameters']:,}",
            f"{f(metric['mean'])} ± {f(metric['sd'])}",
            " · ".join(f(v) for v in metric["seeds"]),
            f(data["sentiment"]["summary"]["metrics"][cell]["nll_mean"]),
        ])
    pair_rows = []
    for pair in d["sentiment"]["pairs"]:
        pair_rows.append([
            f"{LABEL[pair['first']]} − {LABEL[pair['second']]}",
            s(pair["estimate"]),
            f"[{s(pair['bootstrap'][0])}, {s(pair['bootstrap'][1])}]",
            f"[{s(pair['seed'][0])}, {s(pair['seed'][1])}]",
            p_text(pair["p"]),
            p_text(pair["holm"]),
        ])
    test_examples = sc["test_examples"]
    correct = {cell: round(sum(sm[cell]["seeds"]) * test_examples) for cell in CELLS}
    tie_sentence = (
        f"GRU và LSTM có cùng tổng số câu đúng qua ba seed ({correct['gru']:,} trên {3 * test_examples:,}), "
        "nên chênh lệch trung bình đúng bằng 0."
        if correct["gru"] == correct["lstm"]
        else f"GRU − LSTM = {s(sp[('gru', 'lstm')]['estimate'])}."
    )
    parts.append(section("cam-xuc", "3", "Phân loại cảm xúc (SST-2)", f"""
{table(["Cell", "H", "Tham số", "Accuracy (TB ± SD)", "Từng seed (0 · 1 · 2)", "NLL"], sentiment_rows, numeric={1, 2, 3, 4, 5})}
<div class="figure-grid">
{figure("Accuracy trên dev: trung bình ± SD, chấm trắng là từng seed", "dots", "sentiment")}
{figure("Cross-entropy trên validation trong lúc huấn luyện", "curve", "sentiment", f"Trung bình ± SD qua 3 seed; mỗi điểm đo trên {sc['validation_batches']} batch validation.")}
</div>
<h3>So sánh từng cặp</h3>
{table(["Cặp", "Δ accuracy", "CI 95 % bootstrap (ví dụ)", "CI 95 % t (seed)", "McNemar p", "Holm p"], pair_rows, numeric={1, 2, 3, 4, 5})}
{figure("Khoảng tin cậy của chênh lệch", "forest", "sentiment", "Vạch cam đậm: CI t theo seed. Vạch xanh mảnh: CI bootstrap theo ví dụ. Chấm: ước lượng. Đường đứt: 0.")}
{callout("evidence", '<span class="badge">Bằng chứng có điều kiện</span> Cổng giúp ở SST-2', f'''
<p>RNN thua cả hai cell có cổng {f(-100 * sp[('rnn', 'gru')]['estimate'], 1)} điểm. Cả CI theo seed
([{s(sp[('rnn', 'gru')]['seed'][0], 3)}, {s(sp[('rnn', 'gru')]['seed'][1], 3)}] và [{s(sp[('rnn', 'lstm')]['seed'][0], 3)}, {s(sp[('rnn', 'lstm')]['seed'][1], 3)}])
lẫn McNemar sau Holm ({p_text(sp[('rnn', 'gru')]['holm'])} và {p_text(sp[('rnn', 'lstm')]['holm'])}) đều loại giả thuyết “không khác biệt”.
{tie_sentence} Mọi khoảng đều chứa 0 và Holm p = {f(sp[('gru', 'lstm')]['holm'], 3)}.</p>''')}
<h3>Theo độ dài câu</h3>
{figure("Accuracy theo nhóm độ dài (trung bình theo seed, trên từng ví dụ)", "length", None)}
{table(["Nhóm (token)", "Số câu", "RNN", "GRU", "LSTM"], [
    [buckets['labels'][i], str(buckets['counts'][i])] + [f(buckets['series'][cell][i], 3) for cell in CELLS]
    for i in range(len(buckets['labels']))
], numeric={1, 2, 3, 4})}
<p>Từ nhóm 8–15 xuống nhóm {buckets['labels'][-1]}, RNN mất {f(100 * drop['rnn'], 1)} điểm, GRU {f(100 * drop['gru'], 1)}, LSTM {f(100 * drop['lstm'], 1)}.
Cả ba đều kém đi ở câu dài, RNN kém nhanh nhất. Nhóm dài nhất chỉ có {buckets['counts'][-1]} câu (mỗi câu chiếm {100 / buckets['counts'][-1]:.1f} điểm accuracy),
nhóm ngắn nhất {buckets['counts'][0]} câu ({100 / buckets['counts'][0]:.1f} điểm mỗi câu): đây là xu hướng, chưa phải kiểm định.</p>
"""))

    # 4. Generation -------------------------------------------------------
    generation_rows = []
    for cell in CELLS:
        metric = gm[cell]
        generation_rows.append([
            LABEL[cell],
            str(gm_match[cell]["hidden"]),
            f"{gm_match[cell]['parameters']:,}",
            f"{f(metric['mean'])} ± {f(metric['sd'])}",
            " · ".join(f(v) for v in metric["seeds"]),
        ])
    gpair_rows = []
    for pair in d["generation"]["pairs"]:
        gpair_rows.append([
            f"{LABEL[pair['first']]} − {LABEL[pair['second']]}",
            s(pair["estimate"]),
            f"[{s(pair['bootstrap'][0])}, {s(pair['bootstrap'][1])}]",
            f"[{s(pair['seed'][0])}, {s(pair['seed'][1])}]",
            p_text(pair["p"], repetitions_g),
            p_text(pair["holm"], repetitions_g),
        ])
    samples = data["generation"]["samples"].split("## ")[1:]
    sample_blocks = "".join(
        f'<div class="figure"><h4>{LABEL[block.split(chr(10), 1)[0].strip()]}</h4><pre class="sample">{html.escape(block.split(chr(10), 1)[1].strip())}</pre></div>'
        for block in samples
    )
    parts.append(section("sinh-van-ban", "4", "Sinh văn bản mức ký tự (Tiny Shakespeare)", f"""
<p>Thước đo là bits per character, {m(r"\mathrm{BPC} = \mathrm{CE}/\ln 2")}: số bit trung bình để mã hoá một ký tự; thấp hơn là tốt hơn.
Mỗi lần chạy thấy {tokens_per_run:,} token huấn luyện.</p>
{table(["Cell", "H", "Tham số", "BPC (TB ± SD)", "Từng seed (0 · 1 · 2)"], generation_rows, numeric={1, 2, 3, 4})}
<div class="figure-grid">
{figure("BPC trên validation: trung bình ± SD, chấm trắng là từng seed", "dots", "generation")}
{figure("BPC trên validation trong lúc huấn luyện", "curve", "generation", f"Trung bình ± SD qua 3 seed; mỗi điểm đo trên {gc['validation_batches']} batch validation.")}
</div>
<h3>So sánh từng cặp</h3>
{table(["Cặp", "Δ BPC", "CI 95 % bootstrap (cửa sổ)", "CI 95 % t (seed)", "p bootstrap", "Holm p"], gpair_rows, numeric={1, 2, 3, 4, 5})}
{figure("Khoảng tin cậy của chênh lệch BPC", "forest", "generation", "Vạch cam đậm: CI t theo seed. Vạch xanh mảnh: CI bootstrap theo cửa sổ validation. Âm nghĩa là mô hình bên trái tốt hơn.")}
{callout("evidence", '<span class="badge">Bằng chứng có điều kiện</span> LSTM kém ở ngân sách này; RNN − GRU chưa phân định', f'''
<p>LSTM tệ hơn ở cả ba seed; CI theo seed của RNN − LSTM là [{s(gp[('rnn', 'lstm')]['seed'][0], 3)}, {s(gp[('rnn', 'lstm')]['seed'][1], 3)}]
và của GRU − LSTM là [{s(gp[('gru', 'lstm')]['seed'][0], 3)}, {s(gp[('gru', 'lstm')]['seed'][1], 3)}], đều không chứa 0.
RNN thấp hơn GRU {f(-gp[('rnn', 'gru')]['estimate'], 4)} BPC; bootstrap theo cửa sổ cho p {p_text(gp[('rnn', 'gru')]['p'], repetitions_g)} nhưng
CI theo seed [{s(gp[('rnn', 'gru')]['seed'][0], 3)}, {s(gp[('rnn', 'gru')]['seed'][1], 3)}] chứa 0 — bootstrap lấy mẫu cửa sổ của <em>một</em> tập
validation cố định nên hẹp hơn nhiều so với dao động giữa các lần huấn luyện.</p>''')}
<h3>Mẫu sinh (cùng prompt “ROMEO:”, temperature 0.8, top-k 40, cùng seed lấy mẫu)</h3>
<div class="figure-grid">{sample_blocks}</div>
"""))

    # 5. Mechanism --------------------------------------------------------
    def reach_rows(task: str, lags: tuple[int, ...]) -> list[list[str]]:
        rows = []
        for cell in CELLS:
            entry = d["reach"][task]["cells"][cell]
            carry = entry["carry_factor"]
            row = [
                LABEL[cell],
                str(entry["hidden_size"]),
                f(entry["slope"], 4),
                f(entry["per_step_retention"], 3),
                half_life(entry),
            ]
            row += [sci(entry["retention"][str(lag)]) for lag in lags]
            row += [f"{CARRY[cell]}: {f(carry['mean'], 3)}", percent(carry["fraction_above_0.9"])]
            row.append(f(entry["recurrent_spectral_norm"], 2) if cell == "rnn" else "—")
            rows.append(row)
        return rows

    sentiment_probe = d["reach"]["sentiment"]["probe"]
    generation_probe = d["reach"]["generation"]["probe"]
    reach_headers = lambda lags: ["Cell", "H", "λ", "e<sup>λ</sup>/bước", "½-life (bước)"] + [f"g({lag})/g(0)" for lag in lags] + ["TB hệ số mang", "&gt; 0.9", "σ<sub>max</sub>(U)"]
    rnn_early = 1 - sr["rnn"]["retention"]["4"]
    rnn_early_fit = 1 - math.exp(4 * sr["rnn"]["slope"])

    def carry_bound(task: str, cell: str) -> dict:
        """Counting bound on the direct carry path (derivation in the prose)."""
        entry = d["reach"][task]["cells"][cell]
        length = d["reach"][task]["probe"]["length"]
        needed = 1 - entry["slope"] / math.log(0.9)
        observed = entry["carry_factor"]["fraction_above_0.9"]
        return {"needed": needed, "observed": observed, "share": observed * length / (needed * (length - 1))}

    bound = {(task, cell): carry_bound(task, cell) for task in ("sentiment", "generation") for cell in ("gru", "lstm")}
    # The prose below says the bound is tight on sentiment and vacuous on generation.
    assert all(bound[("sentiment", cell)]["share"] < 0.02 for cell in ("gru", "lstm")), bound
    assert all(bound[("generation", cell)]["share"] > 0.25 for cell in ("gru", "lstm")), bound
    bound_rows = []
    for (task, cell), value in bound.items():
        entry = d["reach"][task]["cells"][cell]
        bound_rows.append([
            "Sentiment" if task == "sentiment" else "Sinh văn bản",
            LABEL[cell],
            f"{CARRY[cell]}: {f(entry['carry_factor']['mean'], 3)}",
            f(entry["slope"], 4),
            percent(value["needed"]),
            percent(value["observed"]),
            percent(value["share"]) if value["share"] < 1 else "≥ 100 % (không ràng buộc)",
        ])

    def reach_order(task: str, lag: int) -> list[str]:
        return sorted(CELLS, key=lambda cell: -d["reach"][task]["cells"][cell]["retention"][str(lag)])

    # The table below names these orders; fail loudly if new data changes them.
    assert reach_order("sentiment", 16) == ["lstm", "gru", "rnn"], reach_order("sentiment", 16)
    assert reach_order("generation", 64) == ["lstm", "rnn", "gru"], reach_order("generation", 64)
    longest = max(gr[cell]["retention"]["64"] for cell in CELLS)
    parts.append(section("co-che", "5", "Cơ chế: gradient đi được bao xa", f"""
<h3>Lý thuyết</h3>
<p>Với RNN thuần, {m(r"h_k = \tanh(W x_k + U h_{k-1} + b)")}, Jacobian một bước là {m(r"\partial h_k / \partial h_{k-1} = \operatorname{diag}(1 - h_k^{2})\,U")}. Nhân chuỗi:</p>
{mb(r"\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \;\le\; \prod_{k=t+1}^{T} \left\| \operatorname{diag}(1 - h_k^{2}) \right\|_2 \|U\|_2 \;\le\; \sigma_{\max}(U)^{\,T-t}")}
{callout("theorem", '<span class="badge">Định lý</span> Điều kiện đủ cho gradient tắt (Pascanu, Mikolov &amp; Bengio, 2013)', f'''
<p>Vì {m(r"0 < \tanh' \le 1")}, nếu {m(r"\sigma_{\max}(U) < 1")} thì độ nhạy của {m("h_T")} theo {m("h_t")} giảm ít nhất theo cấp số nhân với độ trễ.
Đây là điều kiện <em>đủ</em>, không phải điều kiện cần: khi {m(r"\sigma_{\max}(U) \ge 1")} định lý không khẳng định gì cả.</p>''')}
<p>Hai kiến trúc có cổng thêm một đường mang thông tin trực tiếp, nhân với một cổng ở mỗi bước:</p>
{mb(r"\text{LSTM: } c_k = f_k \odot c_{k-1} + i_k \odot g_k,\qquad \text{GRU: } h_k = (1 - z_k) \odot n_k + z_k \odot h_{k-1}")}
<p>Phần đạo hàm đi theo đường đó sau {m("T - t")} bước là {m(r"\operatorname{diag}\left(\prod_{k} f_k\right)")} (LSTM) hoặc
{m(r"\operatorname{diag}\left(\prod_{k} z_k\right)")} (GRU): nếu cổng gần 1, thông tin đi xa. Đây là đại số chính xác của quy tắc cập nhật.
Đạo hàm đầy đủ còn có các đường đi qua ma trận hồi quy vào các cổng; định lý trên không nói gì về chúng.</p>
<h3>Phép đo</h3>
<p>Trên chuỗi <b>không padding</b> (để bước cuối là token thật), đo độ nhạy của trạng thái cuối theo từng input:</p>
{mb(r"g(k) = \mathbb{E}\left\| \frac{\partial \lVert h_T \rVert^{2}}{\partial x_{T-k}} \right\|,\qquad \log g(k) \approx a + \lambda k,\qquad t_{1/2} = \frac{\ln 2}{-\lambda}")}
<p>{m(r"e^{\lambda}")} là tỉ lệ độ nhạy còn giữ lại sau mỗi bước, tính trung bình trên cả chuỗi. Phép đo được kiểm chứng bằng test: với RNN tuyến tính
{m(r"W = I,\; U = aI")} và input rất nhỏ, {m(r"h_T = \sum_t a^{T-t} x_t")} nên phải đo được đúng {m(r"\lambda = \ln a")}; test cũng kiểm tra
LSTM có cổng quên mở hoàn toàn cho {m(r"\lambda \approx 0")} và đóng hoàn toàn cho {m(r"\lambda \ll 0")}.</p>
<p class="caption">Đo trên checkpoint seed 0 (Kaggle chỉ lưu checkpoint seed 0): sentiment {sentiment_probe['sequences']} câu dev cắt còn
{sentiment_probe['length']} token; sinh văn bản {generation_probe['sequences']} cửa sổ validation {generation_probe['length']} ký tự. Mỗi số là một mô hình, không có khoảng tin cậy.</p>
<div class="figure-grid">
{figure("Sentiment: g(k)/g(0), đường đứt là đường thẳng bình phương tối thiểu", "reach", "sentiment")}
{figure("Sinh văn bản: g(k)/g(0), đường đứt là đường thẳng bình phương tối thiểu", "reach", "generation")}
</div>
<p>Trên thang log, tắt dần theo cấp số nhân là một đường thẳng. Dữ liệu chỉ gần thẳng: ở sentiment, RNN mất {percent(rnn_early)} độ nhạy trong
4 bước đầu, trong khi đường khớp chỉ dự đoán {percent(rnn_early_fit)}, rồi giảm chậm lại. Vì vậy λ là độ dốc trung bình; các cột
{m("g(k)/g(0)")} dưới đây đọc thẳng từ dữ liệu, không qua mô hình mũ, và cho cùng thứ tự.</p>
<h4>Sentiment ({sentiment_probe['length']} token)</h4>
{table(reach_headers((8, 16)), reach_rows("sentiment", (8, 16)), numeric={1, 2, 3, 4, 5, 6, 8, 9})}
<h4>Sinh văn bản ({generation_probe['length']} ký tự)</h4>
{table(reach_headers((16, 64)), reach_rows("generation", (16, 64)), numeric={1, 2, 3, 4, 5, 6, 8, 9})}
<p class="caption">“Hệ số mang” là đại lượng nhân vào gradient trên đường trực tiếp ở mỗi bước: tanh′ cho RNN, z cho GRU, f cho LSTM.
“&gt; 0.9” là tỉ lệ bộ ba (đơn vị, bước, chuỗi) có hệ số đó lớn hơn 0.9.</p>
{callout("evidence", '<span class="badge">Đối chiếu lý thuyết với đo đạc</span> Điều kiện đủ không thỏa, nhưng gradient vẫn tắt', f'''
<p>RNN đã huấn luyện có {m(r"\sigma_{\max}(U)")} = {f(sr['rnn']['recurrent_spectral_norm'], 2)} (sentiment) và {f(gr['rnn']['recurrent_spectral_norm'], 2)} (sinh văn bản):
điều kiện đủ của định lý <b>không</b> thỏa, nên định lý không bảo đảm gì. Dù vậy độ nhạy vẫn giảm trung bình
{f(100 * (1 - sr['rnn']['per_step_retention']), 1)} % và {f(100 * (1 - gr['rnn']['per_step_retention']), 1)} % mỗi bước. Chặn trên vẫn đúng nhưng lỏng:
nó dùng giá trị lớn nhất của tanh′ (bằng 1) và hướng khuếch đại mạnh nhất của U ở mọi bước, trong khi tanh′ trung bình chỉ
{f(sr['rnn']['carry_factor']['mean'], 3)} và {f(gr['rnn']['carry_factor']['mean'], 3)}, và không có gì buộc gradient nằm trên hướng đó ở mọi bước.
Đây là giải thích khả dĩ cho khoảng cách giữa chặn trên và thực tế, chưa được đo riêng.</p>''')}
<h3>Đường mang trực tiếp có đủ giải thích tầm với không?</h3>
<p>Câu chuyện quen thuộc là “cổng quên gần 1 nên LSTM nhớ lâu”. Ở đây cổng trung bình chỉ quanh 0.5 (bảng dưới). Một lập luận đếm cho biết
đường trực tiếp có thể đóng góp đến đâu. Với một cặp (đơn vị, chuỗi), muốn đường trực tiếp giữ lại ít nhất {m(r"e^{\lambda}")} mỗi bước trên
{m("L = T - 1")} bước thì, vì {m(r"\ln f_k \le 0")} luôn đúng và {m(r"\ln f_k \le \ln 0.9")} khi {m(r"f_k \le 0.9")}:</p>
{mb(r"\prod_{k=1}^{L} f_k \ge e^{\lambda L} \;\Longrightarrow\; \#\{k : f_k \le 0.9\} \le \frac{\lambda}{\ln 0.9}\, L \;\Longrightarrow\; \#\{k : f_k > 0.9\} \ge \left(1 - \frac{\lambda}{\ln 0.9}\right) L")}
<p>Nếu tỉ lệ bộ ba (đơn vị, bước, chuỗi) có cổng lớn hơn 0.9 là {m("F")} trên chuỗi dài {m("T")}, thì tỉ lệ {m("q")} các cặp (đơn vị, chuỗi)
làm được điều đó bị chặn bởi</p>
{mb(r"q \;\le\; \frac{F\,T}{\left(1 - \lambda/\ln 0.9\right)(T - 1)}")}
{table(["Bài toán", "Cell", "cổng TB", "λ đo được", "cần cổng &gt; 0.9 ở ít nhất", "F quan sát", "chặn trên q"], bound_rows, numeric={2, 3, 4, 5, 6})}
{callout("evidence", '<span class="badge">Chặn trên chứng minh được + một suy luận</span> Ở sentiment, đường trực tiếp không mang được phần lớn độ nhạy', f'''
<p>Ở sentiment, nhiều nhất {percent(bound[('sentiment', 'lstm')]['share'])} (LSTM) và {percent(bound[('sentiment', 'gru')]['share'])} (GRU)
các cặp (đơn vị, câu) có đường trực tiếp giữ được tốc độ đo được. Chặn này là toán học, đúng với mọi phân bố cổng có cùng {m("F")}.
Suy luận tiếp theo cần một giả định: trừ khi gần như toàn bộ độ nhạy ở độ trễ lớn dồn vào số ít cặp đó, phần lớn phải đi theo các đường qua
ma trận hồi quy và các cổng. Đo riêng đóng góp của từng đường để kiểm chứng giả định này là chưa làm.
Ở sinh văn bản, chặn trên là {percent(bound[('generation', 'lstm')]['share'])} (LSTM) và
{"không ràng buộc" if bound[('generation', 'gru')]['share'] >= 1 else percent(bound[('generation', 'gru')]['share'])} (GRU): quá lỏng để kết luận gì.</p>
<p>Một nguyên nhân khả dĩ cho cổng quanh 0.5: thí nghiệm không dùng mẹo khởi tạo bias cổng quên bằng 1 (Gers và cộng sự, 2000; Jozefowicz và cộng sự, 2015),
nên sau {sc['max_steps']:,}–{gc['max_steps']:,} bước cổng vẫn gần {m(r"\sigma(0) = 0.5")}. Đây là biến nhiễu cần kiểm soát ở thí nghiệm sau.</p>''')}
<h3>Tầm với có giải thích được kết quả không?</h3>
{table(["Bài toán", "Thứ tự tầm với (đọc thẳng từ dữ liệu)", "Thứ tự kết quả", "Khớp?"], [
    ["Sentiment", "g(16)/g(0): " + " &gt; ".join(f"{LABEL[c]} {sr[c]['retention']['16']:.3f}" for c in reach_order("sentiment", 16)), "GRU ≈ LSTM &gt; RNN", "Một phần: cell có cổng với xa hơn và thắng; nhưng GRU ≈ LSTM dù tầm với khác nhau"],
    ["Sinh văn bản", "g(64)/g(0): " + " &gt; ".join(f"{LABEL[c]} {sci(gr[c]['retention']['64'])}" for c in reach_order("generation", 64)), "RNN &lt; GRU &lt; LSTM (BPC)", "Không: LSTM với xa nhất nhưng tệ nhất"],
])}
<p>Ở sinh văn bản, sau 64 ký tự độ nhạy còn không quá {percent(longest)} ở mọi cell, trong khi ngữ cảnh là {gc['block_size']}. Ở ngân sách này, việc LSTM chỉ có
{gm_match['lstm']['hidden']} đơn vị so với {gm_match['rnn']['hidden']} của RNN (mục 2) là giải thích phù hợp hơn tầm với. Đây vẫn là giả thuyết:
thí nghiệm khớp số đơn vị (mục 9) mới có thể bác bỏ hoặc củng cố nó.</p>
"""))

    # 6. Landscape --------------------------------------------------------
    def surface_rows(task: str) -> list[list[str]]:
        rows = []
        for cell in CELLS:
            entry = surfaces[task][cell]
            alpha, beta = entry["argmin"]
            rows.append([
                LABEL[cell],
                f(entry["centre"]),
                f(entry["minimum"]),
                "có" if entry["centre_is_minimum"] else f"không — min tại (α, β) = ({s(alpha, 3)}, {s(beta, 3)})",
                f(entry["maximum"]),
                f(entry["rise"], 3),
            ])
        return rows

    flattest = {task: min(CELLS, key=lambda c: surfaces[task][c]["rise"]) for task in surfaces}
    worst = {"sentiment": min(CELLS, key=lambda c: sm[c]["mean"]), "generation": max(CELLS, key=lambda c: gm[c]["mean"])}
    # The callout below says that, for the displayed direction pair, the flattest surface is the worst model.
    assert flattest == worst, (flattest, worst)
    landscape = data["landscapes"]
    grid_size = len(landscape["sentiment"]["grid"])
    landscape_sentences = min(landscape["sentiment"]["batches"] * landscape["sentiment"]["batch_size"], sc["test_examples"])
    landscape_windows = landscape["generation"]["batches"] * landscape["generation"]["batch_size"]
    assert landscape["sentiment"]["batches"] == landscape["generation"]["batches"]
    direction_seeds = [landscape["sentiment"]["seed"], *landscape["sentiment"]["extra_seeds"]]
    assert direction_seeds == [landscape["generation"]["seed"], *landscape["generation"]["extra_seeds"]]
    assert len(direction_seeds) > 1, "run loss_landscape.py with --extra-seeds"

    def draws(task: str, cell: str) -> list[dict]:
        entry = landscape[task]["cells"][cell]
        return [entry, *entry["extra_seeds"]]

    def rise(task: str, cell: str, index: int) -> float:
        draw = draws(task, cell)[index]
        return draw["maximum"] - draw["centre"]

    flattest_by_seed = {
        task: [min(CELLS, key=lambda cell: rise(task, cell, index)) for index in range(len(direction_seeds))]
        for task in ("sentiment", "generation")
    }
    rise_orders = {
        task: {tuple(sorted(CELLS, key=lambda cell: rise(task, cell, index))) for index in range(len(direction_seeds))}
        for task in ("sentiment", "generation")
    }
    # The prose says: on sentiment the flattest model changes with the direction
    # pair; on generation the whole ordering is the same for every pair and runs
    # opposite to BPC (best model sharpest).
    assert len(set(flattest_by_seed["sentiment"])) > 1, flattest_by_seed
    assert len(rise_orders["generation"]) == 1, rise_orders
    generation_order = next(iter(rise_orders["generation"]))
    assert list(generation_order) == sorted(CELLS, key=lambda c: -gm[c]["mean"]), generation_order

    def flattest_count(task: str) -> str:
        winners = flattest_by_seed[task]
        return ", ".join(f"{LABEL[cell]} {winners.count(cell)} lần" for cell in CELLS if winners.count(cell))

    def seed_rows(task: str) -> list[list[str]]:
        rows = []
        for cell in CELLS:
            minimum_at_centre = sum(draw["minimum"] == draw["centre"] for draw in draws(task, cell))
            rows.append(
                [LABEL[cell]]
                + [
                    f"<b>{f(rise(task, cell, index), 2)}</b>" if flattest_by_seed[task][index] == cell else f(rise(task, cell, index), 2)
                    for index in range(len(direction_seeds))
                ]
                + [f"{minimum_at_centre}/{len(direction_seeds)}"]
            )
        return rows

    seed_headers = ["Cell"] + [f"seed {seed}" for seed in direction_seeds] + ["θ₀ là min lưới"]
    parts.append(section("mat-loss", "6", "Mặt loss 3D quanh nghiệm", f"""
<p>Mỗi mặt là lát cắt hai chiều quanh checkpoint {m(r"\theta_0")} (seed 0) theo hai hướng ngẫu nhiên (Li và cộng sự, 2018):</p>
{mb(r"\theta(\alpha, \beta) = \theta_0 + \alpha\,\delta_1 + \beta\,\delta_2,\qquad \delta_{1,j} \leftarrow \frac{\delta_{1,j}}{\lVert \delta_{1,j} \rVert}\,\lVert \theta_{0,j} \rVert \;\;\text{cho từng hàng } j")}
<p>{m(r"\delta_1")} được chuẩn hoá theo từng hàng: mỗi hàng của mỗi ma trận dài bằng hàng tương ứng của {m(r"\theta_0")} (với vectơ bias: cả vectơ).
{m(r"\delta_2")} được tạo cùng cách, rồi trực giao hoá với {m(r"\delta_1")} trong từng tensor và co giãn lại về chuẩn ban đầu của tensor đó.
Lưới {grid_size}×{grid_size} với {m(r"\alpha, \beta \in [-1, 1]")}. Loss là cross-entropy (nat) trên {landscape['sentiment']['batches']} batch đầu, không xáo trộn,
của <b>tập đánh giá</b>: {landscape_sentences} câu SST-2 dev; {landscape_windows} cửa sổ × {gc['block_size']} ký tự validation (nat/ký tự, chia cho ln 2 ra BPC).
Tập con nhỏ nên giá trị tại {m(r"\theta_0")} khác số ở mục 3–4. Ở {m(r"\alpha = 1")}, mỗi hàng bị dịch một vectơ dài bằng chính nó:
đây là nhiễu lớn, không phải lân cận vô cùng bé.</p>
<div class="figure" data-chart="surface">
  <h4>Mặt loss tương tác</h4>
  <div class="tabs" role="tablist">
    <button type="button" role="tab" data-task="sentiment">Phân loại cảm xúc</button>
    <button type="button" role="tab" data-task="generation">Sinh văn bản</button>
  </div>
  <div class="surface-target"></div>
  <p class="caption">Chấm cam: checkpoint θ₀. Vòng tròn “min”: điểm thấp nhất của lưới khi nó khác θ₀. Cùng thang màu và chiều cao cho ba mô hình khi ô “cùng thang” được chọn.</p>
</div>
<div class="figure-grid">
{figure("Lát cắt 1D qua θ₀ (β = 0): sentiment", "cut", "sentiment")}
{figure("Lát cắt 1D qua θ₀ (β = 0): sinh văn bản", "cut", "generation")}
</div>
<h4>Sentiment (cặp hướng seed {direction_seeds[0]})</h4>
{table(["Cell", "loss tại θ₀", "min lưới", "θ₀ là min?", "max lưới", "độ dâng (max − θ₀)"], surface_rows("sentiment"), numeric={1, 2, 4, 5})}
<h4>Sinh văn bản (cặp hướng seed {direction_seeds[0]})</h4>
{table(["Cell", "loss tại θ₀", "min lưới", "θ₀ là min?", "max lưới", "độ dâng (max − θ₀)"], surface_rows("generation"), numeric={1, 2, 4, 5})}
<h3>Đổi cặp hướng thì sao?</h3>
<p>Cùng checkpoint, cùng batch, chỉ đổi seed của hai hướng ngẫu nhiên. Mỗi ô là độ dâng (max − θ₀) trên lưới; số in đậm là mặt phẳng nhất của cột đó.</p>
<h4>Sentiment</h4>
{table(seed_headers, seed_rows("sentiment"), numeric=set(range(1, len(seed_headers))))}
<h4>Sinh văn bản</h4>
{table(seed_headers, seed_rows("generation"), numeric=set(range(1, len(seed_headers))))}
{callout("evidence", '<span class="badge">Đọc mặt loss</span> Độ phẳng không giải thích được thứ tự', f'''
<p>Với cặp hướng seed {direction_seeds[0]} (mặt hiển thị ở trên), mặt phẳng nhất thuộc về mô hình tệ nhất ở cả hai bài toán:
{LABEL[flattest['sentiment']]} ở sentiment, {LABEL[flattest['generation']]} ở sinh văn bản. Đổi cặp hướng thì hai bài toán khác nhau:</p>
<ul>
<li><b>Sentiment:</b> mặt phẳng nhất đổi theo cặp hướng ({flattest_count('sentiment')} trên {len(direction_seeds)} cặp). Không có thứ tự độ phẳng
ổn định để đem so với accuracy.</li>
<li><b>Sinh văn bản:</b> thứ tự {" &lt; ".join(LABEL[c] for c in generation_order)} (từ phẳng đến nhọn) giữ nguyên ở cả {len(direction_seeds)} cặp.
Đây là tính chất ổn định của ba nghiệm này theo cách chuẩn hoá này, và nó ngược chiều với BPC: mô hình tốt nhất có mặt nhọn nhất.</li>
</ul>
<p>Kể cả ở sinh văn bản, điều đó không cho phép kết luận về độ phẳng và tổng quát hoá: ba mô hình không đủ để xác lập một tương quan;
độ dâng trên một hộp ±1 không phải là độ cong; và độ nhọn đổi được bằng phép đổi tham số mà không đổi hàm số (Dinh và cộng sự, 2017).
Ba kiến trúc có ba cách tham số hoá khác nhau, nên so độ nhọn giữa chúng không có ý nghĩa bất biến.</p>''')}
{callout("open", '<span class="badge">Vì sao 3D không thể là chứng minh tổng quát</span>', f'''
<p>Một mặt loss cố định <em>một</em> nghiệm {m(r"\theta_0")}, <em>một</em> mẫu dữ liệu và <em>hai</em> hướng ngẫu nhiên trong không gian
{gc['target_parameters'] / 1e6:.1f}–{sc['target_parameters'] / 1e6:.1f} triệu chiều. Một mệnh đề tổng quát có dạng “với mọi phân phối trong lớp {m(r"\mathcal{D}")}, với mọi ngân sách {m("B")}, …”.
Không tập hữu hạn lát cắt nào bao phủ được lượng từ “với mọi” đó. Mặt loss là công cụ chẩn đoán: nó cho biết nghiệm nằm trong vùng trũng hay không,
cong nhiều hay ít theo vài hướng — không cho biết kiến trúc nào tổng quát hoá tốt hơn.</p>''')}
"""))

    # 7. Context sweep ----------------------------------------------------
    sweep = data["context"]
    characters = sweep["evaluated_characters"]
    context_rows = [
        [f"{context}{' (train)' if context == sweep['trained_context'] else ''}", f"{characters[str(context)]:,}"]
        + [f(sweep["results"][cell][str(context)]) for cell in CELLS]
        for context in sweep["contexts"]
    ]
    stable = all(
        sweep["results"]["rnn"][str(c)] < sweep["results"]["gru"][str(c)] < sweep["results"]["lstm"][str(c)]
        for c in sweep["contexts"]
    )
    window_cap = sweep["batches"] * sweep["batch_size"]
    whole = [c for c in sweep["contexts"] if characters[str(c)] < window_cap * c]
    # The prose cites the trained length as every cell's lowest-BPC column.
    assert all(
        min(sweep["contexts"], key=lambda c: sweep["results"][cell][str(c)]) == sweep["trained_context"] for cell in CELLS
    ), sweep["results"]
    parts.append(section("ngu-canh", "7", "Quét độ dài ngữ cảnh", f"""
<p>Cùng checkpoint sinh văn bản seed 0 (huấn luyện ở {sweep['trained_context']} ký tự), đánh giá trên cửa sổ dài {", ".join(map(str, sweep["contexts"]))} ký tự.
Mỗi độ dài dùng tối đa {sweep['batches']} × {sweep['batch_size']} = {window_cap} cửa sổ không chồng lấn đầu tiên của phần validation
{f"(độ dài {', '.join(map(str, whole))} dùng toàn bộ phần validation)" if whole else ""}; BPC tính trên từng ký tự.</p>
<div class="figure-grid">
{figure("BPC theo độ dài ngữ cảnh khi đánh giá", "context", None)}
<div>{table(["Ngữ cảnh", "Số ký tự", "RNN", "GRU", "LSTM"], context_rows, numeric={1, 2, 3, 4})}
<p class="caption">{"Thứ tự RNN &lt; GRU &lt; LSTM giữ nguyên ở mọi độ dài." if stable else "Thứ tự thay đổi theo độ dài."}
Độ dài {max(sweep["contexts"])} là ngoại suy (dài hơn lúc huấn luyện).</p></div>
</div>
<p>Trong một hàng, ba mô hình được đánh giá trên đúng cùng các ký tự, nên so sánh giữa các cell là công bằng. Giữa các hàng, cả độ dài lẫn đoạn văn bản
đều thay đổi ({characters[str(min(sweep["contexts"]))]:,} → {characters[str(max(sweep["contexts"]))]:,} ký tự), nên xu hướng theo cột
(ví dụ BPC thấp nhất ở {sweep['trained_context']}) lẫn hai yếu tố này. Đây là bằng chứng mạnh hơn một điểm đơn lẻ, nhưng vẫn là một corpus,
một ngân sách, một độ dài huấn luyện, một checkpoint. “Ngữ cảnh” ở đây là độ dài khi <em>đánh giá</em>; muốn biết cell nào hưởng lợi từ
ngữ cảnh dài cần <em>huấn luyện</em> ở từng độ dài.</p>
"""))

    # 8. Claims -----------------------------------------------------------
    parts.append(section("ket-luan", "8", "Điều đã chứng minh, điều có bằng chứng, điều chưa biết", f"""
<div class="claims">
{callout("theorem", '<span class="badge">Định lý</span> Đúng không cần thí nghiệm', f'''
<ul>
<li>Số tham số {m(r"P = m(IH + H^{2} + 2H) + \dots")} và hệ quả: cùng {m("P")} thì {m(r"H_{\text{GRU}}/H_{\text{RNN}} \to 1/\sqrt{3}")}, {m(r"H_{\text{LSTM}}/H_{\text{RNN}} \to 1/2")}.</li>
<li>Với RNN, {m(r"\lVert \partial h_T/\partial h_t \rVert \le \sigma_{\max}(U)^{T-t}")}; gradient chắc chắn tắt nếu {m(r"\sigma_{\max}(U) < 1")}.</li>
<li>Đường mang của LSTM/GRU nhân gradient với {m(r"\prod f_k")} hoặc {m(r"\prod z_k")}.</li>
<li>Đếm: để đường đó giữ ít nhất {m(r"e^{\lambda}")} mỗi bước, cổng phải lớn hơn 0.9 ở ít nhất một tỉ lệ {m(r"1 - \lambda/\ln 0.9")} của số bước.</li>
<li>No free lunch: không có thứ tự nào đúng trên mọi phân phối dữ liệu.</li>
</ul>''')}
{callout("evidence", '<span class="badge">Bằng chứng có điều kiện</span> Đúng với dữ liệu, ngân sách, seed này', f'''
<ul>
<li>SST-2, {sc['target_parameters']:,} tham số: GRU và LSTM hơn RNN khoảng {f(-100 * sp[('rnn', 'gru')]['estimate'], 1)} điểm (vượt cả kiểm định theo seed lẫn McNemar sau Holm).</li>
<li>SST-2: GRU và LSTM không phân biệt được.</li>
<li>Shakespeare, {gc['target_parameters']:,} tham số, ngữ cảnh {gc['block_size']}: LSTM kém hơn RNN và GRU ở mọi seed; thứ tự giữ nguyên khi đánh giá ở 32–256 ký tự.</li>
<li>Trên checkpoint đã huấn luyện, {m(r"\sigma_{\max}(U) > 1")} nhưng độ nhạy của RNN vẫn giảm {f(100 * min(1 - sr['rnn']['per_step_retention'], 1 - gr['rnn']['per_step_retention']), 1)}–{f(100 * max(1 - sr['rnn']['per_step_retention'], 1 - gr['rnn']['per_step_retention']), 1)} % mỗi bước: điều kiện đủ của định lý không thỏa mà hệ quả vẫn xảy ra.</li>
<li>SST-2: cell có cổng giữ độ nhạy xa hơn RNN, nhưng đường mang trực tiếp chỉ giữ được tốc độ đó ở nhiều nhất {percent(bound[('sentiment', 'lstm')]['share'])} (LSTM)
và {percent(bound[('sentiment', 'gru')]['share'])} (GRU) các cặp (đơn vị, câu).</li>
</ul>''')}
{callout("open", '<span class="badge">Chưa chứng minh</span> Không suy ra được từ dữ liệu này', '''
<ul>
<li>Thứ tự phổ quát giữa RNN, GRU, LSTM.</li>
<li>RNN tốt hơn GRU ở sinh văn bản (CI theo seed chứa 0).</li>
<li>Thứ tự ở ngữ cảnh huấn luyện dài hơn, ngân sách lớn hơn, corpus khác, hoặc khi mỗi cell được tinh chỉnh siêu tham số riêng.</li>
<li>Đường nào thực sự mang gradient trong LSTM/GRU (mới có chặn trên cho đường trực tiếp).</li>
<li>Độ phẳng của mặt loss quyết định khả năng tổng quát hoá.</li>
<li>Tầm với gradient có khoảng tin cậy (mới đo trên một seed).</li>
</ul>''')}
</div>
<h3>Một chứng minh tổng quát cần gì</h3>
<p>Một trong hai dạng sau, kèm giả định được nói rõ:</p>
<ol>
<li><b>Chặn trên và chặn dưới trên một lớp bài toán.</b> Ví dụ: “với mọi chuỗi có phụ thuộc ở khoảng cách {m("L")} trong lớp {m(r"\mathcal{D}_L")}, RNN với
{m("H")} đơn vị cần ít nhất {m(r"\Omega(f(H, L))")} mẫu, còn LSTM với cùng {m("P")} cần {m(r"O(g(H, L))")}”. Đó là mệnh đề có lượng từ “với mọi” và một lớp bài toán xác định.</li>
<li><b>Phản ví dụ.</b> Một phân phối duy nhất mà thứ tự đảo ngược đủ để bác bỏ “A luôn tốt hơn B”. Chính bộ dữ liệu ở đây đã làm vậy:
LSTM thắng RNN ở sentiment nhưng thua ở sinh văn bản, nên mệnh đề “với cùng số tham số và cùng công thức huấn luyện, LSTM luôn tốt hơn RNN” là sai.
Phản ví dụ này dựa trên kết quả thống kê (CI theo seed không chứa 0), và không bác bỏ được phiên bản cho phép tinh chỉnh siêu tham số riêng
cho từng cell, vì ở đây không tinh chỉnh.</li>
</ol>
"""))

    # 9. Next steps -------------------------------------------------------
    parts.append(section("tiep-theo", "9", "Thí nghiệm tiếp theo", """
<ol>
<li><b>Khớp số đơn vị bên cạnh khớp tham số.</b> Cùng H cho ba cell, để tách tác dụng của cổng khỏi tác dụng của số đơn vị.</li>
<li><b>Bias cổng quên bằng 1</b> cho LSTM (và bias cổng update tương ứng cho GRU), rồi đo lại cổng và tầm với.</li>
<li><b>Huấn luyện ở từng độ dài ngữ cảnh</b> 32 / 128 / 512, không chỉ đánh giá; thêm bài toán tổng hợp có phụ thuộc xa biết trước
(copy, adding), nơi có thể phát biểu và kiểm tra một mệnh đề định lượng.</li>
<li><b>5 seed và lưu checkpoint mọi seed</b> để tầm với và mặt loss có khoảng tin cậy.</li>
<li><b>Các trị riêng lớn nhất của Hessian</b> (power iteration trên tích Hessian–vectơ): một đại lượng độ cong không phụ thuộc hướng ngẫu nhiên,
thay cho độ dâng trên vài cặp hướng.</li>
</ol>
"""))

    # 10. Reproduce -------------------------------------------------------
    parts.append(section("tai-lap", "10", "Tái lập", f"""
<pre><code>python data/prepare_sst2.py
python experiments/sequence_classification/sentiment/run.py --tier development --device cuda
python experiments/language_modeling/recurrent_generation/run.py --tier development --device cuda

# phân tích sau huấn luyện (đọc checkpoint, không huấn luyện lại)
python analysis/gradient_reach.py --task sentiment  --run-directory runs/sequence_classification/sentiment/development --output-directory results/recurrent_comparison/reach
python analysis/gradient_reach.py --task generation --run-directory runs/language_modeling/recurrent_generation/development --output-directory results/recurrent_comparison/reach
# hướng ngẫu nhiên phụ thuộc bộ sinh số của thiết bị: mặt ở đây được tính trên CUDA
python analysis/loss_landscape.py --task sentiment  --run-directory runs/sequence_classification/sentiment/development --output-directory results/recurrent_comparison/landscapes --device cuda --extra-seeds 1 2 3 4
python analysis/loss_landscape.py --task generation --run-directory runs/language_modeling/recurrent_generation/development --output-directory results/recurrent_comparison/landscapes --device cuda --extra-seeds 1 2 3 4
python analysis/context_sweep.py --run-directory runs/language_modeling/recurrent_generation/development --output-directory results/recurrent_comparison/contexts

# báo cáo này (Python 3.12+)
python -m pip install -e ".[analysis]"
python analysis/build_report.py</code></pre>
<p>Các tier <code>tiny</code> và <code>smoke</code> của hai thí nghiệm chạy trên CPU trong test suite. Mã nguồn:
<a href="{REPOSITORY}">{REPOSITORY.removeprefix("https://")}</a>.</p>
"""))

    parts.append("""
<footer>
<h3>Tài liệu tham khảo</h3>
<ol>
<li>Hochreiter, S. &amp; Schmidhuber, J. (1997). Long Short-Term Memory. <i>Neural Computation</i>.</li>
<li>Gers, F., Schmidhuber, J. &amp; Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. <i>Neural Computation</i>.</li>
<li>Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. <i>EMNLP</i>.</li>
<li>Pascanu, R., Mikolov, T. &amp; Bengio, Y. (2013). On the difficulty of training recurrent neural networks. <i>ICML</i>.</li>
<li>Jozefowicz, R., Zaremba, W. &amp; Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. <i>ICML</i>.</li>
<li>Li, H. et al. (2018). Visualizing the Loss Landscape of Neural Nets. <i>NeurIPS</i>.</li>
<li>Dinh, L. et al. (2017). Sharp Minima Can Generalize For Deep Nets. <i>ICML</i>.</li>
<li>Wolpert, D. &amp; Macready, W. (1997). No Free Lunch Theorems for Optimization. <i>IEEE Transactions on Evolutionary Computation</i>.</li>
<li>Dietterich, T. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. <i>Neural Computation</i>.</li>
<li>Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. <i>Scandinavian Journal of Statistics</i>.</li>
<li>Efron, B. &amp; Tibshirani, R. (1993). <i>An Introduction to the Bootstrap</i>.</li>
<li>Socher, R. et al. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. <i>EMNLP</i>; Wang, A. et al. (2018). GLUE. <i>EMNLP Workshop</i>.</li>
</ol>
<p>Báo cáo được tạo tự động bởi <code>analysis/build_report.py</code> từ các file trong <code>results/recurrent_comparison/</code>.</p>
</footer>""")
    return "\n".join(parts)


TOC = [
    ("tom-tat", "Tóm tắt"),
    ("cau-hoi", "1 · “Tốt hơn” nghĩa là gì"),
    ("thiet-ke", "2 · Thiết kế"),
    ("cam-xuc", "3 · Phân loại cảm xúc"),
    ("sinh-van-ban", "4 · Sinh văn bản"),
    ("co-che", "5 · Cơ chế gradient"),
    ("mat-loss", "6 · Mặt loss 3D"),
    ("ngu-canh", "7 · Quét ngữ cảnh"),
    ("ket-luan", "8 · Kết luận"),
    ("tiep-theo", "9 · Bước tiếp theo"),
    ("tai-lap", "10 · Tái lập"),
]


def main() -> None:
    data = collect()
    derived = derive(data)
    body = build_body(data, derived)
    payload = json.dumps(chart_data(data, derived), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    toc = "".join(f'<li><a href="#{anchor}">{title}</a></li>' for anchor, title in TOC)
    document = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RNN, GRU và LSTM ở cùng ngân sách tham số</title>
<style>{(ASSETS / "report.css").read_text(encoding="utf-8")}</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Mục lục"><h2>Mục lục</h2><ol>{toc}</ol></nav>
<main>
{body}
</main>
</div>
<script id="report-data" type="application/json">{payload}</script>
<script>{(ASSETS / "report.js").read_text(encoding="utf-8")}</script>
</body>
</html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
