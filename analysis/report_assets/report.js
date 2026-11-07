// Charts and the 3-D surface viewer for the recurrent comparison report.
// Everything is drawn from the JSON embedded in #report-data; no network access.
(() => {
  "use strict";

  const DATA = JSON.parse(document.getElementById("report-data").textContent);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const CELLS = ["rnn", "gru", "lstm"];
  const LABEL = { rnn: "RNN", gru: "GRU", lstm: "LSTM" };
  const COLOR = { rnn: "var(--rnn)", gru: "var(--gru)", lstm: "var(--lstm)" };

  // ---------------------------------------------------------------- utilities
  const minus = (text) => text.replace("-", "\u2212");
  const fmt = (value, digits = 4) => (Number.isFinite(value) ? minus(value.toFixed(digits)) : "—");
  const signed = (value, digits = 4) => (Number.isFinite(value) ? (value >= 0 ? "+" : "") + fmt(value, digits) : "—");
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  document.body.appendChild(tooltip);

  function showTip(html, event) {
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
    const pad = 14;
    const box = tooltip.getBoundingClientRect();
    let left = event.clientX + pad;
    let top = event.clientY + pad;
    if (left + box.width > window.innerWidth - 8) left = event.clientX - box.width - pad;
    if (top + box.height > window.innerHeight - 8) top = event.clientY - box.height - pad;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }
  const hideTip = () => { tooltip.style.display = "none"; };

  function svg(tag, attributes = {}, parent = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attributes)) {
      if (key === "style") Object.assign(node.style, value);
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function linear(d0, d1, r0, r1) {
    const span = d1 - d0 || 1;
    return (value) => r0 + ((value - d0) / span) * (r1 - r0);
  }
  function logarithmic(d0, d1, r0, r1) {
    const l0 = Math.log10(d0);
    const span = Math.log10(d1) - l0 || 1;
    return (value) => r0 + ((Math.log10(value) - l0) / span) * (r1 - r0);
  }
  function niceStep(span, count) {
    const raw = span / Math.max(count, 1);
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const unit = raw / power;
    return (unit < 1.5 ? 1 : unit < 3 ? 2 : unit < 7 ? 5 : 10) * power;
  }
  function linearTicks(min, max, count = 5) {
    // Integer multiples of the step, so 0 is exactly 0 rather than 7e-18.
    const step = niceStep(max - min || 1, count);
    const ticks = [];
    for (let k = Math.ceil(min / step - 1e-9); k * step <= max + step * 1e-9; k += 1) {
      ticks.push(k === 0 ? 0 : Number((k * step).toPrecision(12)));
    }
    return ticks;
  }
  function paddedDomain(values, fraction = 0.08) {
    const finite = values.filter(Number.isFinite);
    let low = Math.min(...finite);
    let high = Math.max(...finite);
    if (low === high) { const pad = Math.abs(low) * 0.05 || 1; return [low - pad, high + pad]; }
    const pad = (high - low) * fraction;
    return [low - pad, high + pad];
  }
  const tickText = (value) => {
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    if (magnitude >= 1e5 || magnitude < 1e-3) return minus(value.toExponential(0));
    return minus(String(Number(value.toPrecision(4))));
  };

  // Charts are laid out at the container's real width so text keeps its CSS
  // size; the render loop redraws them when that width changes.
  function frame(container, { width = 720, height = 330, margin = {} } = {}) {
    const m = { top: 14, right: 18, bottom: 46, left: 64, ...margin };
    const measured = Math.round(container.getBoundingClientRect().width);
    const w = measured > 0 ? Math.max(300, measured) : width;
    const h = Math.round(Math.min(height, Math.max(240, w * 0.78)));
    const root = svg("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, class: "chart", role: "img" });
    container.appendChild(root);
    return { root, width: w, height: h, x0: m.left, x1: w - m.right, y0: h - m.bottom, y1: m.top };
  }

  function axes(f, x, y, { xTicks, yTicks, xLabel, yLabel, xText = tickText, yLog = false }) {
    const group = svg("g", {}, f.root);
    for (const tick of yTicks) {
      const py = y(tick);
      svg("line", { x1: f.x0, x2: f.x1, y1: py, y2: py, class: "grid" }, group);
      const label = svg("text", { x: f.x0 - 8, y: py + 4, class: "tick", "text-anchor": "end" }, group);
      if (yLog) {
        label.textContent = "10";
        svg("tspan", { dy: "-0.55em", "font-size": "0.75em", text: String(Math.round(Math.log10(tick))) }, label);
      } else {
        label.textContent = tickText(tick);
      }
    }
    for (const tick of xTicks) {
      const px = x(tick);
      svg("line", { x1: px, x2: px, y1: f.y0, y2: f.y0 + 5, class: "axis" }, group);
      svg("text", { x: px, y: f.y0 + 18, class: "tick", "text-anchor": "middle", text: xText(tick) }, group);
    }
    svg("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "axis" }, group);
    svg("line", { x1: f.x0, x2: f.x0, y1: f.y0, y2: f.y1, class: "axis" }, group);
    if (xLabel) svg("text", { x: (f.x0 + f.x1) / 2, y: f.height - 8, class: "axis-label", "text-anchor": "middle", text: xLabel }, group);
    if (yLabel) {
      const cy = (f.y0 + f.y1) / 2;
      svg("text", { x: 14, y: cy, class: "axis-label", "text-anchor": "middle", transform: `rotate(-90 14 ${cy})`, text: yLabel }, group);
    }
  }

  function legend(container, items) {
    const box = document.createElement("div");
    box.className = "legend";
    for (const item of items) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "legend-item";
      const dashed = item.dashed ? " dashed" : "";
      button.innerHTML = `<span class="swatch${dashed}" style="background:${item.color}"></span>${item.label}`;
      button.addEventListener("click", () => {
        const off = button.classList.toggle("off");
        container.querySelectorAll(`[data-key="${item.key}"]`).forEach((node) => node.classList.toggle("hidden", off));
      });
      box.appendChild(button);
    }
    container.appendChild(box);
  }
  const isHidden = (container, key) => {
    const node = container.querySelector(`g[data-key="${key}"]`);
    return node ? node.classList.contains("hidden") : false;
  };
  const localPoint = (root, event) => {
    const point = root.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(root.getScreenCTM().inverse());
  };

  // ------------------------------------------------------------- line chart
  // series: {key, label, x[], y[], lower?[], upper?[], color?, dashed?, markers?}
  function lineChart(container, spec) {
    const { series, xLabel, yLabel, logY = false, categories = null, digits = 4, height = 330, xName = xLabel } = spec;
    container.replaceChildren();
    series.forEach((s) => { s.color = s.color || COLOR[s.key] || COLOR[s.cell]; });
    legend(container, series.filter((s) => s.legend !== false));
    const f = frame(container, { height });

    const xs = series.flatMap((s) => s.x);
    const ys = series
      .flatMap((s) => [...s.y, ...(s.lower || []), ...(s.upper || [])])
      .filter((v) => Number.isFinite(v) && (!logY || v > 0));
    const [xMin, xMax] = categories ? [-0.35, categories.length - 0.65] : [Math.min(...xs), Math.max(...xs)];
    const x = linear(xMin, xMax, f.x0, f.x1);
    let y;
    let yTicks;
    if (logY) {
      const d0 = Math.pow(10, Math.floor(Math.log10(Math.min(...ys))));
      const d1 = Math.pow(10, Math.ceil(Math.log10(Math.max(...ys))));
      y = logarithmic(d0, d1, f.y0, f.y1);
      yTicks = [];
      for (let e = Math.log10(d0); e <= Math.log10(d1) + 1e-9; e += 1) yTicks.push(Math.pow(10, e));
      if (yTicks.length > 9) yTicks = yTicks.filter((_, i) => i % 2 === 0);
    } else {
      const [d0, d1] = paddedDomain(ys);
      y = linear(d0, d1, f.y0, f.y1);
      yTicks = linearTicks(d0, d1, 5).filter((t) => t >= d0 && t <= d1);
    }
    const xTicks = categories ? categories.map((_, i) => i) : linearTicks(xMin, xMax, 6);
    axes(f, x, y, { xTicks, yTicks, xLabel, yLabel, yLog: logY, xText: categories ? (i) => categories[i] : tickText });

    for (const s of series) {
      const group = svg("g", { "data-key": s.key }, f.root);
      if (s.lower && s.upper) {
        const top = s.x.map((v, i) => `${x(v)},${y(s.upper[i])}`);
        const bottom = s.x.map((v, i) => `${x(v)},${y(s.lower[i])}`).reverse();
        svg("polygon", { points: [...top, ...bottom].join(" "), class: "band", style: { fill: s.color } }, group);
      }
      const points = s.x.map((v, i) => [v, s.y[i]]).filter(([, v]) => Number.isFinite(v) && (!logY || v > 0));
      svg("polyline", {
        points: points.map(([a, b]) => `${x(a)},${y(b)}`).join(" "),
        class: s.dashed ? "line dashed" : "line",
        style: { stroke: s.color },
      }, group);
      const markers = s.markers ?? points.length <= 40;
      if (markers) for (const [a, b] of points) svg("circle", { cx: x(a), cy: y(b), r: 3, class: "marker", style: { fill: s.color } }, group);
    }

    const overlay = svg("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, class: "overlay" }, f.root);
    const cross = svg("line", { y1: f.y1, y2: f.y0, class: "crosshair" }, f.root);
    const xValues = [...new Set(xs)].sort((a, b) => a - b);
    overlay.addEventListener("mousemove", (event) => {
      const local = localPoint(f.root, event);
      let best = xValues[0];
      for (const value of xValues) if (Math.abs(x(value) - local.x) < Math.abs(x(best) - local.x)) best = value;
      cross.setAttribute("x1", x(best));
      cross.setAttribute("x2", x(best));
      cross.style.display = "block";
      const rows = series
        .filter((s) => s.tooltip !== false && !isHidden(container, s.key))
        .map((s) => {
          const index = s.x.indexOf(best);
          if (index < 0 || !Number.isFinite(s.y[index])) return "";
          const spread = s.lower ? ` <span class="muted">± ${fmt((s.upper[index] - s.lower[index]) / 2, digits)}</span>` : "";
          const value = logY ? s.y[index].toExponential(3) : fmt(s.y[index], digits);
          return `<div><span class="swatch" style="background:${s.color}"></span>${s.label}: <b>${value}</b>${spread}</div>`;
        })
        .join("");
      const head = categories ? categories[best] : `${xName} = ${tickText(best)}`;
      showTip(`<div class="tip-head">${head}</div>${rows}`, event);
    });
    overlay.addEventListener("mouseleave", () => { cross.style.display = "none"; hideTip(); });
  }

  // -------------------------------------------- mean ± sd whiskers with seeds
  // A dot plot, not bars: the axis does not start at zero, and bars would
  // exaggerate the differences.
  function dotChart(container, spec) {
    const { rows, yLabel, digits = 4, height = 300 } = spec;
    container.replaceChildren();
    const f = frame(container, { width: 560, height });
    const values = rows.flatMap((r) => [r.mean - r.sd, r.mean + r.sd, ...r.seeds]);
    const [d0, d1] = paddedDomain(values, 0.15);
    const y = linear(d0, d1, f.y0, f.y1);
    const band = (f.x1 - f.x0) / rows.length;
    const x = (index) => f.x0 + band * (index + 0.5);
    axes(f, x, y, {
      xTicks: rows.map((_, i) => i),
      yTicks: linearTicks(d0, d1, 5).filter((t) => t >= d0 && t <= d1),
      yLabel,
      xText: (i) => LABEL[rows[i].cell],
    });
    rows.forEach((row, index) => {
      const group = svg("g", {}, f.root);
      const cx = x(index);
      svg("rect", { x: cx - band / 2, y: f.y1, width: band, height: f.y0 - f.y1, class: "hit" }, group);
      svg("line", { x1: cx, x2: cx, y1: y(row.mean - row.sd), y2: y(row.mean + row.sd), class: "whisker" }, group);
      for (const value of [row.mean - row.sd, row.mean + row.sd]) {
        svg("line", { x1: cx - 10, x2: cx + 10, y1: y(value), y2: y(value), class: "whisker" }, group);
      }
      const offset = Math.min(26, band * 0.22);
      const spacing = Math.min(11, band * 0.1);
      row.seeds.forEach((value, seed) => {
        svg("circle", { cx: cx + offset + seed * spacing, cy: y(value), r: 4, class: "seed-dot" }, group);
      });
      svg("circle", { cx, cy: y(row.mean), r: 7, class: "mean-dot", style: { fill: COLOR[row.cell] } }, group);
      group.addEventListener("mousemove", (event) => showTip(
        `<div class="tip-head">${LABEL[row.cell]}</div>trung bình: <b>${fmt(row.mean, digits)}</b><br>` +
        `độ lệch chuẩn (n−1): ${fmt(row.sd, digits)}<br>từng seed: ${row.seeds.map((v) => fmt(v, digits)).join(" · ")}`,
        event,
      ));
      group.addEventListener("mouseleave", hideTip);
    });
  }

  // ------------------------------------------------------------ forest plot
  // rows: {label, estimate, bootstrap:[low, high], seed:[low, high], detail}
  function forestPlot(container, spec) {
    const { rows, xLabel, digits = 4 } = spec;
    container.replaceChildren();
    const rowHeight = 54;
    const f = frame(container, { height: 34 + rows.length * rowHeight + 44, margin: { left: 118, top: 16, right: 24 } });
    const values = rows.flatMap((r) => [...r.bootstrap, ...r.seed, r.estimate, 0]);
    const [d0, d1] = paddedDomain(values, 0.06);
    const x = linear(d0, d1, f.x0, f.x1);
    const ticks = linearTicks(d0, d1, 6).filter((t) => t >= d0 && t <= d1);
    const group = svg("g", {}, f.root);
    for (const tick of ticks) {
      svg("line", { x1: x(tick), x2: x(tick), y1: f.y1, y2: f.y0, class: "grid" }, group);
      svg("text", { x: x(tick), y: f.y0 + 18, class: "tick", "text-anchor": "middle", text: tickText(tick) }, group);
    }
    svg("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "axis" }, group);
    svg("line", { x1: x(0), x2: x(0), y1: f.y1, y2: f.y0, class: "zero" }, group);
    svg("text", { x: (f.x0 + f.x1) / 2, y: f.height - 8, class: "axis-label", "text-anchor": "middle", text: xLabel }, group);
    rows.forEach((row, index) => {
      const cy = f.y1 + 22 + index * rowHeight;
      const g = svg("g", {}, f.root);
      svg("rect", { x: 0, y: cy - rowHeight / 2, width: f.width, height: rowHeight, class: "hit" }, g);
      svg("text", { x: f.x0 - 12, y: cy + 4, class: "row-label", "text-anchor": "end", text: row.label }, g);
      svg("line", { x1: x(row.seed[0]), x2: x(row.seed[1]), y1: cy - 7, y2: cy - 7, class: "interval-seed", style: { stroke: "var(--accent)" } }, g);
      svg("line", { x1: x(row.bootstrap[0]), x2: x(row.bootstrap[1]), y1: cy + 7, y2: cy + 7, class: "interval-bootstrap", style: { stroke: "var(--link)" } }, g);
      svg("circle", { cx: x(row.estimate), cy, r: 5.5, class: "estimate", style: { fill: "var(--fg)" } }, g);
      g.addEventListener("mousemove", (event) => showTip(
        `<div class="tip-head">${row.label}</div>ước lượng: <b>${signed(row.estimate, digits)}</b><br>` +
        `CI t theo seed: [${signed(row.seed[0], digits)}, ${signed(row.seed[1], digits)}]<br>` +
        `CI bootstrap: [${signed(row.bootstrap[0], digits)}, ${signed(row.bootstrap[1], digits)}]` +
        (row.detail ? `<br>${row.detail}` : ""),
        event,
      ));
      g.addEventListener("mouseleave", hideTip);
    });
  }

  // ------------------------------------------------------ 3-D surface viewer
  const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  function viridis(t) {
    const scaled = clamp(t, 0, 1) * (VIRIDIS.length - 1);
    const index = Math.min(VIRIDIS.length - 2, Math.floor(scaled));
    const fraction = scaled - index;
    const a = VIRIDIS[index];
    const b = VIRIDIS[index + 1];
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * fraction)).join(",")})`;
  }

  function surfaceExplorer(root, task) {
    const landscape = DATA.landscapes[task];
    const grid = landscape.grid;
    const n = grid.length;
    const centre = (n - 1) / 2;
    const defaults = { yaw: -0.85, pitch: 0.62, zoom: 1 };
    const camera = { ...defaults };
    let shared = true;
    const everything = CELLS.flatMap((cell) => landscape.cells[cell].surface.flat());
    const sharedRange = [Math.min(...everything), Math.max(...everything)];

    root.replaceChildren();
    const toolbar = document.createElement("div");
    toolbar.className = "surface-toolbar";
    toolbar.innerHTML =
      '<label><input type="checkbox" checked> Cùng thang loss cho ba mô hình</label>' +
      '<button type="button">Đặt lại góc nhìn</button>' +
      '<span class="muted">Kéo để xoay (ba mô hình xoay cùng nhau) · cuộn để phóng to · rê chuột để đọc α, β, loss</span>';
    root.appendChild(toolbar);
    const row = document.createElement("div");
    row.className = "surface-row";
    root.appendChild(row);
    const colorbar = document.createElement("div");
    colorbar.className = "colorbar";
    root.appendChild(colorbar);

    const views = CELLS.map((cell) => {
      const entry = landscape.cells[cell];
      const figure = document.createElement("figure");
      figure.className = "surface";
      figure.innerHTML =
        `<figcaption><span class="swatch" style="background:${COLOR[cell]}"></span><b>${LABEL[cell]}</b>` +
        `<span class="muted">θ₀ = ${fmt(entry.centre)} · min = ${fmt(entry.minimum)} · max = ${fmt(entry.maximum)}</span></figcaption>`;
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-label", `Mặt loss của ${LABEL[cell]}`);
      figure.appendChild(canvas);
      row.appendChild(figure);
      return { cell, canvas, values: entry.surface, range: [entry.minimum, entry.maximum], hover: null, projected: null };
    });

    function updateColorbar() {
      const ranges = shared ? [sharedRange] : views.map((v) => v.range);
      const text = shared
        ? `${fmt(sharedRange[0], 3)} → ${fmt(sharedRange[1], 3)} (chung cho ba mô hình)`
        : views.map((v) => `${LABEL[v.cell]} ${fmt(v.range[0], 3)}→${fmt(v.range[1], 3)}`).join(" · ");
      colorbar.innerHTML = `<span>loss</span><span class="ramp"></span><span>${text}</span>`;
      return ranges;
    }

    function project(x, y, z, width, height) {
      const cy = Math.cos(camera.yaw);
      const sy = Math.sin(camera.yaw);
      const cp = Math.cos(camera.pitch);
      const sp = Math.sin(camera.pitch);
      const rx = x * cy - y * sy;
      const ry = x * sy + y * cy;
      const up = z * cp + ry * sp;
      const depth = ry * cp - z * sp;
      const scale = Math.min(width, height * 1.25) * 0.31 * camera.zoom;
      return [width / 2 + rx * scale, height * 0.55 - up * scale, depth];
    }

    function draw(view) {
      const canvas = view.canvas;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!width || !height) return;
      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
      if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const style = getComputedStyle(document.documentElement);
      const mesh = style.getPropertyValue("--mesh").trim();
      const muted = style.getPropertyValue("--muted").trim();
      const fg = style.getPropertyValue("--fg").trim();
      const accent = style.getPropertyValue("--accent").trim();
      const [low, high] = shared ? sharedRange : view.range;
      const span = high - low || 1;
      const height01 = (value) => (value - low) / span;
      const zOf = (value) => height01(value) - 0.5;

      const points = [];
      for (let i = 0; i < n; i += 1) {
        points.push([]);
        for (let j = 0; j < n; j += 1) points[i].push(project(grid[i], grid[j], zOf(view.values[i][j]), width, height));
      }
      view.projected = points;

      // Base plane and its labels.
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => project(a, b, -0.5, width, height));
      context.lineWidth = 1;
      context.strokeStyle = muted;
      context.globalAlpha = 0.55;
      context.beginPath();
      corners.forEach(([px, py], k) => (k ? context.lineTo(px, py) : context.moveTo(px, py)));
      context.closePath();
      context.stroke();
      for (const t of [-0.5, 0, 0.5]) {
        const a = project(t, -1, -0.5, width, height);
        const b = project(t, 1, -0.5, width, height);
        const c = project(-1, t, -0.5, width, height);
        const d = project(1, t, -0.5, width, height);
        context.beginPath();
        context.moveTo(a[0], a[1]); context.lineTo(b[0], b[1]);
        context.moveTo(c[0], c[1]); context.lineTo(d[0], d[1]);
        context.stroke();
      }
      context.globalAlpha = 1;
      context.fillStyle = muted;
      context.font = "12px system-ui, sans-serif";
      context.textAlign = "center";
      const alphaLabel = project(0, -1.25, -0.5, width, height);
      const betaLabel = project(1.25, 0, -0.5, width, height);
      context.fillText("α", alphaLabel[0], alphaLabel[1]);
      context.fillText("β", betaLabel[0], betaLabel[1]);
      for (const t of [-1, 1]) {
        const p = project(t, -1.12, -0.5, width, height);
        context.fillText(String(t), p[0], p[1]);
        const q = project(1.12, t, -0.5, width, height);
        context.fillText(String(t), q[0], q[1]);
      }

      // Surface, painted back to front.
      const quads = [];
      for (let i = 0; i < n - 1; i += 1) {
        for (let j = 0; j < n - 1; j += 1) {
          const c = [points[i][j], points[i + 1][j], points[i + 1][j + 1], points[i][j + 1]];
          const value = (view.values[i][j] + view.values[i + 1][j] + view.values[i + 1][j + 1] + view.values[i][j + 1]) / 4;
          quads.push({ c, depth: (c[0][2] + c[1][2] + c[2][2] + c[3][2]) / 4, value });
        }
      }
      quads.sort((a, b) => b.depth - a.depth);
      context.lineWidth = 0.6;
      context.strokeStyle = mesh;
      for (const quad of quads) {
        context.beginPath();
        quad.c.forEach(([px, py], k) => (k ? context.lineTo(px, py) : context.moveTo(px, py)));
        context.closePath();
        context.fillStyle = viridis(height01(quad.value));
        context.fill();
        context.stroke();
      }

      // Trained point theta_0 and the grid minimum.
      const ci = centre;
      const pin = points[ci][ci];
      const foot = project(0, 0, -0.5, width, height);
      context.setLineDash([3, 3]);
      context.strokeStyle = fg;
      context.beginPath();
      context.moveTo(foot[0], foot[1]);
      context.lineTo(pin[0], pin[1]);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = accent;
      context.strokeStyle = "#fff";
      context.lineWidth = 1.5;
      context.beginPath();
      context.arc(pin[0], pin[1], 5, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.fillStyle = fg;
      context.textAlign = "left";
      context.fillText("θ₀", pin[0] + 8, pin[1] - 6);

      let minI = 0;
      let minJ = 0;
      for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) if (view.values[i][j] < view.values[minI][minJ]) { minI = i; minJ = j; }
      if (minI !== ci || minJ !== ci) {
        const p = points[minI][minJ];
        context.strokeStyle = fg;
        context.lineWidth = 2;
        context.beginPath();
        context.arc(p[0], p[1], 5.5, 0, Math.PI * 2);
        context.stroke();
        context.fillText("min", p[0] + 8, p[1] + 14);
      }

      if (view.hover) {
        const [i, j] = view.hover;
        const p = points[i][j];
        context.lineWidth = 2.5;
        context.strokeStyle = "#fff";
        context.beginPath();
        context.arc(p[0], p[1], 4.5, 0, Math.PI * 2);
        context.stroke();
        context.lineWidth = 1;
        context.strokeStyle = "#000";
        context.stroke();
      }
    }

    const drawAll = () => { updateColorbar(); views.forEach(draw); };

    for (const view of views) {
      const canvas = view.canvas;
      let drag = null;
      canvas.addEventListener("pointerdown", (event) => {
        drag = { x: event.clientX, y: event.clientY };
        canvas.setPointerCapture(event.pointerId);
        hideTip();
      });
      canvas.addEventListener("pointermove", (event) => {
        if (drag) {
          camera.yaw += (event.clientX - drag.x) * 0.01;
          camera.pitch = clamp(camera.pitch + (event.clientY - drag.y) * 0.008, 0.06, 1.5);
          drag = { x: event.clientX, y: event.clientY };
          views.forEach(draw);
          return;
        }
        if (!view.projected) return;
        const box = canvas.getBoundingClientRect();
        const mx = event.clientX - box.left;
        const my = event.clientY - box.top;
        let best = null;
        let bestDistance = 18 * 18;
        for (let i = 0; i < n; i += 1) {
          for (let j = 0; j < n; j += 1) {
            const [px, py] = view.projected[i][j];
            const distance = (px - mx) ** 2 + (py - my) ** 2;
            if (distance < bestDistance) { bestDistance = distance; best = [i, j]; }
          }
        }
        view.hover = best;
        draw(view);
        if (best) {
          const [i, j] = best;
          const loss = view.values[i][j];
          const delta = loss - view.values[centre][centre];
          showTip(
            `<div class="tip-head">${LABEL[view.cell]}</div>α = ${fmt(grid[i], 3)}, β = ${fmt(grid[j], 3)}<br>` +
            `loss = <b>${fmt(loss)}</b> <span class="muted">(${signed(delta)} so với θ₀)</span>`,
            event,
          );
        } else {
          hideTip();
        }
      });
      const release = (event) => {
        drag = null;
        if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      };
      canvas.addEventListener("pointerup", release);
      canvas.addEventListener("pointercancel", release);
      canvas.addEventListener("pointerleave", () => { if (!drag) { view.hover = null; draw(view); hideTip(); } });
      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        camera.zoom = clamp(camera.zoom * Math.exp(-event.deltaY * 0.0012), 0.5, 2.6);
        views.forEach(draw);
      }, { passive: false });
    }

    toolbar.querySelector("input").addEventListener("change", (event) => { shared = event.target.checked; drawAll(); });
    toolbar.querySelector("button").addEventListener("click", () => { Object.assign(camera, defaults); drawAll(); });
    new ResizeObserver(() => views.forEach(draw)).observe(row);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
    drawAll();
  }

  // ------------------------------------------------------------- renderers
  const charts = {
    curve(node) {
      const task = node.dataset.task;
      const curves = DATA[task].curves;
      lineChart(node, {
        series: CELLS.map((cell) => ({
          key: cell, label: LABEL[cell], x: curves.steps, y: curves[cell].mean,
          lower: curves[cell].mean.map((m, i) => m - curves[cell].sd[i]),
          upper: curves[cell].mean.map((m, i) => m + curves[cell].sd[i]),
        })),
        xLabel: "bước optimizer",
        yLabel: curves.label,
        xName: "bước",
      });
    },
    dots(node) {
      const task = node.dataset.task;
      const metrics = DATA[task].metrics;
      dotChart(node, {
        rows: CELLS.map((cell) => ({ cell, mean: metrics[cell].mean, sd: metrics[cell].sd, seeds: metrics[cell].seeds })),
        yLabel: DATA[task].metricLabel,
      });
    },
    forest(node) {
      const task = node.dataset.task;
      forestPlot(node, { rows: DATA[task].pairs, xLabel: DATA[task].differenceLabel });
    },
    length(node) {
      const buckets = DATA.sentiment.lengthBuckets;
      lineChart(node, {
        series: CELLS.map((cell) => ({ key: cell, label: LABEL[cell], x: buckets.labels.map((_, i) => i), y: buckets.series[cell] })),
        categories: buckets.labels.map((label, i) => `${label} (n=${buckets.counts[i]})`),
        xLabel: "độ dài câu (token)",
        yLabel: "accuracy trên tập dev",
      });
    },
    reach(node) {
      const reach = DATA.reach[node.dataset.task];
      const series = [];
      for (const cell of CELLS) {
        const entry = reach.cells[cell];
        series.push({ key: cell, label: `${LABEL[cell]} (đo)`, x: entry.lags, y: entry.relative, markers: false });
        series.push({
          key: `${cell}-fit`, cell, label: `${LABEL[cell]} khớp: λ = ${fmt(entry.slope, 4)}`,
          x: entry.lags, y: entry.fitted, dashed: true, markers: false, tooltip: false,
        });
      }
      lineChart(node, { series, xLabel: "độ trễ k tính từ bước cuối", yLabel: "g(k) / g(0)  (thang log)", logY: true, xName: "k", digits: 4 });
    },
    cut(node) {
      const landscape = DATA.landscapes[node.dataset.task];
      const centre = (landscape.grid.length - 1) / 2;
      lineChart(node, {
        series: CELLS.map((cell) => ({ key: cell, label: LABEL[cell], x: landscape.grid, y: landscape.cells[cell].surface.map((row) => row[centre]) })),
        xLabel: "α (β = 0)",
        yLabel: "loss trên tập đánh giá",
        xName: "α",
      });
    },
    context(node) {
      const sweep = DATA.context;
      lineChart(node, {
        series: CELLS.map((cell) => ({ key: cell, label: LABEL[cell], x: sweep.contexts.map((_, i) => i), y: sweep.contexts.map((c) => sweep.results[cell][String(c)]) })),
        categories: sweep.contexts.map((c) => (c === sweep.trained ? `${c} (train)` : String(c))),
        xLabel: "độ dài ngữ cảnh khi đánh giá (ký tự)",
        yLabel: "BPC trên validation",
      });
    },
    surface(node) {
      const buttons = [...node.querySelectorAll(".tabs button")];
      const target = node.querySelector(".surface-target");
      const select = (task) => {
        buttons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.task === task)));
        surfaceExplorer(target, task);
      };
      buttons.forEach((b) => b.addEventListener("click", () => select(b.dataset.task)));
      select(buttons[0].dataset.task);
    },
  };

  for (const node of document.querySelectorAll("[data-chart]")) {
    const render = () => {
      try {
        charts[node.dataset.chart](node);
      } catch (error) {
        node.textContent = `Không vẽ được biểu đồ: ${error.message}`;
        console.error(error);
      }
    };
    render();
    if (node.dataset.chart === "surface") continue; // the viewer resizes its own canvases
    let width = node.getBoundingClientRect().width;
    let pending = 0;
    new ResizeObserver(() => {
      const next = node.getBoundingClientRect().width;
      if (Math.abs(next - width) < 2) return;
      width = next;
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(render);
    }).observe(node);
  }

  // ------------------------------------------------------- table of contents
  const links = [...document.querySelectorAll(".toc a")];
  const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`));
    }
  }, { rootMargin: "-20% 0px -70% 0px" });
  targets.forEach((t) => observer.observe(t));
})();
