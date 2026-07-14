import { app } from "../../../scripts/app.js";

/**
 * SyntaxNodes custom canvas widgets (classic LiteGraph frontend)
 *
 * - Compact value controls bound to the node's original widgets
 * - Side-by-side value and motion-curve workspace for SyntaxFeedbackSampler
 * - Prompt timeline strip for SyntaxPromptTravelKSampler
 *
 * No backend changes: all state lives in the node's original (hidden) widgets.
 * Right-click a node -> "Show classic widgets" to fall back to stock UI.
 */

// ---------------------------------------------------------------------------
// Schedule string <-> keyframe points
// Must accept exactly what the Python side parses: (\d+):\s*\(([^)]+)\)
// ---------------------------------------------------------------------------

const KEYFRAME_RE = /(\d+)\s*:\s*\(([^)]+)\)/g;

export function parseSchedule(text) {
    const points = [];
    if (!text || !text.trim()) return { points, raw: false };
    let match = null;
    let any = false;
    KEYFRAME_RE.lastIndex = 0;
    while ((match = KEYFRAME_RE.exec(text)) !== null) {
        any = true;
        const value = parseFloat(match[2]);
        if (!isFinite(value)) return { points: [], raw: true };
        points.push({ frame: parseInt(match[1], 10), value });
    }
    if (!any) return { points: [], raw: true };
    points.sort((a, b) => a.frame - b.frame);
    return { points, raw: false };
}

export function formatNum(v) {
    if (Number.isInteger(v)) return v.toFixed(1);
    let s = v.toFixed(4).replace(/0+$/, "");
    if (s.endsWith(".")) s += "0";
    return s;
}

export function formatSchedule(points) {
    if (!points.length) return "";
    return points.map((p) => `${p.frame}:(${formatNum(p.value)})`).join(", ");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

export function hideWidget(w) {
    if (!w || w._syntaxHidden) return;
    w._syntaxHidden = true;
    w._origType = w.type;
    w._origComputeSize = w.computeSize;
    w._origHidden = w.hidden;
    w._origComputedHeight = w.computedHeight;
    w.type = "hidden";
    w.hidden = true;
    w.computedHeight = 0;
    w.computeSize = () => [0, -4];
}

export function showWidget(w) {
    if (!w || !w._syntaxHidden) return;
    w._syntaxHidden = false;
    w.type = w._origType;
    w.computeSize = w._origComputeSize;
    w.hidden = w._origHidden;
    if (w._origComputedHeight === undefined) delete w.computedHeight;
    else w.computedHeight = w._origComputedHeight;
}

function eventIs(event, suffix) {
    return event?.type?.endsWith(suffix);
}

export function effectiveWidgetType(widget) {
    return widget?._origType ?? widget?.type;
}

export const PROMPT_EDITOR_HEIGHTS = Object.freeze({
    prompt_schedule: 240,
    negative_prompt_schedule: 160,
});

export function setWidgetHeight(widget, height) {
    if (!widget) return;
    widget._syntaxOriginalComputeSize ??= widget.computeSize;
    widget._syntaxOriginalComputedHeight ??= widget.computedHeight;
    widget.computedHeight = height;
    widget.computeSize = (width) => [width ?? 0, height];
}

function ensureMinSize(node) {
    const min = node.computeSize();
    if (node.size[0] < min[0] || node.size[1] < min[1]) {
        node.setSize([Math.max(node.size[0], min[0]), Math.max(node.size[1], min[1])]);
        node.setDirtyCanvas(true, true);
    }
}

const COLORS = {
    bg: "#14181d",
    cell: "#1d242c",
    cellEdge: "#2b3540",
    accent: "#4fd6ff",
    accentDim: "#2a5a70",
    grid: "#252c34",
    zero: "#39434e",
    axis: "#5c6873",
    line: "#56b6ff",
    point: "#9ad1ff",
    pointActive: "#ffd24a",
    label: "#7d8894",
    value: "#e8eef4",
    hint: "#5c6873",
    tab: "#1d242c",
    tabActive: "#2a5a70",
    padDot: "#ffd24a",
    section: "#4fd6ff",
    seg: ["#2a5a70", "#5a3a7a", "#3a7a5a", "#7a5a3a", "#7a3a4a", "#4a7a3a"],
};

// ---------------------------------------------------------------------------
// Control deck: grid of compact cells bound to hidden default widgets.
// number: drag on the configured axis (shift = 10x) or double-click to type
// combo:  click for menu · toggle: click to flip
// ---------------------------------------------------------------------------

const DECK = { margin: 10, cols: 4, cellH: 30, headerH: 15, gap: 4 };

const LABELS = {
    seed: "seed", control_after_generate: "seed ctrl", seed_variation: "seed var",
    steps: "steps", cfg: "cfg", sampler_name: "sampler", scheduler: "scheduler",
    denoise: "denoise", iterations: "frames", frame_cadence: "cadence", feedback_denoise: "fb denoise",
    zoom_value: "zoom", angle: "angle", translation_x: "tx", translation_y: "ty",
    translation_z: "tz", rotation_3d_x: "rx", rotation_3d_y: "ry", rotation_3d_z: "rz",
    color_coherence: "color mode", color_coherence_strength: "color str",
    contrast_boost: "contrast", sharpen_amount: "sharpen", noise_amount: "noise",
    noise_type: "noise type", lumina_mode: "lumina", temporal_smoothing: "temporal",
    cond_blend_strength: "cond blend", width: "width", height: "height",
    frames_per_transition: "frames/trans", interpolation_mode: "interp", loop: "loop",
    print_output: "print",
};

const FEEDBACK_DECK = [
    { title: "sampling", items: ["seed", "control_after_generate", "seed_variation", "steps", "cfg", "sampler_name", "scheduler", "denoise"] },
    { title: "feedback loop", items: ["iterations", "frame_cadence", "feedback_denoise", "zoom_value"] },
    { title: "motion · static fallbacks", items: ["angle", "translation_x", "translation_y", "translation_z", "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"] },
    { title: "color & detail", items: ["color_coherence", "color_coherence_strength", "contrast_boost", "sharpen_amount", "noise_amount", "noise_type"] },
    { title: "smoothing", items: ["lumina_mode", "temporal_smoothing", "cond_blend_strength"] },
];

const TRAVEL_DECK = [
    { title: "sampling", items: ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"] },
    { title: "canvas", items: ["width", "height"] },
    { title: "travel", items: ["frames_per_transition", "interpolation_mode", "loop"] },
];

function numberStep(w) {
    // ComfyUI stores step * 10 in widget options
    const s = w.options?.step;
    return s ? s / 10 : 1;
}

function numberPrecision(w) {
    if (w.options?.precision !== undefined) return w.options.precision;
    const step = numberStep(w);
    if (Number.isInteger(step)) return 0;
    return Math.min(4, String(step).split(".")[1]?.length ?? 2);
}

function cellValueText(w) {
    if (effectiveWidgetType(w) === "toggle") return w.value ? "on" : "off";
    if (typeof w.value === "number") {
        const prec = numberPrecision(w);
        return w.value > 999999 ? String(w.value) : w.value.toFixed(prec);
    }
    return String(w.value ?? "");
}

function makeDeckWidget(node, sections, options = {}) {
    const deck = { ...DECK, dragAxis: "vertical", ...options };
    // resolve which widgets actually exist on this node
    const resolved = sections
        .map((s) => ({ title: s.title, items: s.items.map((n) => getWidget(node, n)).filter(Boolean) }))
        .filter((s) => s.items.length);

    let height = deck.margin;
    for (const s of resolved) {
        height += deck.headerH + Math.ceil(s.items.length / deck.cols) * (deck.cellH + 3) + deck.gap;
    }
    height += deck.margin - deck.gap;

    const widget = {
        type: "syntax.deck",
        name: "control_deck",
        value: null,
        options: { serialize: false },
        last_y: 0,
        rects: [],
        drag: null,
        lastClick: { name: null, time: 0 },

        computeSize(width) {
            return [width ?? 300, height];
        },

        draw(ctx, drawNode, width, y) {
            if (this._syntaxHidden) return;
            this.last_y = y;
            this.rects = [];
            const innerW = width - deck.margin * 2;
            const cellW = (innerW - (deck.cols - 1) * 3) / deck.cols;
            let cy = y + deck.margin;

            ctx.textBaseline = "middle";
            for (const section of resolved) {
                // section header
                ctx.fillStyle = COLORS.section;
                ctx.font = "9px monospace";
                ctx.textAlign = "left";
                ctx.fillText(section.title.toUpperCase(), deck.margin + 1, cy + deck.headerH / 2);
                ctx.strokeStyle = COLORS.accentDim;
                ctx.lineWidth = 1;
                const tw = ctx.measureText(section.title.toUpperCase()).width;
                ctx.beginPath();
                ctx.moveTo(deck.margin + tw + 8, cy + deck.headerH / 2);
                ctx.lineTo(width - deck.margin, cy + deck.headerH / 2);
                ctx.stroke();
                cy += deck.headerH;

                for (let i = 0; i < section.items.length; i++) {
                    const w = section.items[i];
                    const col = i % deck.cols;
                    const row = Math.floor(i / deck.cols);
                    const cx = deck.margin + col * (cellW + 3);
                    const ry = cy + row * (deck.cellH + 3);

                    ctx.fillStyle = COLORS.cell;
                    ctx.beginPath();
                    ctx.roundRect(cx, ry, cellW, deck.cellH, 3);
                    ctx.fill();
                    // accent underline
                    ctx.fillStyle = this.drag?.widget === w ? COLORS.accent : COLORS.cellEdge;
                    ctx.fillRect(cx, ry + deck.cellH - 2, cellW, 2);

                    ctx.fillStyle = COLORS.label;
                    ctx.font = "8px monospace";
                    ctx.textAlign = "left";
                    ctx.fillText(LABELS[w.name] ?? w.name, cx + 5, ry + 8);

                    ctx.fillStyle = effectiveWidgetType(w) === "toggle" && w.value ? COLORS.accent : COLORS.value;
                    ctx.font = "11px monospace";
                    let text = cellValueText(w);
                    const maxChars = Math.floor((cellW - 10) / 6.2);
                    if (text.length > maxChars) text = text.slice(0, Math.max(1, maxChars - 1)) + "…";
                    ctx.fillText(text, cx + 5, ry + 21);

                    if (effectiveWidgetType(w) === "combo") {
                        ctx.fillStyle = COLORS.label;
                        ctx.font = "8px monospace";
                        ctx.textAlign = "right";
                        ctx.fillText("▾", cx + cellW - 5, ry + 21);
                        ctx.textAlign = "left";
                    }

                    // node-local rect for hit testing (y relative to widget top)
                    this.rects.push({ x: cx, y: ry - y, w: cellW, h: deck.cellH, widget: w });
                }
                cy += Math.ceil(section.items.length / deck.cols) * (deck.cellH + 3) + deck.gap;
            }
        },

        hitCell(pos) {
            const localY = pos[1] - this.last_y;
            return this.rects.find((r) => pos[0] >= r.x && pos[0] <= r.x + r.w && localY >= r.y && localY <= r.y + r.h);
        },

        applyNumber(w, value) {
            const step = numberStep(w);
            value = Math.round(value / step) * step;
            const min = w.options?.min;
            const max = w.options?.max;
            if (min !== undefined) value = Math.max(min, value);
            if (max !== undefined) value = Math.min(max, value);
            const prec = numberPrecision(w);
            w.value = prec > 0 ? parseFloat(value.toFixed(prec)) : Math.round(value);
            w.callback?.(w.value, app.canvas, node);
            node.setDirtyCanvas(true, true);
        },

        mouse(event, pos, mouseNode) {
            if (this._syntaxHidden) return false;

            if (eventIs(event, "down")) {
                const cell = this.hitCell(pos);
                if (!cell) return true;
                const w = cell.widget;
                const now = Date.now();
                const isDouble = this.lastClick.name === w.name && now - this.lastClick.time < 300;
                this.lastClick = { name: w.name, time: now };

                const type = effectiveWidgetType(w);
                if (type === "toggle") {
                    w.value = !w.value;
                    w.callback?.(w.value, app.canvas, node);
                    node.setDirtyCanvas(true, true);
                    return true;
                }
                if (type === "combo") {
                    let values = w.options?.values ?? [];
                    if (typeof values === "function") values = values(w, node);
                    new window.LiteGraph.ContextMenu(values, {
                        event,
                        className: "dark",
                        callback: (v) => {
                            w.value = v;
                            w.callback?.(v, app.canvas, node);
                            node.setDirtyCanvas(true, true);
                        },
                    });
                    return true;
                }
                // number
                if (isDouble) {
                    app.canvas.prompt(LABELS[w.name] ?? w.name, w.value, (v) => {
                        const num = Number(v);
                        if (isFinite(num)) this.applyNumber(w, num);
                    }, event);
                    this.drag = null;
                    return true;
                }
                this.drag = { widget: w, startX: pos[0], startY: pos[1], startValue: w.value };
                return true;
            }

            if (eventIs(event, "move") && this.drag) {
                const w = this.drag.widget;
                const step = numberStep(w);
                const mult = event.shiftKey ? 10 : 1;
                const delta = deck.dragAxis === "horizontal"
                    ? (pos[0] - this.drag.startX) * step * mult
                    : (this.drag.startY - pos[1]) * step * mult;
                this.applyNumber(w, this.drag.startValue + delta);
                return true;
            }

            if (eventIs(event, "up")) {
                this.drag = null;
                return true;
            }
            return false;
        },
    };
    return { widget, managed: resolved.flatMap((s) => s.items) };
}

// ---------------------------------------------------------------------------
// Keyframe curve editor
// ---------------------------------------------------------------------------

export const CURVE_PARAMS = [
    { key: "zoom", sched: "zoom_schedule", range: [-0.5, 0.5] },
    { key: "angle", sched: "angle_schedule", range: [-360, 360] },
    { key: "tx", sched: "translation_x_schedule", range: [-500, 500] },
    { key: "ty", sched: "translation_y_schedule", range: [-500, 500] },
    { key: "tz", sched: "translation_z_schedule", range: [-500, 500] },
    { key: "rx", sched: "rotation_3d_x_schedule", range: [-360, 360] },
    { key: "ry", sched: "rotation_3d_y_schedule", range: [-360, 360] },
    { key: "rz", sched: "rotation_3d_z_schedule", range: [-360, 360] },
];

export function resetMotionSchedules(node) {
    let changed = false;
    for (const { sched } of CURVE_PARAMS) {
        const widget = getWidget(node, sched);
        if (widget && widget.value !== "") {
            widget.value = "";
            changed = true;
        }
    }
    if (changed) node.setDirtyCanvas?.(true, true);
    return changed;
}

const CURVE = { margin: 12, tabH: 20, graphH: 118, hintH: 14, pad: 6 };

function makeCurveWidget(node, options = {}) {
    const curve = { ...CURVE, ...options };
    const totalHeight = curve.tabH + curve.graphH + curve.hintH + curve.pad * 2;
    const widget = {
        type: "syntax.curves",
        name: "motion_curves",
        value: null,
        options: { serialize: false },
        activeTab: 0,
        dragIndex: null,
        last_y: 0,

        computeSize(width) {
            return [width ?? 300, totalHeight];
        },

        param() {
            return CURVE_PARAMS[this.activeTab];
        },

        schedWidget() {
            return getWidget(node, this.param().sched);
        },

        maxFrame() {
            const iterations = getWidget(node, "iterations")?.value ?? 120;
            const { points } = parseSchedule(this.schedWidget()?.value);
            const dataMax = points.length ? points[points.length - 1].frame : 0;
            return Math.max(iterations, dataMax, 2);
        },

        yRange(points) {
            let [lo, hi] = this.param().range;
            for (const p of points) {
                if (p.value < lo) lo = p.value;
                if (p.value > hi) hi = p.value;
            }
            const padY = (hi - lo) * 0.05;
            return [lo - padY, hi + padY];
        },

        graphRect(width) {
            return {
                x: curve.margin,
                y: curve.pad + curve.tabH + 2,
                w: width - curve.margin * 2,
                h: curve.graphH - 4,
            };
        },

        toCanvas(rect, frame, value, maxF, yLo, yHi) {
            const x = rect.x + (frame / maxF) * rect.w;
            const y = rect.y + rect.h - ((value - yLo) / (yHi - yLo)) * rect.h;
            return [x, y];
        },

        fromCanvas(rect, cx, cy, maxF, yLo, yHi) {
            const frame = Math.round(((cx - rect.x) / rect.w) * maxF);
            const value = yLo + ((rect.y + rect.h - cy) / rect.h) * (yHi - yLo);
            return {
                frame: Math.max(0, Math.min(maxF, frame)),
                value: Math.max(yLo, Math.min(yHi, value)),
            };
        },

        writePoints(points) {
            const w = this.schedWidget();
            if (!w) return;
            const seen = new Map();
            for (const p of points) seen.set(p.frame, p.value);
            const clean = [...seen.entries()]
                .map(([frame, value]) => ({ frame, value }))
                .sort((a, b) => a.frame - b.frame);
            w.value = formatSchedule(clean);
            node.setDirtyCanvas(true, true);
        },

        draw(ctx, drawNode, width, y) {
            if (this._syntaxHidden) return;
            this.last_y = y;
            const parsed = parseSchedule(this.schedWidget()?.value);
            const rect = this.graphRect(width);
            const maxF = this.maxFrame();
            const [yLo, yHi] = this.yRange(parsed.points);

            // tab strip
            const tabW = (width - curve.margin * 2) / CURVE_PARAMS.length;
            ctx.font = "10px monospace";
            ctx.textBaseline = "middle";
            for (let i = 0; i < CURVE_PARAMS.length; i++) {
                const tx = curve.margin + i * tabW;
                ctx.fillStyle = i === this.activeTab ? COLORS.tabActive : COLORS.tab;
                ctx.beginPath();
                ctx.roundRect(tx + 1, y + curve.pad, tabW - 2, curve.tabH - 4, 3);
                ctx.fill();
                ctx.fillStyle = i === this.activeTab ? "#fff" : COLORS.label;
                ctx.textAlign = "center";
                const hasData = !!getWidget(node, CURVE_PARAMS[i].sched)?.value?.trim();
                ctx.fillText(CURVE_PARAMS[i].key + (hasData ? "•" : ""), tx + tabW / 2, y + curve.pad + (curve.tabH - 4) / 2);
            }

            const gy = y + rect.y;
            ctx.fillStyle = COLORS.bg;
            ctx.beginPath();
            ctx.roundRect(rect.x, gy, rect.w, rect.h, 4);
            ctx.fill();

            ctx.strokeStyle = COLORS.grid;
            ctx.lineWidth = 1;
            for (let i = 1; i < 4; i++) {
                const ly = gy + (rect.h / 4) * i;
                ctx.beginPath();
                ctx.moveTo(rect.x, ly);
                ctx.lineTo(rect.x + rect.w, ly);
                ctx.stroke();
            }
            if (yLo < 0 && yHi > 0) {
                const zy = gy + rect.h - ((0 - yLo) / (yHi - yLo)) * rect.h;
                ctx.strokeStyle = COLORS.zero;
                ctx.beginPath();
                ctx.moveTo(rect.x, zy);
                ctx.lineTo(rect.x + rect.w, zy);
                ctx.stroke();
            }

            ctx.fillStyle = COLORS.axis;
            ctx.font = "9px monospace";
            ctx.textAlign = "left";
            ctx.fillText(formatNum(yHi), rect.x + 3, gy + 8);
            ctx.fillText(formatNum(yLo), rect.x + 3, gy + rect.h - 6);
            ctx.textAlign = "right";
            ctx.fillText(`${maxF}f`, rect.x + rect.w - 3, gy + rect.h - 6);

            if (parsed.raw) {
                ctx.fillStyle = COLORS.hint;
                ctx.font = "10px monospace";
                ctx.textAlign = "center";
                ctx.fillText("expression schedule — edit as text (right-click node)", rect.x + rect.w / 2, gy + rect.h / 2);
            } else if (!parsed.points.length) {
                ctx.fillStyle = COLORS.hint;
                ctx.font = "10px monospace";
                ctx.textAlign = "center";
                ctx.fillText("click to add keyframes · static value used", rect.x + rect.w / 2, gy + rect.h / 2);
            } else {
                ctx.strokeStyle = COLORS.line;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                const first = this.toCanvas(rect, parsed.points[0].frame, parsed.points[0].value, maxF, yLo, yHi);
                ctx.moveTo(rect.x, gy + (first[1] - rect.y));
                for (const p of parsed.points) {
                    const [px, py] = this.toCanvas(rect, p.frame, p.value, maxF, yLo, yHi);
                    ctx.lineTo(px, gy + (py - rect.y));
                }
                const lastP = parsed.points[parsed.points.length - 1];
                const last = this.toCanvas(rect, lastP.frame, lastP.value, maxF, yLo, yHi);
                ctx.lineTo(rect.x + rect.w, gy + (last[1] - rect.y));
                ctx.stroke();

                for (let i = 0; i < parsed.points.length; i++) {
                    const p = parsed.points[i];
                    const [px, py] = this.toCanvas(rect, p.frame, p.value, maxF, yLo, yHi);
                    ctx.fillStyle = i === this.dragIndex ? COLORS.pointActive : COLORS.point;
                    ctx.beginPath();
                    ctx.arc(px, gy + (py - rect.y), 3.5, 0, Math.PI * 2);
                    ctx.fill();
                }

                if (this.dragIndex !== null && parsed.points[this.dragIndex]) {
                    const p = parsed.points[this.dragIndex];
                    ctx.fillStyle = COLORS.pointActive;
                    ctx.font = "10px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText(`${p.frame}:(${formatNum(p.value)})`, rect.x + rect.w / 2, gy + 10);
                }
            }

            ctx.fillStyle = COLORS.hint;
            ctx.font = "9px monospace";
            ctx.textAlign = "center";
            ctx.fillText("click: add · drag: move · ctrl/alt+click point: delete", width / 2, gy + rect.h + 10);
        },

        mouse(event, pos, mouseNode) {
            if (this._syntaxHidden) return false;
            const localY = pos[1] - this.last_y;
            const width = mouseNode.size[0];
            const rect = this.graphRect(width);

            if (eventIs(event, "down")) {
                if (localY >= curve.pad && localY <= curve.pad + curve.tabH) {
                    const tabW = (width - curve.margin * 2) / CURVE_PARAMS.length;
                    const idx = Math.floor((pos[0] - curve.margin) / tabW);
                    if (idx >= 0 && idx < CURVE_PARAMS.length) {
                        this.activeTab = idx;
                        this.dragIndex = null;
                    }
                    return true;
                }

                if (localY >= rect.y && localY <= rect.y + rect.h && pos[0] >= rect.x && pos[0] <= rect.x + rect.w) {
                    const parsed = parseSchedule(this.schedWidget()?.value);
                    if (parsed.raw) return true;
                    const maxF = this.maxFrame();
                    const [yLo, yHi] = this.yRange(parsed.points);

                    let hit = -1;
                    for (let i = 0; i < parsed.points.length; i++) {
                        const [px, py] = this.toCanvas(rect, parsed.points[i].frame, parsed.points[i].value, maxF, yLo, yHi);
                        if (Math.abs(px - pos[0]) < 7 && Math.abs(py - localY) < 7) {
                            hit = i;
                            break;
                        }
                    }

                    if (hit >= 0 && (event.ctrlKey || event.altKey)) {
                        parsed.points.splice(hit, 1);
                        this.dragIndex = null;
                        this.writePoints(parsed.points);
                        return true;
                    }
                    if (hit >= 0) {
                        this.dragIndex = hit;
                        return true;
                    }
                    const np = this.fromCanvas(rect, pos[0], localY, maxF, yLo, yHi);
                    parsed.points.push(np);
                    parsed.points.sort((a, b) => a.frame - b.frame);
                    this.dragIndex = parsed.points.findIndex((p) => p === np);
                    this.writePoints(parsed.points);
                    return true;
                }
                return true;
            }

            if (eventIs(event, "move") && this.dragIndex !== null) {
                const parsed = parseSchedule(this.schedWidget()?.value);
                if (parsed.raw || !parsed.points.length) return true;
                const maxF = this.maxFrame();
                const [yLo, yHi] = this.yRange(parsed.points);
                const np = this.fromCanvas(rect, pos[0], localY, maxF, yLo, yHi);
                const idx = Math.max(0, Math.min(this.dragIndex, parsed.points.length - 1));
                parsed.points[idx] = np;
                parsed.points.sort((a, b) => a.frame - b.frame);
                this.dragIndex = parsed.points.findIndex((p) => p === np);
                this.writePoints(parsed.points);
                return true;
            }

            if (eventIs(event, "up")) {
                this.dragIndex = null;
                return true;
            }
            return false;
        },
    };
    return widget;
}

// ---------------------------------------------------------------------------
// Feedback workspace: values on the left, motion curves on the right.
// ---------------------------------------------------------------------------

const FEEDBACK_WORKSPACE = { minWidth: 720, gap: 12, minLeft: 220, maxLeft: 390 };

export function feedbackWorkspaceLayout(width) {
    const total = Math.max(Number(width) || FEEDBACK_WORKSPACE.minWidth, 360);
    const leftWidth = Math.max(
        FEEDBACK_WORKSPACE.minLeft,
        Math.min(FEEDBACK_WORKSPACE.maxLeft, Math.round(total * 0.48)),
    );
    const rightX = leftWidth + FEEDBACK_WORKSPACE.gap;
    return {
        total,
        leftWidth,
        rightX,
        rightWidth: Math.max(120, total - rightX),
    };
}

function makeFeedbackWorkspaceWidget(node) {
    const { widget: values, managed } = makeDeckWidget(node, FEEDBACK_DECK, {
        cols: 3,
        dragAxis: "horizontal",
    });
    const panelHeight = values.computeSize(FEEDBACK_WORKSPACE.minLeft)[1];
    const graphH = panelHeight - CURVE.tabH - CURVE.hintH - CURVE.pad * 2;
    const curves = makeCurveWidget(node, { graphH });

    const widget = {
        type: "syntax.feedback_workspace",
        name: "feedback_workspace",
        value: null,
        options: { serialize: false },
        activePane: null,

        computeSize(width) {
            return [Math.max(width ?? 0, FEEDBACK_WORKSPACE.minWidth), panelHeight];
        },

        draw(ctx, drawNode, width, y) {
            if (this._syntaxHidden) return;
            const layout = feedbackWorkspaceLayout(width);
            values.draw(ctx, drawNode, layout.leftWidth, y);

            ctx.save();
            ctx.translate(layout.rightX, 0);
            curves.draw(ctx, drawNode, layout.rightWidth, y);
            ctx.restore();

            ctx.strokeStyle = COLORS.cellEdge;
            ctx.beginPath();
            ctx.moveTo(layout.rightX - FEEDBACK_WORKSPACE.gap / 2, y + 8);
            ctx.lineTo(layout.rightX - FEEDBACK_WORKSPACE.gap / 2, y + panelHeight - 8);
            ctx.stroke();
        },

        mouse(event, pos, mouseNode) {
            if (this._syntaxHidden) return false;
            const layout = feedbackWorkspaceLayout(mouseNode.size[0]);
            if (eventIs(event, "down")) {
                this.activePane = pos[0] < layout.leftWidth ? "values" : "curves";
            }

            let handled = false;
            if (this.activePane === "values") {
                handled = values.mouse(event, pos, { size: [layout.leftWidth, mouseNode.size[1]] });
            } else if (this.activePane === "curves") {
                handled = curves.mouse(
                    event,
                    [pos[0] - layout.rightX, pos[1]],
                    { size: [layout.rightWidth, mouseNode.size[1]] },
                );
            }

            if (eventIs(event, "up")) this.activePane = null;
            return handled;
        },
    };

    return { widget, managed };
}

// ---------------------------------------------------------------------------
// XY motion pad
// ---------------------------------------------------------------------------

const PAD = { h: 138, size: 100, pad: 6 };

function makeXYPadWidget(node) {
    const widget = {
        type: "syntax.xypad",
        name: "motion_pad",
        value: null,
        options: { serialize: false },
        dragging: false,
        lastClick: 0,
        last_y: 0,

        computeSize(width) {
            return [width ?? 300, PAD.h];
        },

        wx() {
            return getWidget(node, "translation_x");
        },
        wy() {
            return getWidget(node, "translation_y");
        },

        range(w) {
            const min = w?.options?.min ?? -500;
            const max = w?.options?.max ?? 500;
            return [min, max];
        },

        padRect(width) {
            return { x: (width - PAD.size) / 2, y: PAD.pad + 12, s: PAD.size };
        },

        draw(ctx, drawNode, width, y) {
            if (this._syntaxHidden) return;
            this.last_y = y;
            const wx = this.wx();
            const wy = this.wy();
            if (!wx || !wy) return;
            const r = this.padRect(width);
            const py = y + r.y;

            ctx.fillStyle = COLORS.label;
            ctx.font = "10px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("motion pad (translation x/y)", width / 2, y + PAD.pad + 4);

            ctx.fillStyle = COLORS.bg;
            ctx.beginPath();
            ctx.roundRect(r.x, py, r.s, r.s, 4);
            ctx.fill();

            ctx.strokeStyle = COLORS.grid;
            ctx.beginPath();
            ctx.moveTo(r.x + r.s / 2, py);
            ctx.lineTo(r.x + r.s / 2, py + r.s);
            ctx.moveTo(r.x, py + r.s / 2);
            ctx.lineTo(r.x + r.s, py + r.s / 2);
            ctx.stroke();

            const [minX, maxX] = this.range(wx);
            const [minY, maxY] = this.range(wy);
            const dx = r.x + ((wx.value - minX) / (maxX - minX)) * r.s;
            const dy = py + ((wy.value - minY) / (maxY - minY)) * r.s;
            ctx.fillStyle = COLORS.padDot;
            ctx.beginPath();
            ctx.arc(dx, dy, 5, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = COLORS.label;
            ctx.textAlign = "center";
            ctx.fillText(`tx: ${formatNum(wx.value)}   ty: ${formatNum(wy.value)}`, width / 2, py + r.s + 12);
        },

        applyFromPos(pos, width) {
            const wx = this.wx();
            const wy = this.wy();
            if (!wx || !wy) return;
            const r = this.padRect(width);
            const localY = pos[1] - this.last_y - r.y;
            const localX = pos[0] - r.x;
            const nx = Math.max(0, Math.min(1, localX / r.s));
            const ny = Math.max(0, Math.min(1, localY / r.s));
            const [minX, maxX] = this.range(wx);
            const [minY, maxY] = this.range(wy);
            wx.value = Math.round((minX + nx * (maxX - minX)) * 10) / 10;
            wy.value = Math.round((minY + ny * (maxY - minY)) * 10) / 10;
            node.setDirtyCanvas(true, true);
        },

        mouse(event, pos, mouseNode) {
            if (this._syntaxHidden) return false;
            const width = mouseNode.size[0];
            const r = this.padRect(width);
            const localY = pos[1] - this.last_y;
            const inside = pos[0] >= r.x && pos[0] <= r.x + r.s && localY >= r.y && localY <= r.y + r.s;

            if (eventIs(event, "down") && inside) {
                const now = Date.now();
                if (now - this.lastClick < 300) {
                    const wx = this.wx();
                    const wy = this.wy();
                    if (wx) wx.value = 0;
                    if (wy) wy.value = 0;
                    node.setDirtyCanvas(true, true);
                    this.dragging = false;
                } else {
                    this.dragging = true;
                    this.applyFromPos(pos, width);
                }
                this.lastClick = now;
                return true;
            }
            if (eventIs(event, "move") && this.dragging) {
                this.applyFromPos(pos, width);
                return true;
            }
            if (eventIs(event, "up")) {
                this.dragging = false;
                return true;
            }
            return false;
        },
    };
    return widget;
}

// ---------------------------------------------------------------------------
// Prompt timeline strip (read-only, Prompt Travel)
// ---------------------------------------------------------------------------

const STRIP = { h: 52, pad: 6, barH: 18 };

function makeTimelineWidget(node) {
    const widget = {
        type: "syntax.timeline",
        name: "prompt_timeline",
        value: null,
        options: { serialize: false },
        last_y: 0,

        computeSize(width) {
            return [width ?? 300, STRIP.h];
        },

        draw(ctx, drawNode, width, y) {
            if (this._syntaxHidden) return;
            this.last_y = y;
            const text = getWidget(node, "prompts")?.value ?? "";
            const fpt = getWidget(node, "frames_per_transition")?.value ?? 30;
            const loop = getWidget(node, "loop")?.value ?? false;
            const prompts = text.split("|").map((p) => p.trim()).filter(Boolean);
            const margin = 12;
            const bw = width - margin * 2;
            const by = y + STRIP.pad + 2;

            ctx.textBaseline = "middle";
            if (prompts.length < 2) {
                ctx.fillStyle = COLORS.hint;
                ctx.font = "10px monospace";
                ctx.textAlign = "center";
                ctx.fillText("need 2+ prompts separated by |", width / 2, by + STRIP.barH / 2);
                return;
            }

            const transitions = prompts.length - 1 + (loop ? 1 : 0);
            const segW = bw / transitions;

            ctx.font = "9px monospace";
            for (let t = 0; t < transitions; t++) {
                ctx.fillStyle = COLORS.seg[t % COLORS.seg.length];
                ctx.beginPath();
                ctx.roundRect(margin + t * segW + 1, by, segW - 2, STRIP.barH, 3);
                ctx.fill();

                const label = prompts[t % prompts.length];
                const maxChars = Math.max(0, Math.floor((segW - 8) / 5.5));
                if (maxChars >= 4) {
                    ctx.fillStyle = "#ddd";
                    ctx.textAlign = "left";
                    ctx.fillText(label.length > maxChars ? label.slice(0, maxChars - 1) + "…" : label, margin + t * segW + 5, by + STRIP.barH / 2);
                }
            }

            ctx.fillStyle = COLORS.label;
            ctx.font = "9px monospace";
            ctx.textAlign = "center";
            const total = transitions * fpt;
            ctx.fillText(`${prompts.length} prompts · ${transitions} transitions · ${total} frames${loop ? " · loop" : ""}`, width / 2, by + STRIP.barH + 12);
        },
    };
    return widget;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

function setupNode(node, deckSections, withCurves) {
    const { widget: deck, managed } = makeDeckWidget(node, deckSections);
    node._syntaxManaged = managed;
    node._syntaxCustom = [deck];

    for (const w of managed) hideWidget(w);
    node.addCustomWidget(deck);

    if (withCurves) {
        for (const p of CURVE_PARAMS) hideWidget(getWidget(node, p.sched));
        const curves = makeCurveWidget(node);
        const pad = makeXYPadWidget(node);
        node.addCustomWidget(curves);
        node.addCustomWidget(pad);
        node._syntaxCustom.push(curves, pad);
    } else {
        const strip = makeTimelineWidget(node);
        node.addCustomWidget(strip);
        node._syntaxCustom.push(strip);
    }

    node.setSize([Math.max(node.size?.[0] ?? 0, 380), node.computeSize()[1]]);
}

function setupFeedbackNode(node) {
    const { widget: workspace, managed } = makeFeedbackWorkspaceWidget(node);
    node._syntaxManaged = managed;
    node._syntaxCustom = [workspace];
    node._syntaxHasCurves = true;

    for (const w of managed) hideWidget(w);
    for (const p of CURVE_PARAMS) hideWidget(getWidget(node, p.sched));
    const promptWidget = getWidget(node, "prompt_schedule");
    const negativePromptWidget = getWidget(node, "negative_prompt_schedule");
    if (promptWidget) promptWidget.label = "prompt / schedule";
    if (negativePromptWidget) negativePromptWidget.label = "negative prompt / schedule";
    for (const [name, height] of Object.entries(PROMPT_EDITOR_HEIGHTS)) {
        setWidgetHeight(getWidget(node, name), height);
    }
    node.addWidget(
        "button",
        "Reset motion schedules",
        null,
        () => resetMotionSchedules(node),
        { serialize: false },
    );
    node.addCustomWidget(workspace);
    node.setSize([Math.max(node.size?.[0] ?? 0, FEEDBACK_WORKSPACE.minWidth), node.computeSize()[1]]);
}

function toggleClassic(node) {
    const goClassic = !node._syntaxClassic;
    node._syntaxClassic = goClassic;
    for (const w of node._syntaxManaged ?? []) (goClassic ? showWidget : hideWidget)(w);
    if (node._syntaxHasCurves || node._syntaxCustom?.some((w) => w.name === "motion_curves")) {
        for (const p of CURVE_PARAMS) (goClassic ? showWidget : hideWidget)(getWidget(node, p.sched));
    }
    for (const w of node._syntaxCustom ?? []) (goClassic ? hideWidget : showWidget)(w);
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

function addClassicToggle(nodeType) {
    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        if (getExtraMenuOptions) getExtraMenuOptions.apply(this, arguments);
        options.push({
            content: this._syntaxClassic ? "Use Syntax workspace" : "Show classic widgets",
            callback: () => toggleClassic(this),
        });
    };
}

function chainConfigure(nodeType) {
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        if (onConfigure) onConfigure.apply(this, arguments);
        ensureMinSize(this);
    };
}

app.registerExtension({
    name: "SyntaxNodes.CustomWidgets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SyntaxFeedbackSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                setupFeedbackNode(this);
            };
            addClassicToggle(nodeType);
            chainConfigure(nodeType);
        }

        if (nodeData.name === "SyntaxPromptTravelKSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                setupNode(this, TRAVEL_DECK, false);
            };
            addClassicToggle(nodeType);
            chainConfigure(nodeType);
        }
    },
});
