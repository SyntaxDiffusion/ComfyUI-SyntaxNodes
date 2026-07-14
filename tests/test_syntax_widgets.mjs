// Round-trip tests for the schedule parse/format logic in syntax_widgets.js.
// The widget file imports ComfyUI's browser-only app.js, so load it with that
// import stripped. Run: node tests/test_syntax_widgets.mjs
import { readFileSync, writeFileSync, rmSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "..", "web", "js", "syntax_widgets.js"), "utf8");
const stripped = src
    .replace(/^import .*scripts\/app\.js.*$/m, "const app = { registerExtension() {} };")
    .replace(/^export /gm, "export ");
const tmp = join(here, "_widgets_under_test.mjs");
writeFileSync(tmp, stripped);

const {
    parseSchedule,
    formatSchedule,
    formatNum,
    feedbackWorkspaceLayout,
    effectiveWidgetType,
    PROMPT_EDITOR_HEIGHTS,
    hideWidget,
    showWidget,
    setWidgetHeight,
    CURVE_PARAMS,
    resetMotionSchedules,
} = await import(pathToFileURL(tmp));
rmSync(tmp);

function assert(cond, msg) {
    if (!cond) {
        console.error(`FAIL: ${msg}`);
        process.exit(1);
    }
}

// Basic parse — must accept what the Python regex (\d+):\s*\(([^)]+)\) accepts
let r = parseSchedule("0:(0.05), 60:(0.1), 120:(-0.2)");
assert(!r.raw, "numeric schedule should not be raw");
assert(r.points.length === 3, `expected 3 points, got ${r.points.length}`);
assert(r.points[0].frame === 0 && r.points[0].value === 0.05, "point 0 parsed wrong");
assert(r.points[2].value === -0.2, "negative value parsed wrong");

// Out-of-order input gets sorted
r = parseSchedule("60:(1.0), 0:(0.0)");
assert(r.points[0].frame === 0, "points should be sorted by frame");

// Whitespace variants (Python side has \s* between : and ()
r = parseSchedule("0: (0.5),  30 :(1)");
assert(r.points.length === 2, "whitespace variants should parse");

// Empty => static mode, not raw
r = parseSchedule("");
assert(r.points.length === 0 && !r.raw, "empty is static, not raw");

// numexpr expression => raw (leave string untouched)
r = parseSchedule("0:(sin(t/10)*0.5), 30:(0.1)");
assert(r.raw, "expression schedule must be flagged raw");

// Garbage => raw
r = parseSchedule("hello world");
assert(r.raw, "unparseable text must be flagged raw");

// Round trip: format output must re-parse identically
const pts = [
    { frame: 0, value: 0.05 },
    { frame: 60, value: 12 },
    { frame: 120, value: -0.333 },
];
const s = formatSchedule(pts);
const rt = parseSchedule(s);
assert(!rt.raw, `round trip went raw: ${s}`);
assert(rt.points.length === 3, "round trip lost points");
assert(rt.points[1].value === 12, `integer round trip: ${rt.points[1].value}`);
assert(Math.abs(rt.points[2].value - -0.333) < 1e-9, "float round trip");

// Format output must match the Python backend regex exactly
const pyRegex = /(\d+):\s*\(([^)]+)\)/g;
const matches = [...s.matchAll(pyRegex)];
assert(matches.length === 3, `backend regex must match all keyframes in: ${s}`);
assert(matches.every((m) => isFinite(parseFloat(m[2]))), "backend must parse all values as floats");

// formatNum edge cases
assert(formatNum(12) === "12.0", `formatNum int: ${formatNum(12)}`);
assert(formatNum(0.05) === "0.05", `formatNum trailing zeros: ${formatNum(0.05)}`);
assert(formatNum(-0.5) === "-0.5", `formatNum negative: ${formatNum(-0.5)}`);

// Feedback controls and graph must remain side-by-side at the enforced node width.
const layout = feedbackWorkspaceLayout(720);
assert(layout.leftWidth >= 220, `feedback value pane too narrow: ${layout.leftWidth}`);
assert(layout.rightX > layout.leftWidth, "feedback graph must start to the right of values");
assert(layout.rightWidth >= 300, `feedback graph pane too narrow: ${layout.rightWidth}`);
assert(layout.rightX + layout.rightWidth === 720, "feedback panes must fill the node width");

// Hidden native widgets must retain their original interaction behavior in the
// custom panel; otherwise combos become numeric drags and cannot be selected.
assert(effectiveWidgetType({ type: "syntaxhidden", _origType: "combo" }) === "combo",
    "hidden sampler combo lost its original type");
assert(effectiveWidgetType({ type: "syntaxhidden", _origType: "toggle" }) === "toggle",
    "hidden toggle lost its original type");
assert(effectiveWidgetType({ type: "number" }) === "number",
    "visible widget type should remain unchanged");
assert(PROMPT_EDITOR_HEIGHTS.prompt_schedule >= 200,
    "positive prompt schedule editor should support large schedules");
assert(PROMPT_EDITOR_HEIGHTS.negative_prompt_schedule >= 120,
    "negative prompt schedule editor should support large schedules");

const originalComputeSize = () => [320, 80];
const hiddenWidget = {
    type: "customtext",
    computeSize: originalComputeSize,
    computedHeight: 80,
    hidden: false,
};
hideWidget(hiddenWidget);
assert(hiddenWidget.computeSize()[1] === -4, "hidden widget retained layout height");
assert(hiddenWidget.type === "hidden" && hiddenWidget.hidden,
    "hidden component widget remained visible");
assert(hiddenWidget.computedHeight === 0, "hidden component widget retained rendered height");
showWidget(hiddenWidget);
assert(hiddenWidget.computeSize === originalComputeSize, "classic mode did not restore widget sizing");
assert(hiddenWidget.type === "customtext" && !hiddenWidget.hidden,
    "classic mode did not restore component visibility");
assert(hiddenWidget.computedHeight === 80, "classic mode did not restore component height");

const resizedWidget = { computeSize: originalComputeSize, computedHeight: 80 };
setWidgetHeight(resizedWidget, 240);
assert(resizedWidget.computeSize(720)[1] === 240,
    "prompt editor layout did not receive requested height");
assert(resizedWidget.computedHeight === 240,
    "prompt editor component did not receive requested rendered height");

const expectedCurveMappings = {
    zoom: "zoom_schedule",
    angle: "angle_schedule",
    tx: "translation_x_schedule",
    ty: "translation_y_schedule",
    tz: "translation_z_schedule",
    rx: "rotation_3d_x_schedule",
    ry: "rotation_3d_y_schedule",
    rz: "rotation_3d_z_schedule",
};
assert(CURVE_PARAMS.length === Object.keys(expectedCurveMappings).length,
    "motion graph tab count does not match backend schedules");
for (const param of CURVE_PARAMS) {
    assert(expectedCurveMappings[param.key] === param.sched,
        `motion graph ${param.key} maps to unexpected widget ${param.sched}`);
    assert(Array.isArray(param.range) && param.range.length === 2 && param.range[0] < param.range[1],
        `motion graph ${param.key} has an invalid value range`);
}
assert(new Set(CURVE_PARAMS.map((param) => param.sched)).size === CURVE_PARAMS.length,
    "multiple graph tabs write to the same schedule widget");

const motionWidgets = CURVE_PARAMS.map((param, index) => ({
    name: param.sched,
    value: `${index}:(${index + 1}.0)`,
}));
const unrelatedWidget = { name: "prompt_schedule", value: "photorealistic cat" };
let dirtyCalls = 0;
const resetNode = {
    widgets: [...motionWidgets, unrelatedWidget],
    setDirtyCanvas() { dirtyCalls += 1; },
};
assert(resetMotionSchedules(resetNode), "reset did not report clearing populated schedules");
assert(motionWidgets.every((widget) => widget.value === ""),
    "reset did not clear every motion schedule");
assert(unrelatedWidget.value === "photorealistic cat",
    "reset changed a non-motion widget");
assert(dirtyCalls === 1, "reset should refresh the node exactly once");
assert(!resetMotionSchedules(resetNode), "empty motion schedules should be a no-op");
assert(dirtyCalls === 1, "no-op reset unnecessarily refreshed the node");

console.log(`PASS: all schedule parse/format tests (canonical output: "${s}")`);
