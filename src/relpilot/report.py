import sys, json, pandas as pd
from pathlib import Path

def generate_report(run_dir: str) -> Path:
    p = Path(run_dir)
    m = json.loads((p / "metrics.json").read_text())
    preds = pd.read_parquet(p / "predictions.parquet")
    pred_col = [c for c in preds.columns if c.endswith("_pred")][0]
    top = preds.sort_values(pred_col, ascending=False).head(8)

    pts, svg_elems = [], ['<line x1="40" y1="180" x2="280" y2="20" stroke="#94a3b8" stroke-dasharray="4" stroke-width="1.5"/>']
    for b in m.get("reliability_bins", []):
        if b.get("count", 0) > 0:
            x, y = 40 + b["mean_pred"] * 240, 180 - b["empirical_rate"] * 160
            pts.append(f"{x:.1f},{y:.1f}")
            svg_elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>')
    if pts:
        svg_elems.insert(0, f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    svg = f'''<svg viewBox="0 0 320 220" style="width:100%;max-width:320px;background:#f8fafc;border-radius:8px;padding:8px">
        <line x1="40" y1="20" x2="40" y2="180" stroke="#cbd5e1"/><line x1="40" y1="180" x2="280" y2="180" stroke="#cbd5e1"/>
        <text x="40" y="195" font-size="10" fill="#64748b">0.0</text><text x="270" y="195" font-size="10" fill="#64748b">1.0</text>
        <text x="15" y="180" font-size="10" fill="#64748b">0</text><text x="15" y="25" font-size="10" fill="#64748b">1</text>
        <text x="110" y="210" font-size="11" fill="#475569">Mean Predicted</text>
        <text x="5" y="105" font-size="11" fill="#475569" transform="rotate(-90 15,105)">Empirical</text>
        {"".join(svg_elems)}</svg>'''

    rows = "".join(f"<tr><td style='font-family:monospace'>{r.iloc[0][:16]}...</td><td>{str(r.iloc[1])[:10]}</td><td><b>{r[pred_col]:.3f}</b></td></tr>" for _, r in top.iterrows())
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>RelPilot: {m['run_id']}</title>
    <style>body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;background:#f1f5f9;color:#0f172a}}
    .c{{background:#fff;border-radius:10px;padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
    table{{width:100%;border-collapse:collapse;margin-top:8px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #e2e8f0;font-size:13px}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:15px;margin:0 0 8px;color:#334155}}
    .badge{{display:inline-block;padding:4px 8px;background:#dbeafe;color:#1e40af;border-radius:4px;font-size:12px;font-weight:600}}</style></head><body>
    <div class="c"><h1>RelPilot Trust &amp; Action Report</h1><span class="badge">task: {m['task']}</span> <span class="badge">run: {m['run_id']}</span></div>
    <div class="grid"><div class="c"><h2>Reliability Chart (ECE: {m.get('ece',0):.3f})</h2>{svg}</div>
    <div class="c"><h2>Gate Decision &amp; Baselines</h2>
    <p><b>Recommended Threshold (&tau;):</b> {m.get('recommended_threshold','N/A')}<br>
    <b>Backtest Support:</b> {m.get('threshold_support','N/A')} entities &nbsp;|&nbsp; <b>Precision:</b> {m.get('threshold_precision',0)*100:.1f}%</p>
    <table><tr><th>Model</th><th>AUROC</th></tr>
    <tr><td><b>TabPFN-Rel</b></td><td><b>{m.get('auroc',0):.4f}</b></td></tr>
    <tr><td>Global Constant</td><td>{m.get('baseline_global_auroc',0.5):.4f}</td></tr>
    <tr><td>Per-Entity Constant</td><td>{m.get('baseline_per_entity_auroc',0.5):.4f}</td></tr></table></div></div>
    <div class="c"><h2>Top Risk Entities</h2><table><tr><th>Entity</th><th>Timestamp</th><th>Risk Score</th></tr>{rows}</table></div>
    </body></html>"""

    out = p / "report.html"
    out.write_text(html)
    print(f"Report generated: {out}")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m relpilot.report runs/<id>")
        sys.exit(1)
    generate_report(sys.argv[1])
