from pathlib import Path
import csv

OUT_DIR = Path('docs/assets')
OUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = [
    ('Logistic Regression', 0.869507, 0.658089, 0.575689, 0.768018),
    ('Naive Bayes',         0.815001, 0.425029, 0.313450, 0.659953),
    ('Decision Tree',       0.894918, 0.765093, 0.784468, 0.746652),
    ('SVM',                 0.913823, 0.769231, 0.658413, 0.924901),
    ('XGBoost',             0.937024, 0.838932, 0.751829, 0.948864),
]

with open(OUT_DIR / 'model_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy', 'F1 Score', 'Recall', 'Precision'])
    writer.writerows(metrics)

# Simple SVG bar chart for Accuracy
max_w = 600
bar_h = 38
gap = 14
left = 220
top = 40
height = top + len(metrics) * (bar_h + gap) + 60
width = 900
colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2']

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
svg.append('<rect width="100%" height="100%" fill="#0F172A"/>')
svg.append('<text x="30" y="28" fill="#E2E8F0" font-size="22" font-family="Arial" font-weight="bold">Credit Risk Model Accuracy Benchmark</text>')
for i, (model, acc, *_rest) in enumerate(metrics):
    y = top + i * (bar_h + gap)
    w = int(acc * max_w)
    svg.append(f'<text x="30" y="{y+24}" fill="#CBD5E1" font-size="16" font-family="Arial">{model}</text>')
    svg.append(f'<rect x="{left}" y="{y}" width="{w}" height="{bar_h}" fill="{colors[i%len(colors)]}" rx="6"/>')
    svg.append(f'<text x="{left+w+10}" y="{y+24}" fill="#F8FAFC" font-size="15" font-family="Arial">{acc:.3f}</text>')
svg.append('</svg>')
(OUT_DIR / 'model_benchmark.svg').write_text('\n'.join(svg), encoding='utf-8')

# Simple scatter-like svg for precision/recall
w, h = 900, 520
xmin, xmax = 0.28, 0.85
ymin, ymax = 0.60, 0.98
pad = 80


def sx(x):
    return pad + (x - xmin) / (xmax - xmin) * (w - 2 * pad)

def sy(y):
    return h - pad - (y - ymin) / (ymax - ymin) * (h - 2 * pad)

svg2 = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
svg2.append('<rect width="100%" height="100%" fill="#111827"/>')
svg2.append('<text x="30" y="35" fill="#E5E7EB" font-size="22" font-family="Arial" font-weight="bold">Precision vs Recall by Model</text>')
svg2.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#9CA3AF"/>')
svg2.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#9CA3AF"/>')
svg2.append(f'<text x="{w/2}" y="{h-20}" fill="#D1D5DB" font-size="14">Recall</text>')
svg2.append(f'<text x="20" y="{h/2}" fill="#D1D5DB" font-size="14" transform="rotate(-90 20,{h/2})">Precision</text>')
for i, (model, acc, _f1, rec, pre) in enumerate(metrics):
    x, y = sx(rec), sy(pre)
    r = 7 + int((acc-0.80)*35)
    c = colors[i%len(colors)]
    svg2.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{c}" opacity="0.9"/>')
    svg2.append(f'<text x="{x+12:.1f}" y="{y-10:.1f}" fill="#F9FAFB" font-size="13" font-family="Arial">{model}</text>')
svg2.append('</svg>')
(OUT_DIR / 'precision_recall_tradeoff.svg').write_text('\n'.join(svg2), encoding='utf-8')

print('Generated assets.')
