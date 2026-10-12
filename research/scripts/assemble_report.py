"""Inline generated tables + overlap results into the report template.
Usage: python -I assemble_report.py <research_dir> <out_md>
"""
import sys

D, out = sys.argv[1], sys.argv[2]
body = open(f"{D}/work/report_body.md").read()
for name in ("papers", "repos", "models", "datasets", "competitions"):
    body = body.replace("{{TABLE:%s}}" % name, open(f"{D}/work/report_tables/{name}.md").read().strip())
ov = []
for f, label in (("overlap_vishlb.txt", "vishlb/speech-emotion-recognition-hindi (ODbL)"), ("overlap_swast.txt", "swastikatinkhede/multilingual-speech-dataset, Hindi subset (CC BY 4.0)")):
    ov.append(f"- {label}: `{open(f'{D}/work/{f}').read().strip().splitlines()[-1]}` → **no overlap** (true twins score ≥ 0.95).")
ov.append("- github.com/Himanshi7104/SRCASW-BhavVaani (sample of the competition's source data): 20/20 clips match competition audio (NCC ≥ 0.99), 5 of them test rows → **Not recommended / rule violation** (§16).")
ov.append("- Earlier hash-based check (exact PCM MD5 + coarse quantized fingerprint) on the same two Kaggle datasets also found 0 matches.")
body = body.replace("{{OVERLAP}}", "\n".join(ov))
assert "{{" not in body, "unfilled placeholder"
open(out, "w").write(body)
print("report chars", len(body), "lines", body.count("\n"))
