"""Inline a plain predictions.json into the page (no password gate) for design previews.  Never deploy the output."""
import json, sys
from pathlib import Path
src, data, dst = map(Path, sys.argv[1:4])
blob = json.dumps(json.loads(data.read_text()), separators=(",", ":")).replace("</", "<\\/")
dst.write_text(src.read_text().replace("<!--PREVIEW-->", f'<script id="preview-data" type="application/json">{blob}</script>'))
