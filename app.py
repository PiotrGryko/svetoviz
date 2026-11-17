"""Local preview server for the SVETOVIZ GitHub README.

README.md is the deliverable that is published on GitHub. This tiny Flask app
exists only so the document can be rendered and reviewed in a browser, with the
screenshots from assets/images/ served correctly.

    pip install flask markdown
    python app.py
    open http://127.0.0.1:5000
"""

import os
import re

import markdown
from flask import Flask, abort, make_response, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
README_PATH = os.path.join(BASE_DIR, "README.md")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
WEBSITE_URL = "https://www.svetoviz.com"
DOCS_URL = "https://www.svetoviz.com/docs"

app = Flask(__name__)

_md = markdown.Markdown(
    extensions=[
        "tables",
        "fenced_code",
        "toc",
        "sane_lists",
        "md_in_html",
        "attr_list",
    ],
)

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVETOVIZ - README preview</title>
<style>
  :root {
    --bg: #1e1f22;
    --panel: #2b2d31;
    --border: #3c3f45;
    --text: #d7dae0;
    --muted: #9aa0aa;
    --orange: #cc7832;
    --green: #6a8759;
    --blue: #6897bb;
    --code: #26282c;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .topbar {
    position: sticky; top: 0; z-index: 20;
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
    padding: 12px 22px;
    background: rgba(30, 31, 34, 0.96);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(6px);
  }
  .topbar .brand {
    font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--orange); font-size: 13px;
  }
  .topbar .note { color: var(--muted); font-size: 12px; }
  .topbar .links { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
  .topbar a.btn {
    display: inline-block; padding: 6px 12px; border-radius: 5px;
    border: 1px solid var(--border); color: var(--text);
    text-decoration: none; font-size: 12px; letter-spacing: 0.04em;
    text-transform: uppercase; font-weight: 600;
  }
  .topbar a.btn.primary { background: var(--orange); border-color: var(--orange); color: #14150f; }
  .topbar a.btn:hover { border-color: var(--orange); color: var(--orange); }
  .topbar a.btn.primary:hover { color: #14150f; opacity: 0.9; }
  .wrap {
    max-width: 1010px; margin: 0 auto; padding: 42px 26px 120px;
    background: var(--bg);
  }
  .wrap a { color: var(--blue); text-decoration: none; }
  .wrap a:hover { text-decoration: underline; }
  h1, h2, h3, h4 { color: #fff; line-height: 1.25; font-weight: 650; scroll-margin-top: 72px; }
  h1 { font-size: 2.05em; margin: 1.6em 0 0.6em; }
  h2 {
    font-size: 1.55em; margin: 2.4em 0 0.7em; padding-bottom: 0.32em;
    border-bottom: 1px solid var(--border);
  }
  h3 { font-size: 1.18em; margin: 1.9em 0 0.5em; color: var(--green); }
  h4 { font-size: 1em; margin: 1.4em 0 0.4em; color: var(--muted); }
  p, li { color: var(--text); }
  strong { color: #fff; }
  hr { border: 0; border-top: 1px solid var(--border); margin: 2.2em 0; }
  img { max-width: 100%; height: auto; }
  table { border-collapse: collapse; width: 100%; margin: 1.2em 0; font-size: 0.92em; display: block; overflow-x: auto; }
  th, td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; vertical-align: top; }
  th { background: var(--panel); color: #fff; font-weight: 600; }
  tr:nth-child(even) td { background: rgba(255, 255, 255, 0.015); }
  td img { display: block; margin: 0 auto; }
  code {
    background: var(--code); border: 1px solid var(--border); border-radius: 4px;
    padding: 1px 5px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.88em; color: #e0c9a6;
  }
  pre {
    background: var(--code); border: 1px solid var(--border); border-radius: 6px;
    padding: 16px 18px; overflow-x: auto; margin: 1.1em 0;
  }
  pre code { background: none; border: 0; padding: 0; color: #ced2d8; font-size: 0.86em; line-height: 1.55; }
  blockquote {
    margin: 1.2em 0; padding: 0.4em 1.1em; border-left: 3px solid var(--orange);
    background: var(--panel); color: var(--muted);
  }
  div[align="center"] { text-align: center; }
  footer.foot {
    border-top: 1px solid var(--border); padding: 18px 26px; text-align: center;
    color: var(--muted); font-size: 12px;
  }
  footer.foot a { color: var(--orange); text-decoration: none; }
</style>
</head>
<body>
<div class="topbar">
  <span class="brand">SVETOVIZ</span>
  <span class="note">README.md preview</span>
  <span class="links">
    <a class="btn" href="/README.md">Raw markdown</a>
    <a class="btn" href="{docs}">Docs</a>
    <a class="btn primary" href="{site}">www.svetoviz.com</a>
  </span>
</div>
<main class="wrap">
{content}
</main>
<footer class="foot">
  Preview of the GitHub landing page for SVETOVIZ. The live project is at
  <a href="{site}">{site}</a>.
</footer>
</body>
</html>
"""

_cache = {"mtime": None, "html": None}


def render_readme():
    """Convert README.md to HTML, reusing the result until the file changes."""
    if not os.path.exists(README_PATH):
        abort(404, "README.md not found")

    mtime = os.path.getmtime(README_PATH)
    if _cache["mtime"] != mtime:
        with open(README_PATH, encoding="utf-8") as handle:
            source = handle.read()
        # GitHub renders markdown inside raw <div> blocks; python-markdown needs
        # the hint. This is a render-time tweak only, README.md stays untouched.
        source = source.replace('<div align="center">', '<div align="center" markdown="1">')
        _md.reset()
        html = _md.convert(source)
        # keep relative asset paths working when served from a nested route
        html = html.replace('src="assets/', 'src="/assets/')
        _cache["html"] = html
        _cache["mtime"] = mtime
    return _cache["html"]


def readme_stats():
    with open(README_PATH, encoding="utf-8") as handle:
        source = handle.read()
    sections = len(re.findall(r"^## \d+\. ", source, flags=re.MULTILINE))
    images = len(set(re.findall(r"assets/images/[A-Za-z0-9_.\-]+", source)))
    return sections, images, len(source.split("\n"))


@app.route("/")
def index():
    sections, images, lines = readme_stats()
    note = f"{sections} sections | {images} screenshots | {lines} lines"
    page = PAGE.replace("{content}", render_readme())
    page = page.replace("{site}", WEBSITE_URL).replace("{docs}", DOCS_URL)
    page = page.replace("README.md preview", f"README.md preview - {note}")
    return make_response(page)


@app.route("/README.md")
def raw():
    return send_from_directory(BASE_DIR, "README.md", mimetype="text/markdown")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.route("/healthz")
def healthz():
    return {"status": "ok", "readme": os.path.exists(README_PATH)}


def serve(host, port):
    print(f"Serving {README_PATH}")
    print(f"Open http://{host}:{port}/")
    app.run(host=host, port=port, debug=bool(os.environ.get("DEBUG")), use_reloader=False)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    try:
        serve(host, port)
    except (OSError, SystemExit):
        # port already taken (macOS AirPlay Receiver also sits on 5000)
        fallback = port + 1
        print(f"Port {port} is busy, falling back to {fallback}.")
        serve(host, fallback)
