"""Live page for PRAVAH ingestion. Vercel loads the ``app`` object."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from pravah.glossary import glossary_payload
from pravah.web.service import PayloadError, iter_events

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "public"), static_url_path="/static")


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/about")
def about_page():
    return send_from_directory(app.static_folder, "about.html")


@app.get("/learn")
def learn_page():
    return send_from_directory(app.static_folder, "learn.html")


@app.get("/links")
def links_page():
    return send_from_directory(app.static_folder, "links.html")


@app.get("/phases.js")
def phases_script():
    return send_from_directory(app.static_folder, "phases.js")


@app.get("/favicon.svg")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg")


@app.get("/api/glossary")
def glossary():
    return jsonify(glossary_payload())


@app.post("/api/ingest")
def ingest_upload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON body."}), 400
    sample = payload.get("sample")
    overrides = payload.get("overrides") if isinstance(payload.get("overrides"), dict) else None
    survival = payload.get("survival") if isinstance(payload.get("survival"), dict) else None
    modules = payload.get("modules") if isinstance(payload.get("modules"), dict) else None
    content = None
    filename = None
    if not sample:
        encoded = payload.get("content_base64")
        if not isinstance(encoded, str) or not encoded:
            return jsonify({"error": "Choose a file first."}), 400
        try:
            content = base64.b64decode(encoded, validate=True)
        except binascii.Error:
            return jsonify({"error": "The upload was incomplete. Try the file again."}), 400
        filename = str(payload.get("filename") or "")
    try:
        stream = iter_events(
            sample=str(sample) if sample else None,
            filename=filename,
            content=content,
            overrides=overrides,
            survival=survival,
            modules=modules,
        )
        # Touch the first line so a bad sample fails before the response starts.
        first = next(stream)
    except PayloadError as exc:
        return jsonify({"error": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Ingestion failed")
        return jsonify(
            {"error": "The file could not be read. Check that it is a CSV, Excel, or XES export."}
        ), 500

    def generate():
        yield first
        yield from stream

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
