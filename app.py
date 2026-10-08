#!/usr/bin/env python3
"""
Qwitter Web Application Server

Integrates the existing Qwitter Python logic (qwitter.py) with a lightweight Flask web app.
Reuses the exact Gemini API setup and .env key configuration.
"""

from flask import Flask, render_template, request, jsonify
from qwitter import ask_qwitter

app = Flask(__name__)


@app.route("/")
def index():
    """Renders the Qwitter web UI."""
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """API endpoint receiving question payload and returning single-word Qwitter JSON response."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    response = ask_qwitter(question)
    return jsonify(response)


if __name__ == "__main__":
    print("Starting Qwitter Web Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
