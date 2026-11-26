"""User-facing frontend (Flask). Renders a form built from config.yaml and
calls the model API over HTTP — it never loads the model itself."""

from __future__ import annotations

import os

import requests
from flask import Flask, render_template, request

from mlops_core.config import load_config

API_URL = os.environ.get("API_URL", "http://api:8000")
CFG = load_config()

app = Flask(__name__)


def _form_fields() -> list[dict]:
    feats = CFG["features"]
    fields = []
    for col in feats["numeric"]:
        lo, hi = feats.get("bounds", {}).get(col, (None, None))
        fields.append(
            {
                "name": col,
                "label": col.replace("_", " ").title(),
                "type": "number",
                "default": feats.get("defaults", {}).get(col, ""),
                "min": lo,
                "max": hi,
            }
        )
    for col in feats["categorical"]:
        fields.append(
            {
                "name": col,
                "label": col.replace("_", " ").title(),
                "type": "text",
                "default": feats.get("defaults", {}).get(col, ""),
            }
        )
    return fields


def _parse_form(form) -> tuple[dict, list[str]]:
    payload, errors = {}, []
    for field in _form_fields():
        raw = form.get(field["name"], "").strip()
        if raw == "":
            errors.append(f"{field['label']} is required.")
            continue
        if field["type"] == "number":
            try:
                payload[field["name"]] = float(raw)
            except ValueError:
                errors.append(f"{field['label']} must be a number.")
        else:
            payload[field["name"]] = raw
    return payload, errors


def _api_error_message(resp: requests.Response) -> str:
    if resp.status_code == 422:
        try:
            details = resp.json().get("detail", [])
            msgs = [f"{d['loc'][-1].replace('_', ' ')}: {d['msg']}" for d in details]
            return "Please check your input — " + "; ".join(msgs)
        except Exception:
            return "Please check your input values."
    if resp.status_code == 503:
        return "The model is still loading. Please try again in a few seconds."
    return f"The prediction service returned an error ({resp.status_code})."


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, values = None, [], {f["name"]: f["default"] for f in _form_fields()}
    if request.method == "POST":
        values.update(request.form.to_dict())
        payload, errors = _parse_form(request.form)
        if not errors:
            try:
                resp = requests.post(
                    f"{API_URL}/predict", json=payload, timeout=10, headers={"X-Source": "frontend"}
                )
                if resp.ok:
                    result = resp.json()
                    result["percent"] = round(result["probability"] * 100, 1)
                    result["positive"] = result["prediction"] == 1
                else:
                    errors.append(_api_error_message(resp))
            except requests.RequestException:
                errors.append("Could not reach the prediction service. Is the API running?")
    return render_template(
        "index.html",
        project=CFG["project"],
        fields=_form_fields(),
        values=values,
        result=result,
        errors=errors,
        classes=CFG["classes"],
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
