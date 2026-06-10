"""
app.py — Dash Web Application: Automatische Textzusammenfassung

Entry point for the document summariser web application.
Provides an interactive two-column UI where users upload a PDF document,
configure BART inference parameters, and receive an abstractive summary.
Evaluation results from the offline benchmark run are displayed as
embedded Plotly charts at the bottom of the page.

Run locally:
    python app.py

Production (via Gunicorn):
    gunicorn app:server --bind 0.0.0.0:8050 --workers 1 --timeout 120
"""

import base64
import json
import os

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from model.summarizer import summarize
from utils.metrics import Timer, compute_rouge
from utils.pdf_reader import extract_text_from_b64
from utils.logger import setup_logger, log_request

_log = setup_logger()

# ---------------------------------------------------------------------------
# Load evaluation results at startup (if available)
# ---------------------------------------------------------------------------
_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "eval_results.json")

def _load_eval():
    try:
        with open(_EVAL_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

_EVAL = _load_eval()


def _build_eval_section():
    if _EVAL is None:
        return html.Div()

    s = _EVAL["summary"]
    articles = _EVAL["per_article"]

    r1 = [a["rouge1"] for a in articles]
    r2 = [a["rouge2"] for a in articles]
    rl = [a["rougeL"] for a in articles]
    inf = [a["inference_s"] for a in articles]
    x_idx = list(range(1, len(articles) + 1))

    # ── Chart 1: ROUGE box plot ──────────────────────────────────────────────
    box_fig = go.Figure()
    for label, values, color in [
        ("ROUGE-1", r1, "#0d6efd"),
        ("ROUGE-2", r2, "#198754"),
        ("ROUGE-L", rl, "#0dcaf0"),
    ]:
        box_fig.add_trace(go.Box(
            y=values, name=label, marker_color=color,
            boxmean=True, jitter=0.3, pointpos=-1.6,
            marker=dict(size=4, opacity=0.6),
        ))
    box_fig.update_layout(
        title="ROUGE-Score Verteilung (50 Artikel)",
        yaxis_title="F1-Score",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=40, l=50, r=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    box_fig.update_xaxes(showgrid=False)
    box_fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    # ── Chart 2: ROUGE scores per article (line) ─────────────────────────────
    line_fig = go.Figure()
    for label, values, color, dash in [
        ("ROUGE-1", r1, "#0d6efd", "solid"),
        ("ROUGE-2", r2, "#198754", "dot"),
        ("ROUGE-L", rl, "#0dcaf0", "dash"),
    ]:
        line_fig.add_trace(go.Scatter(
            x=x_idx, y=values, mode="lines+markers",
            name=label, line=dict(color=color, dash=dash, width=1.5),
            marker=dict(size=4),
        ))
    line_fig.update_layout(
        title="ROUGE-Scores je Artikel",
        xaxis_title="Artikel-Nr.",
        yaxis_title="F1-Score",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=40, l=50, r=20),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    line_fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    line_fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return html.Div(
        [
            html.Hr(className="my-4"),
            html.H4("Evaluierungsergebnisse", className="mb-1"),
            html.P(
                f"Modell: {s['model']}  ·  Datensatz: {s['dataset']}  ·  "
                f"Stichprobe: {s['num_samples']} Artikel  ·  Stand: {s['timestamp'][:10]}",
                className="text-muted small mb-4",
            ),

            # Summary metric cards
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("ROUGE-1 Ø", className="text-muted small mb-1"),
                        html.H3(f"{s['avg_rouge1']:.4f}", className="text-primary mb-0"),
                    ]), color="light"), md=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("ROUGE-2 Ø", className="text-muted small mb-1"),
                        html.H3(f"{s['avg_rouge2']:.4f}", className="text-success mb-0"),
                    ]), color="light"), md=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("ROUGE-L Ø", className="text-muted small mb-1"),
                        html.H3(f"{s['avg_rougeL']:.4f}", className="text-info mb-0"),
                    ]), color="light"), md=3),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Ø Inferenzzeit", className="text-muted small mb-1"),
                        html.H3(f"{s['avg_inference_s']:.2f} s", className="text-secondary mb-0"),
                    ]), color="light"), md=3),
                ],
                className="mb-4 g-3",
            ),

            # Charts
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody(
                        dcc.Graph(figure=box_fig, config={"displayModeBar": False})
                    ), color="light"), md=6),
                    dbc.Col(dbc.Card(dbc.CardBody(
                        dcc.Graph(figure=line_fig, config={"displayModeBar": False})
                    ), color="light"), md=6),
                ],
                className="mb-5 g-3",
            ),
        ]
    )

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

_DROPZONE = html.Div(
    [
        html.Div("⬆", style={"fontSize": "2.5rem", "lineHeight": "1"}),
        html.Strong("PDF-Datei hochladen", className="d-block mt-2"),
        html.Span("Klicken oder Datei hierher ziehen", className="text-muted small"),
        html.Div("Maximal 10 MB", className="mt-1 text-muted small"),
    ],
    style={"textAlign": "center", "padding": "2rem 1rem"},
)

app.layout = dbc.Container(
    [
        # ── Header ──────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.Span(
                            "📄",
                            style={
                                "fontSize": "2.5rem",
                                "verticalAlign": "middle",
                                "marginRight": "0.6rem",
                            },
                        ),
                        html.Span(
                            "Automatische Textzusammenfassung",
                            style={
                                "fontSize": "1.8rem",
                                "fontWeight": "600",
                                "verticalAlign": "middle",
                            },
                        ),
                        html.P(
                            "Machine Learning basierte Zusammenfassung von PDF-Dokumenten",
                            className="text-muted mb-0",
                            style={"marginLeft": "3.4rem"},
                        ),
                    ],
                    className="mt-4 mb-4",
                )
            )
        ),

        # ── Two-column body ─────────────────────────────────────────────────
        dbc.Row(
            [
                # ── LEFT ────────────────────────────────────────────────────
                dbc.Col(
                    [
                        # PDF-Upload card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("PDF-Upload", className="card-title mb-1"),
                                    html.P(
                                        "Laden Sie ein PDF-Dokument hoch (maximal 10 MB)",
                                        className="text-muted small mb-3",
                                    ),
                                    dcc.Upload(
                                        id="upload-pdf",
                                        children=_DROPZONE,
                                        accept=".pdf",
                                        max_size=10 * 1024 * 1024,
                                        style={
                                            "border": "2px dashed #ced4da",
                                            "borderRadius": "8px",
                                            "cursor": "pointer",
                                        },
                                        multiple=False,
                                    ),
                                    # File info + preview (shown after upload)
                                    html.Div(id="upload-feedback", className="mt-3"),
                                ]
                            ),
                            className="mb-3",
                        ),

                        # Summarize button
                        dbc.Button(
                            "Zusammenfassen",
                            id="summarize-btn",
                            color="dark",
                            size="lg",
                            className="w-100 mb-4",
                        ),

                        # Loading + output
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=html.Div(id="output-summary"),
                        ),
                        html.Div(id="output-rouge", className="mt-3 mb-5"),
                    ],
                    lg=8,
                ),

                # ── RIGHT sidebar ────────────────────────────────────────────
                dbc.Col(
                    [
                        # Parameter card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Parameter", className="card-title mb-1"),
                                    html.P(
                                        "Steuern Sie die Länge der Zusammenfassung",
                                        className="text-muted small mb-4",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong("Maximale Länge"),
                                                    dbc.Badge(
                                                        id="badge-max-len",
                                                        children="130 Tokens",
                                                        color="secondary",
                                                        className="ms-2",
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-1",
                                            ),
                                            dcc.Slider(
                                                id="max-length",
                                                min=50, max=200, step=10, value=130,
                                                marks={v: str(v) for v in range(50, 210, 50)},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                            html.P(
                                                "Maximale Anzahl von Tokens in der Zusammenfassung",
                                                className="text-muted small mt-1",
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong("Minimale Länge"),
                                                    dbc.Badge(
                                                        id="badge-min-len",
                                                        children="30 Tokens",
                                                        color="secondary",
                                                        className="ms-2",
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-1",
                                            ),
                                            dcc.Slider(
                                                id="min-length",
                                                min=10, max=80, step=5, value=30,
                                                marks={v: str(v) for v in range(10, 85, 20)},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                            html.P(
                                                "Minimale Anzahl von Tokens in der Zusammenfassung",
                                                className="text-muted small mt-1",
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Strong("Beam Size"),
                                                    dbc.Badge(
                                                        id="badge-beams",
                                                        children="4",
                                                        color="secondary",
                                                        className="ms-2",
                                                    ),
                                                ],
                                                className="d-flex align-items-center mb-1",
                                            ),
                                            dcc.Slider(
                                                id="num-beams",
                                                min=1, max=8, step=1, value=4,
                                                marks={v: str(v) for v in range(1, 9, 2)},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                            html.P(
                                                "Breite der Beam-Search (höher = genauer, langsamer)",
                                                className="text-muted small mt-1",
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),

                        # Information card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Information", className="card-title mb-3"),
                                    html.Dl(
                                        [
                                            html.Dt("Unterstützte Formate"),
                                            html.Dd("PDF-Dokumente (max. 10 MB)", className="text-muted small"),
                                            html.Dt("Datengrundlage"),
                                            html.Dd("CNN/DailyMail Dataset (311k Artikel)", className="text-muted small"),
                                            html.Dt("Modell"),
                                            html.Dd("Hugging Face Transformer (finetuned)", className="text-muted small"),
                                            html.Dt("ROUGE-Metriken"),
                                            html.Dd(
                                                "Maß für die Übereinstimmung mit Referenzzusammenfassungen",
                                                className="text-muted small",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            className="mb-3",
                        ),

                        dcc.Textarea(id="reference-text", style={"display": "none"}),
                    ],
                    lg=4,
                ),
            ]
        ),
        # ── Evaluation section ───────────────────────────────────────────────
        dbc.Row(dbc.Col(_build_eval_section())),
    ],
    fluid=True,
    className="px-4",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("upload-feedback", "children"),
    Input("upload-pdf", "contents"),
    State("upload-pdf", "filename"),
    prevent_initial_call=True,
)
def show_upload_feedback(contents, filename):
    """Show file info card and text preview immediately after upload."""
    if not contents:
        return None
    try:
        text, page_count = extract_text_from_b64(contents)
        raw_bytes = base64.b64decode(contents.split(",", 1)[1])
        size_kb = round(len(raw_bytes) / 1024, 1)
    except Exception as exc:
        return dbc.Alert(f"Fehler beim Lesen der PDF: {exc}", color="danger")

    char_count = len(text)
    preview = text[:400] + ("…" if len(text) > 400 else "")

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Span("📄", style={"fontSize": "1.5rem"}),
                                width="auto",
                                className="d-flex align-items-center",
                            ),
                            dbc.Col(
                                [
                                    html.Strong(filename, className="d-block"),
                                    html.Span(
                                        f"{size_kb} KB  ·  {page_count} Seite(n)",
                                        className="text-muted small",
                                    ),
                                ]
                            ),
                        ],
                        align="center",
                    ),
                    className="py-2",
                ),
                style={"border": "1.5px solid #198754"},
                className="mb-3",
            ),
            html.Div(
                [
                    html.Strong("Extrahierter Text (Vorschau)", className="small"),
                    html.Div(
                        preview,
                        style={
                            "background": "#f8f9fa",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "6px",
                            "padding": "0.75rem",
                            "fontSize": "0.8rem",
                            "lineHeight": "1.5",
                            "maxHeight": "160px",
                            "overflowY": "auto",
                            "whiteSpace": "pre-wrap",
                            "marginTop": "0.4rem",
                        },
                    ),
                    html.P(
                        f"{char_count:,} Zeichen extrahiert".replace(",", "."),
                        className="text-muted small mt-1 mb-0",
                    ),
                ]
            ),
        ]
    )


@callback(Output("badge-max-len", "children"), Input("max-length", "value"))
def update_max_badge(val):
    return f"{val} Tokens"


@callback(Output("badge-min-len", "children"), Input("min-length", "value"))
def update_min_badge(val):
    return f"{val} Tokens"


@callback(Output("badge-beams", "children"), Input("num-beams", "value"))
def update_beams_badge(val):
    return str(val)


@callback(
    Output("output-summary", "children"),
    Output("output-rouge", "children"),
    Input("summarize-btn", "n_clicks"),
    State("upload-pdf", "contents"),
    State("upload-pdf", "filename"),
    State("reference-text", "value"),
    State("max-length", "value"),
    State("min-length", "value"),
    State("num-beams", "value"),
    prevent_initial_call=True,
)
def run_summarization(n_clicks, contents, filename, reference_text, max_len, min_len, num_beams):
    if not contents:
        _log.warning("[validation] Zusammenfassen ohne hochgeladene PDF angeklickt.")
        return dbc.Alert("Bitte laden Sie zuerst eine PDF-Datei hoch.", color="warning"), None

    try:
        input_text, pages = extract_text_from_b64(contents)
    except Exception as exc:
        _log.error("[request] PDF '%s' nicht lesbar: %s", filename, exc)
        log_request(
            _log,
            filename=filename or "unknown",
            char_count=0, pages=0,
            max_len=max_len, min_len=min_len, num_beams=num_beams,
            summary="", inference_s=0.0, truncated=False,
            status="error", error=str(exc),
        )
        return dbc.Alert(f"PDF konnte nicht gelesen werden: {exc}", color="danger"), None

    if not input_text.strip():
        _log.warning("[validation] '%s': kein extrahierbarer Text.", filename)
        return dbc.Alert("Die PDF enthält keinen extrahierbaren Text.", color="warning"), None

    # BART input limit: 1 024 tokens ≈ ~4 000 characters (rough estimate)
    truncated = len(input_text) > 4000
    if truncated:
        _log.warning("[validation] '%s': Eingabe > 4000 Zeichen, wird auf 1024 Tokens gekürzt.", filename)

    min_len = min(min_len, max_len - 1)

    with Timer() as t:
        summary = summarize(input_text, max_len=max_len, min_len=min_len, num_beams=num_beams)

    log_request(
        _log,
        filename=filename or "unknown",
        char_count=len(input_text),
        pages=pages,
        max_len=max_len, min_len=min_len, num_beams=num_beams,
        summary=summary,
        inference_s=t.elapsed,
        truncated=truncated,
    )

    truncation_hint = dbc.Alert(
        "Hinweis: Der Text überschreitet 1.024 Tokens. Die Eingabe wurde automatisch gekürzt.",
        color="info",
        className="mb-2 py-2 small",
    ) if truncated else None

    summary_card = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Zusammenfassung", className="card-title"),
                truncation_hint,
                html.P(summary, className="card-text"),
                html.Small(f"Inferenzzeit: {t.elapsed:.2f} s", className="text-muted"),
            ]
        ),
        color="light",
        className="mb-3",
    )

    rouge_section = None
    if reference_text and reference_text.strip():
        scores = compute_rouge(summary, reference_text)
        rouge_section = dbc.Card(
            dbc.CardBody(
                [
                    html.H5("ROUGE-Auswertung", className="card-title mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("ROUGE-1", className="text-center mb-1"),
                                        html.H4(f"{scores['rouge1']:.4f}", className="text-center text-primary"),
                                    ]),
                                    color="light",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("ROUGE-2", className="text-center mb-1"),
                                        html.H4(f"{scores['rouge2']:.4f}", className="text-center text-success"),
                                    ]),
                                    color="light",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H6("ROUGE-L", className="text-center mb-1"),
                                        html.H4(f"{scores['rougeL']:.4f}", className="text-center text-info"),
                                    ]),
                                    color="light",
                                ),
                                width=4,
                            ),
                        ]
                    ),
                ]
            ),
            color="light",
        )

    return summary_card, rouge_section


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8051, debug=False)
