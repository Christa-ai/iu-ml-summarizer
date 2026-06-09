import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

from model.summarizer import summarize
from utils.metrics import Timer, compute_rouge

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1("Dokumentzusammenfasser"),
                        html.P("Interaktive Webanwendung zur automatischen Textzusammenfassung."),
                    ],
                    className="mt-4 mb-4",
                )
            )
        ),

        # Text input
        dbc.Row(
            dbc.Col(
                [
                    html.Label("Originaltext"),
                    dcc.Textarea(
                        id="input-text",
                        placeholder="Originaltext hier einfügen …",
                        style={"width": "100%", "height": "250px"},
                    ),
                ],
                className="mb-3",
            )
        ),

        # Optional reference text for ROUGE
        dbc.Row(
            dbc.Col(
                [
                    html.Label(
                        [
                            "Referenzzusammenfassung ",
                            html.Small(
                                "(optional – für ROUGE-Auswertung)",
                                className="text-muted",
                            ),
                        ]
                    ),
                    dcc.Textarea(
                        id="reference-text",
                        placeholder="Referenzzusammenfassung hier einfügen (optional) …",
                        style={"width": "100%", "height": "100px"},
                    ),
                ],
                className="mb-4",
            )
        ),

        # Parameter sliders
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Maximale Länge"),
                        dcc.Slider(
                            id="max-length",
                            min=50,
                            max=200,
                            step=10,
                            value=130,
                            marks={v: str(v) for v in range(50, 210, 30)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Minimale Länge"),
                        dcc.Slider(
                            id="min-length",
                            min=10,
                            max=80,
                            step=5,
                            value=30,
                            marks={v: str(v) for v in range(10, 85, 15)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Label("Beam Size"),
                        dcc.Slider(
                            id="num-beams",
                            min=1,
                            max=8,
                            step=1,
                            value=4,
                            marks={v: str(v) for v in range(1, 9)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    width=4,
                ),
            ],
            className="mb-4",
        ),

        # Summarize button
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Zusammenfassen",
                    id="summarize-btn",
                    color="primary",
                    size="lg",
                    className="w-100",
                ),
                className="mb-4",
            )
        ),

        # Loading spinner + summary output
        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    id="loading",
                    type="circle",
                    children=html.Div(id="output-summary"),
                )
            )
        ),

        # ROUGE results
        dbc.Row(
            dbc.Col(
                html.Div(id="output-rouge"),
                className="mt-4 mb-5",
            )
        ),
    ],
    fluid=True,
)


@callback(
    Output("output-summary", "children"),
    Output("output-rouge", "children"),
    Input("summarize-btn", "n_clicks"),
    State("input-text", "value"),
    State("reference-text", "value"),
    State("max-length", "value"),
    State("min-length", "value"),
    State("num-beams", "value"),
    prevent_initial_call=True,
)
def run_summarization(n_clicks, input_text, reference_text, max_len, min_len, num_beams):
    if not input_text or not input_text.strip():
        return (
            dbc.Alert("Bitte geben Sie einen Text ein.", color="warning"),
            None,
        )

    # Clamp min_len so it never exceeds max_len
    min_len = min(min_len, max_len - 1)

    with Timer() as t:
        summary = summarize(
            input_text,
            max_len=max_len,
            min_len=min_len,
            num_beams=num_beams,
        )

    summary_card = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Zusammenfassung", className="card-title"),
                html.P(summary, className="card-text"),
                html.Small(
                    f"Inferenzzeit: {t.elapsed:.2f} s",
                    className="text-muted",
                ),
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
                                    dbc.CardBody(
                                        [
                                            html.H6("ROUGE-1", className="text-center mb-1"),
                                            html.H4(
                                                f"{scores['rouge1']:.4f}",
                                                className="text-center text-primary",
                                            ),
                                        ]
                                    ),
                                    color="light",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H6("ROUGE-2", className="text-center mb-1"),
                                            html.H4(
                                                f"{scores['rouge2']:.4f}",
                                                className="text-center text-success",
                                            ),
                                        ]
                                    ),
                                    color="light",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H6("ROUGE-L", className="text-center mb-1"),
                                            html.H4(
                                                f"{scores['rougeL']:.4f}",
                                                className="text-center text-info",
                                            ),
                                        ]
                                    ),
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
    app.run(debug=True)
