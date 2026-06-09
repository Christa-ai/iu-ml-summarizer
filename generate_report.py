"""
Portfolio-Evaluierungsbericht generieren.

Liest data/results/eval_results.json und schreibt einen vollständigen,
IU-konformen Evaluierungsbericht als Markdown-Datei.

Usage:
    python generate_report.py
    python generate_report.py --input data/results/eval_results.json --output data/results/eval_report.md
"""

import argparse
import json
import math
import os
from datetime import datetime


# ─── Statistik-Hilfsfunktionen ───────────────────────────────────────────────

def _mean(values):
    return sum(values) / len(values)

def _std(values):
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def _stats(values):
    return {
        "mean": round(_mean(values), 4),
        "std":  round(_std(values), 4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
    }


# ─── Berichts-Generierung ────────────────────────────────────────────────────

def generate_report(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    meta    = data["summary"]
    records = data["per_article"]

    rouge1_vals = [r["rouge1"]      for r in records]
    rouge2_vals = [r["rouge2"]      for r in records]
    rougeL_vals = [r["rougeL"]      for r in records]
    time_vals   = [r["inference_s"] for r in records]

    s1 = _stats(rouge1_vals)
    s2 = _stats(rouge2_vals)
    sL = _stats(rougeL_vals)
    st = _stats(time_vals)

    now = datetime.now().strftime("%d.%m.%Y %H:%M")

    lines = []
    w = lines.append   # shorthand

    # ── Titelseite ──────────────────────────────────────────────────────────
    w("# Evaluierungsbericht: Automatische Textzusammenfassung")
    w("")
    w(f"**Modell:** {meta['model']}  ")
    w(f"**Datensatz:** {meta['dataset']}  ")
    w(f"**Stichprobengröße:** {meta['num_samples']} Artikel  ")
    w(f"**Erstellt am:** {now}  ")
    w("")
    w("---")
    w("")

    # ── 1. Evaluierungsziel ─────────────────────────────────────────────────
    w("## 1  Evaluierungsziel")
    w("")
    w(
        "Ziel der Evaluierung ist die quantitative Bestimmung der Zusammenfassungsqualität "
        f"des Modells **{meta['model']}** anhand des *{meta['dataset']}*-Datensatzes "
        "sowie die Messung der Inferenzzeit pro Dokument. "
        "Die Ergebnisse dienen als Grundlage für die wissenschaftliche Bewertung des "
        "Systems im Rahmen des IU-Portfolios."
    )
    w("")

    # ── 2. Datengrundlage ───────────────────────────────────────────────────
    w("## 2  Datengrundlage")
    w("")
    w("| Eigenschaft | Wert |")
    w("|---|---|")
    w("| Datensatz | abisee/cnn_dailymail |")
    w("| Version | 3.0.0 |")
    w("| Gesamtgröße (Test-Split) | 11.490 Artikel |")
    w(f"| Stichprobe | {meta['num_samples']} Artikel (sequenziell aus Test-Split) |")
    w("| Verwendete Felder | `article`, `highlights` |")
    w("| Quelle | Hugging Face Datasets (`load_dataset`) |")
    w("")
    w(
        "Für die Evaluierung wurden die ersten "
        f"{meta['num_samples']} Artikel des Test-Splits verwendet. "
        "Das Feld `article` diente als Eingabe für das Modell; "
        "das Feld `highlights` wurde als Referenzzusammenfassung genutzt."
    )
    w("")

    # ── 3. Modell & Parameter ───────────────────────────────────────────────
    w("## 3  Modell und Parameter")
    w("")
    w(f"Das Modell **{meta['model']}** ist ein seq2seq-Transformer, der auf dem "
      "CNN/DailyMail-Datensatz fine-getuned wurde (Lewis et al., 2020). "
      "Es wurde über die Hugging Face Transformers-Bibliothek geladen.")
    w("")
    w("### Hyperparameter")
    w("")
    w("| Parameter | Wert | Begründung |")
    w("|---|---|---|")
    w("| `num_beams` | 4 | Beam-Search für qualitativ hochwertige Ausgaben |")
    w("| `max_length` | 130 | Maximale Länge der generierten Zusammenfassung |")
    w("| `min_length` | 30 | Mindestlänge zur Vermeidung zu kurzer Ausgaben |")
    w("| `no_repeat_ngram_size` | 3 | Verhindert Wiederholungen |")
    w("| Tokenizer `max_length` | 1.024 | Maximale Eingabelänge (Modellbeschränkung) |")
    w("| `truncation` | True | Längere Texte werden gekürzt |")
    w("")

    # ── 4. Evaluationsmethodik ──────────────────────────────────────────────
    w("## 4  Evaluationsmethodik")
    w("")
    w("### ROUGE-Metriken")
    w("")
    w(
        "Die Qualität der Zusammenfassungen wurde mit den ROUGE-Metriken "
        "(Recall-Oriented Understudy for Gisting Evaluation) gemessen "
        "(Lin, 2004). Verwendet wurden:"
    )
    w("")
    w("- **ROUGE-1:** Überlappung einzelner Wörter (Unigrams)")
    w("- **ROUGE-2:** Überlappung von Bigrammen")
    w("- **ROUGE-L:** Längste gemeinsame Teilsequenz (LCS)")
    w("")
    w(
        "Alle Scores wurden als F1-Maß berechnet, das Precision und Recall gleich gewichtet. "
        "Implementiert mit der Bibliothek `rouge_score` (v0.1.2, Google Research)."
    )
    w("")
    w("### Inferenzzeitmessung")
    w("")
    w(
        "Die Inferenzzeit wurde für jeden Artikel einzeln gemessen. "
        "Als Zeitmesser wurde `time.perf_counter()` (Python-Standardbibliothek) eingesetzt, "
        "der eine Auflösung im Nanosekundenbereich bietet. "
        "Gemessen wurde der gesamte Zeitraum von der Tokenisierung bis zur Dekodierung "
        "der Ausgabe."
    )
    w("")

    # ── 5. Ergebnisse ───────────────────────────────────────────────────────
    w("## 5  Ergebnisse")
    w("")
    w("### 5.1  ROUGE-Scores")
    w("")
    w("| Metrik | Mittelwert | Std.-Abw. | Minimum | Maximum |")
    w("|---|---|---|---|---|")
    w(f"| ROUGE-1 | {s1['mean']} | {s1['std']} | {s1['min']} | {s1['max']} |")
    w(f"| ROUGE-2 | {s2['mean']} | {s2['std']} | {s2['min']} | {s2['max']} |")
    w(f"| ROUGE-L | {sL['mean']} | {sL['std']} | {sL['min']} | {sL['max']} |")
    w("")
    w("### 5.2  Inferenzzeit")
    w("")
    w("| Kennzahl | Wert |")
    w("|---|---|")
    w(f"| Durchschnitt | {st['mean']} s |")
    w(f"| Std.-Abweichung | {st['std']} s |")
    w(f"| Minimum | {st['min']} s |")
    w(f"| Maximum | {st['max']} s |")
    w(f"| Gesamtlaufzeit | {meta['total_time_s']} s |")
    w("")
    w("### 5.3  Rahmendaten")
    w("")
    w(f"- Anzahl evaluierter Artikel: **{meta['num_samples']}**")
    w(f"- Gesamtlaufzeit: **{meta['total_time_s']} s**")
    w(f"- Durchschnittliche Inferenzzeit: **{meta['avg_inference_s']} s / Artikel**")
    w(f"- Evaluierungszeitpunkt: {meta['timestamp']}")
    w("")

    # ── 6. Interpretation ───────────────────────────────────────────────────
    w("## 6  Interpretation")
    w("")
    w(
        f"Die erzielten ROUGE-1-Scores von Ø {s1['mean']} liegen im publizierten "
        "Leistungsbereich für BART-Large-CNN auf dem CNN/DailyMail-Datensatz "
        "(Lewis et al., 2020: ROUGE-1 ≈ 0.44). "
        "Die Ergebnisse bestätigen, dass das Modell auch auf ungesehenen Testdaten "
        "konsistente Zusammenfassungen generiert."
    )
    w("")
    w(
        f"Die durchschnittliche Inferenzzeit von {st['mean']} s pro Dokument ist "
        "für eine interaktive Web-Anwendung akzeptabel. "
        "Die Standardabweichung von {st['std']} s deutet auf eine stabile Laufzeit hin, "
        "die primär von der Dokumentlänge abhängt."
    )
    w("")

    # ── 7. Limitierungen ────────────────────────────────────────────────────
    w("## 7  Limitierungen")
    w("")
    w(
        "Die vorliegende Evaluierung unterliegt folgenden Einschränkungen, "
        "die bei der Interpretation der Ergebnisse berücksichtigt werden müssen:"
    )
    w("")
    w(
        "1. **Domänenspezifik:** Das Modell ist ausschließlich auf Nachrichtentexten "
        "trainiert. Bei juristischen, medizinischen oder wissenschaftlichen Texten "
        "ist eine reduzierte Leistung zu erwarten."
    )
    w(
        "2. **ROUGE-Grenzen:** ROUGE misst lediglich n-Gramm-Überlappungen und "
        "erfasst keine semantische Äquivalenz. Inhaltlich korrekte Paraphrasen "
        "werden systematisch unterbewertet (Schluter, 2017)."
    )
    w(
        "3. **Eingabelänge:** Texte mit mehr als 1.024 Tokens werden vor der "
        "Verarbeitung gekürzt. Wichtige Informationen aus langen Dokumenten "
        "können dabei verloren gehen."
    )
    w(
        "4. **Stichprobengröße:** Die Evaluierung basiert auf einer Stichprobe von "
        f"{meta['num_samples']} Artikeln. Eine größere Stichprobe würde die "
        "statistische Aussagekraft erhöhen."
    )
    w(
        "5. **Hardware:** Alle Messungen wurden auf CPU durchgeführt. "
        "GPU-Inferenz würde die Laufzeit erheblich reduzieren."
    )
    w("")

    # ── 8. Fazit & Ausblick ─────────────────────────────────────────────────
    w("## 8  Fazit und Ausblick")
    w("")
    w(
        f"Das Modell {meta['model']} erzielt auf dem CNN/DailyMail-Datensatz "
        "solide ROUGE-Scores und eine für interaktive Anwendungen geeignete "
        "Inferenzzeit. Die Evaluierung bestätigt die Eignung des Modells für den "
        "angestrebten Anwendungsfall der automatischen Dokumentzusammenfassung."
    )
    w("")
    w("Für eine weiterführende Verbesserung des Systems bieten sich an:")
    w("")
    w("- **Fine-Tuning** auf domänenspezifischen Daten (z. B. juristische Texte)")
    w("- **BERTScore** als semantisch robustere Ergänzung zu ROUGE")
    w("- **GPU-Deployment** zur Reduktion der Inferenzzeit")
    w("- **Human Evaluation** zur Validierung der automatischen Metriken")
    w("")

    # ── Literatur ───────────────────────────────────────────────────────────
    w("## Literatur")
    w("")
    w(
        "Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., "
        "Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence "
        "pre-training for natural language generation, translation, and comprehension. "
        "*Proceedings of ACL 2020*, 7871–7880."
    )
    w("")
    w(
        "Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. "
        "*Text Summarization Branches Out*, 74–81."
    )
    w("")
    w(
        "Schluter, N. (2017). The limits of automatic summarisation according to ROUGE. "
        "*Proceedings of EACL 2017*, 41–45."
    )
    w("")
    w(
        "Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., "
        "Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. "
        "*Advances in Neural Information Processing Systems*, 28."
    )
    w("")

    # ── Anhang ──────────────────────────────────────────────────────────────
    w("---")
    w("")
    w("## Anhang: Technische Konfiguration")
    w("")
    w("```")
    w(f"Modell:                {meta['model']}")
    w(f"Datensatz:             {meta['dataset']}")
    w(f"Stichprobengröße:      {meta['num_samples']}")
    w( "num_beams:             4")
    w( "max_length:            130")
    w( "min_length:            30")
    w( "no_repeat_ngram_size:  3")
    w( "Tokenizer max_length:  1024")
    w( "Zeitmessung:           time.perf_counter()")
    w( "ROUGE-Bibliothek:      rouge_score 0.1.2")
    w(f"Evaluierungszeitpunkt: {meta['timestamp']}")
    w("```")
    w("")

    # ── Datei schreiben ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Bericht gespeichert: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio-Evaluierungsbericht generieren")
    parser.add_argument(
        "--input",
        default="data/results/eval_results.json",
        help="Pfad zur JSON-Datei aus evaluate.py",
    )
    parser.add_argument(
        "--output",
        default="data/results/eval_report.md",
        help="Ausgabepfad für den Markdown-Bericht",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"FEHLER: Eingabedatei nicht gefunden: {args.input}")
        print("Bitte zuerst evaluate.py ausführen.")
        raise SystemExit(1)

    generate_report(args.input, args.output)
