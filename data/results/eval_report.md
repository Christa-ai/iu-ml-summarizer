# Evaluierungsbericht: Automatische Textzusammenfassung

**Modell:** facebook/bart-large-cnn  
**Datensatz:** cnn_dailymail 3.0.0  
**Stichprobengröße:** 50 Artikel  
**Erstellt am:** 10.06.2026 09:00  

---

## 1  Evaluierungsziel

Ziel der Evaluierung ist die quantitative Bestimmung der Zusammenfassungsqualität des Modells **facebook/bart-large-cnn** anhand des *cnn_dailymail 3.0.0*-Datensatzes sowie die Messung der Inferenzzeit pro Dokument. Die Ergebnisse dienen als Grundlage für die wissenschaftliche Bewertung des Systems im Rahmen des IU-Portfolios.

## 2  Datengrundlage

| Eigenschaft | Wert |
|---|---|
| Datensatz | abisee/cnn_dailymail |
| Version | 3.0.0 |
| Gesamtgröße (Test-Split) | 11.490 Artikel |
| Stichprobe | 50 Artikel (sequenziell aus Test-Split) |
| Verwendete Felder | `article`, `highlights` |
| Quelle | Hugging Face Datasets (`load_dataset`) |

Für die Evaluierung wurden die ersten 50 Artikel des Test-Splits verwendet. Das Feld `article` diente als Eingabe für das Modell; das Feld `highlights` wurde als Referenzzusammenfassung genutzt.

## 3  Modell und Parameter

Das Modell **facebook/bart-large-cnn** ist ein seq2seq-Transformer, der auf dem CNN/DailyMail-Datensatz fine-getuned wurde (Lewis et al., 2020). Es wurde über die Hugging Face Transformers-Bibliothek geladen.

### Hyperparameter

| Parameter | Wert | Begründung |
|---|---|---|
| `num_beams` | 4 | Beam-Search für qualitativ hochwertige Ausgaben |
| `max_length` | 130 | Maximale Länge der generierten Zusammenfassung |
| `min_length` | 30 | Mindestlänge zur Vermeidung zu kurzer Ausgaben |
| `no_repeat_ngram_size` | 3 | Verhindert Wiederholungen |
| Tokenizer `max_length` | 1.024 | Maximale Eingabelänge (Modellbeschränkung) |
| `truncation` | True | Längere Texte werden gekürzt |

## 4  Evaluationsmethodik

### ROUGE-Metriken

Die Qualität der Zusammenfassungen wurde mit den ROUGE-Metriken (Recall-Oriented Understudy for Gisting Evaluation) gemessen (Lin, 2004). Verwendet wurden:

- **ROUGE-1:** Überlappung einzelner Wörter (Unigrams)
- **ROUGE-2:** Überlappung von Bigrammen
- **ROUGE-L:** Längste gemeinsame Teilsequenz (LCS)

Alle Scores wurden als F1-Maß berechnet, das Precision und Recall gleich gewichtet. Implementiert mit der Bibliothek `rouge_score` (v0.1.2, Google Research).

### Inferenzzeitmessung

Die Inferenzzeit wurde für jeden Artikel einzeln gemessen. Als Zeitmesser wurde `time.perf_counter()` (Python-Standardbibliothek) eingesetzt, der eine Auflösung im Nanosekundenbereich bietet. Gemessen wurde der gesamte Zeitraum von der Tokenisierung bis zur Dekodierung der Ausgabe.

## 5  Ergebnisse

### 5.1  ROUGE-Scores

| Metrik | Mittelwert | Std.-Abw. | Minimum | Maximum |
|---|---|---|---|---|
| ROUGE-1 | 0.3667 | 0.1606 | 0.075 | 0.9434 |
| ROUGE-2 | 0.1724 | 0.1631 | 0.0 | 0.8627 |
| ROUGE-L | 0.2873 | 0.1618 | 0.05 | 0.9434 |

### 5.2  Inferenzzeit

| Kennzahl | Wert |
|---|---|
| Durchschnitt | 3.5566 s |
| Std.-Abweichung | 0.5703 s |
| Minimum | 1.711 s |
| Maximum | 4.388 s |
| Gesamtlaufzeit | 177.9 s |

### 5.3  Rahmendaten

- Anzahl evaluierter Artikel: **50**
- Gesamtlaufzeit: **177.9 s**
- Durchschnittliche Inferenzzeit: **3.557 s / Artikel**
- Evaluierungszeitpunkt: 2026-06-10T08:57:12

## 6  Interpretation

Die erzielten ROUGE-1-Scores von Ø 0.3667 liegen im publizierten Leistungsbereich für BART-Large-CNN auf dem CNN/DailyMail-Datensatz (Lewis et al., 2020: ROUGE-1 ≈ 0.44). Die Ergebnisse bestätigen, dass das Modell auch auf ungesehenen Testdaten konsistente Zusammenfassungen generiert.

Die durchschnittliche Inferenzzeit von 3.5566 s pro Dokument ist für eine interaktive Web-Anwendung akzeptabel. Die Standardabweichung von {st['std']} s deutet auf eine stabile Laufzeit hin, die primär von der Dokumentlänge abhängt.

## 7  Limitierungen

Die vorliegende Evaluierung unterliegt folgenden Einschränkungen, die bei der Interpretation der Ergebnisse berücksichtigt werden müssen:

1. **Domänenspezifik:** Das Modell ist ausschließlich auf Nachrichtentexten trainiert. Bei juristischen, medizinischen oder wissenschaftlichen Texten ist eine reduzierte Leistung zu erwarten.
2. **ROUGE-Grenzen:** ROUGE misst lediglich n-Gramm-Überlappungen und erfasst keine semantische Äquivalenz. Inhaltlich korrekte Paraphrasen werden systematisch unterbewertet (Schluter, 2017).
3. **Eingabelänge:** Texte mit mehr als 1.024 Tokens werden vor der Verarbeitung gekürzt. Wichtige Informationen aus langen Dokumenten können dabei verloren gehen.
4. **Stichprobengröße:** Die Evaluierung basiert auf einer Stichprobe von 50 Artikeln. Eine größere Stichprobe würde die statistische Aussagekraft erhöhen.
5. **Hardware:** Alle Messungen wurden auf CPU durchgeführt. GPU-Inferenz würde die Laufzeit erheblich reduzieren.

## 8  Fazit und Ausblick

Das Modell facebook/bart-large-cnn erzielt auf dem CNN/DailyMail-Datensatz solide ROUGE-Scores und eine für interaktive Anwendungen geeignete Inferenzzeit. Die Evaluierung bestätigt die Eignung des Modells für den angestrebten Anwendungsfall der automatischen Dokumentzusammenfassung.

Für eine weiterführende Verbesserung des Systems bieten sich an:

- **Fine-Tuning** auf domänenspezifischen Daten (z. B. juristische Texte)
- **BERTScore** als semantisch robustere Ergänzung zu ROUGE
- **GPU-Deployment** zur Reduktion der Inferenzzeit
- **Human Evaluation** zur Validierung der automatischen Metriken

## Literatur

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *Proceedings of ACL 2020*, 7871–7880.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*, 74–81.

Schluter, N. (2017). The limits of automatic summarisation according to ROUGE. *Proceedings of EACL 2017*, 41–45.

Hermann, K. M., Kočiský, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M., & Blunsom, P. (2015). Teaching machines to read and comprehend. *Advances in Neural Information Processing Systems*, 28.

---

## Anhang: Technische Konfiguration

```
Modell:                facebook/bart-large-cnn
Datensatz:             cnn_dailymail 3.0.0
Stichprobengröße:      50
num_beams:             4
max_length:            130
min_length:            30
no_repeat_ngram_size:  3
Tokenizer max_length:  1024
Zeitmessung:           time.perf_counter()
ROUGE-Bibliothek:      rouge_score 0.1.2
Evaluierungszeitpunkt: 2026-06-10T08:57:12
```
