# case_study

A full deep dive on one representative user (selected by `src/ingestion/select_case_study_user.py`).

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `case_study_report.py` | Turn-by-turn and cross-session report for the case-study user | `data/swechat_data/`, `outputs/source1_case_study/chat_{turn,session}_vectors.csv` | `outputs/source1_case_study/report/` |
| `case_study_variance.py` | Within-user (turns-in-session) ICC variance decomposition for the case-study user | `outputs/source1_case_study/chat_{turn,session}_vectors.csv` | `outputs/source1_case_study/variance/` |

Run from the repository root, e.g.:

```bash
python src/case_study/case_study_report.py --out-dir outputs/source1_case_study/report
```
