# swe-chat-stats

A self-contained sub-pipeline that computes dataset-wide descriptive statistics about SWE-chat for
the writeup — separate from the preference-scoring pipeline in the rest of `src/`.

| Script | Purpose | Writes |
|---|---|---|
| `section0_.py` | Dataset overview and annotation-feasibility numbers | `outputs/swe-chat-stats/section0_out/` |
| `section0_c.py` | Subsession reconstruction by timestamp alignment | `outputs/swe-chat-stats/section0c_out/` |
| `section15.py`, `data_distribution.py` | Data-distribution figures/CSVs (near-duplicate scripts — `data_distribution.py` differs only in a binning-label change) | `outputs/swe-chat-stats/section15_out/` |
| `table3.py` | Recomputes/verifies the intro-section table numbers | `outputs/swe-chat-stats/intro_out/` |

All scripts read from `data/swechat_data/`. Run from the repository root, e.g.:

```bash
python src/swe-chat-stats/section0_.py --data-dir data/swechat_data --out-dir outputs/swe-chat-stats/section0_out
```
