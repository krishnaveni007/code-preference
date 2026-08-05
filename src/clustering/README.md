# clustering

Persona discovery: clusters users on their 14-axis preference profile and tests how stable/valid
those clusters are.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `cluster_user_profiles.py` | k-means clusters users on their chat preference profile. Supports `--center`, `--k`, `--row-zscore` flags, and was run multiple times with different flags/inputs | session + turn vectors CSVs (or `outputs/high_density/*.csv`) | `outputs/cluster_out/`, `cluster_out_centered/`, `cluster_k10/`, `cluster_full100_zscore/`, `cluster_highdensity/`, `cluster_highdensity_zscore/` |
| `cluster_decode.py` | Same clustering approach applied to the DECODE dataset | `data/scored_pairs.jsonl` | `outputs/cluster_scored_pairs_out/` |
| `cluster_stability_analysis.py` | Subsample stability of k-means centroids | `outputs/chat_vectors/chat_session_vectors_full100.csv` | `outputs/stability_out/` |
| `cluster_prediction_strength.py` | Train/test prediction-strength test of clustering generalization | session-vectors CSVs | `outputs/prediction_strength_out/` |
| `cluster_external_features.py` | Associates clusters with external features (language, repo domain, agent) | `data/swechat_data/`, `outputs/cluster_k10/user_clusters.csv` | `outputs/cluster_k10/features/user_features.csv` |
| `compare_clusters.py` | Cross-dataset (SWE-chat vs DECODE) persona comparison | `outputs/cluster_highdensity_zscore/*`, `outputs/cluster_scored_pairs_out/*` | `outputs/cross_dataset_compare/` |
| `build_profiles.py` | Firing-rate distribution + user-profile heatmap (100 users × 14 axes) | `outputs/chat_vectors/chat_turn_vectors_full100.csv`, `chat_session_vectors_full100.csv` | `outputs/profile_out/` |

Run from the repository root, e.g.:

```bash
python src/clustering/cluster_user_profiles.py --out-dir outputs/cluster_out
```
