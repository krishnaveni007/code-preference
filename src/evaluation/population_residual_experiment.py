#!/usr/bin/env python3
"""Test whether population-surprising preferences carry user-specific signal.

The experiment uses an early-session history and later-session holdout. Population
priors are leave-one-user-out, and zero rubric scores are treated as no evidence.
"""

import argparse
import os

import numpy as np
import pandas as pd


RUBRIC_IDS = [f"R{i:02d}" for i in range(1, 15)]
EPS = 1e-9


def chronological_split(
    turns: pd.DataFrame, train_fraction: float, split_mode: str
) -> pd.DataFrame:
    parts = []
    for user_id, group in turns.groupby("user_id"):
        sessions = (
            group[["session_id", "created_at"]]
            .drop_duplicates()
            .sort_values(["created_at", "session_id"])
        )
        if split_mode == "session_count":
            n_train = int(np.floor(len(sessions) * train_fraction))
            n_train = min(max(n_train, 1), len(sessions) - 1)
            train_ids = set(sessions.iloc[:n_train]["session_id"])
        else:
            start = sessions.created_at.min()
            end = sessions.created_at.max()
            cutoff = start + train_fraction * (end - start)
            train_ids = set(sessions.loc[sessions.created_at <= cutoff, "session_id"])
            # A user needs observations on both sides. When all timestamps are
            # identical, retain the count split as a deterministic fallback.
            if not train_ids or len(train_ids) == len(sessions):
                n_train = int(np.floor(len(sessions) * train_fraction))
                n_train = min(max(n_train, 1), len(sessions) - 1)
                train_ids = set(sessions.iloc[:n_train]["session_id"])
        piece = group.copy()
        piece["split"] = np.where(piece["session_id"].isin(train_ids), "train", "test")
        parts.append(piece)
    return pd.concat(parts, ignore_index=True)


def counts_by_user_axis(active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = (active.set_index("user_id")[[f"score_{r}" for r in RUBRIC_IDS]] > 0).groupby(level=0).sum()
    total = active.set_index("user_id")[[f"score_{r}" for r in RUBRIC_IDS]].ne(0).groupby(level=0).sum()
    positive.columns = RUBRIC_IDS
    total.columns = RUBRIC_IDS
    return positive.astype(float), total.astype(float)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def bootstrap_user_metric(frame: pd.DataFrame, value: str, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    users = frame["user_id"].unique()
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(users, size=len(users), replace=True)
        sampled = pd.concat([frame[frame.user_id == u] for u in chosen], ignore_index=True)
        vals.append(sampled[value].mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-vectors", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument(
        "--split-mode", choices=["session_count", "elapsed_time"],
        default="session_count",
    )
    ap.add_argument("--prior-strength", type=float, default=10.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    turns = pd.read_csv(args.turn_vectors)
    sessions = pd.read_parquet(
        os.path.join(args.data_dir, "sessions.parquet"),
        columns=["session_id", "created_at"],
    )
    turns = turns.merge(sessions, on="session_id", how="left")
    turns["created_at"] = pd.to_datetime(turns["created_at"], utc=True)
    turns = turns.dropna(subset=["created_at"])
    turns = chronological_split(turns, args.train_fraction, args.split_mode)
    score_cols = [f"score_{r}" for r in RUBRIC_IDS]
    active_train = turns[turns.split.eq("train") & turns[score_cols].ne(0).any(axis=1)]
    active_test = turns[turns.split.eq("test") & turns[score_cols].ne(0).any(axis=1)]

    train_pos, train_n = counts_by_user_axis(active_train)
    users = sorted(set(train_pos.index) & set(active_test.user_id.unique()))
    train_pos = train_pos.reindex(users, fill_value=0)
    train_n = train_n.reindex(users, fill_value=0)
    global_pos = train_pos.sum(axis=0)
    global_n = train_n.sum(axis=0)

    # Future prediction: each evaluated observation gets a LOO population prior
    # and a candidate user's shrunk posterior.
    prediction_rows = []
    observation_rows = []
    for user in users:
        loo_pos = global_pos - train_pos.loc[user]
        loo_n = global_n - train_n.loc[user]
        pop_p = (loo_pos + 0.5) / (loo_n + 1.0)
        alpha = args.prior_strength * pop_p + train_pos.loc[user]
        beta = args.prior_strength * (1 - pop_p) + train_n.loc[user] - train_pos.loc[user]
        user_p = alpha / (alpha + beta)
        user_test = active_test[active_test.user_id.eq(user)]
        for rid in RUBRIC_IDS:
            values = user_test.loc[user_test[f"score_{rid}"].ne(0), f"score_{rid}"]
            for value in values:
                y = int(value > 0)
                pp = float(pop_p[rid])
                up = float(user_p[rid])
                support = int(train_n.loc[user, rid])
                positives = int(train_pos.loc[user, rid])
                negatives = support - positives
                consistency = max(positives, negatives) / support if support else 0.0
                directional_margin = abs(positives - negatives) / support if support else 0.0
                observation_rows.append({
                    "user_id": user, "axis": rid, "y": y,
                    "population_p": pp, "user_p": up,
                    "train_support": support,
                    "train_positive": positives,
                    "train_negative": negatives,
                    "train_consistency": consistency,
                    "train_directional_margin": directional_margin,
                    "population_log_loss": log_loss(np.array([y]), np.array([pp])),
                    "user_log_loss": log_loss(np.array([y]), np.array([up])),
                    "minority": bool((y == 1 and pp < 0.5) or (y == 0 and pp >= 0.5)),
                    "surprisal": float(-np.log(pp if y else 1 - pp)),
                })
        obs = pd.DataFrame([r for r in observation_rows if r["user_id"] == user])
        if len(obs):
            prediction_rows.append({
                "user_id": user, "n_test_observations": len(obs),
                "population_log_loss": obs.population_log_loss.mean(),
                "user_log_loss": obs.user_log_loss.mean(),
                "log_loss_improvement": obs.population_log_loss.mean() - obs.user_log_loss.mean(),
            })

    observations = pd.DataFrame(observation_rows)
    predictions = pd.DataFrame(prediction_rows)

    # Retrieval: score each true user's held-out evidence against every candidate
    # historical profile. LOO population priors are defined relative to the true user.
    retrieval_rows = []
    modes = ["all", "majority", "minority", "surprisal_weighted"]
    for true_user in users:
        obs = observations[observations.user_id.eq(true_user)]
        candidate_p = pd.DataFrame(index=users, columns=RUBRIC_IDS, dtype=float)
        for candidate in users:
            # Every stored profile uses a population prior that excludes that
            # candidate, independent of which held-out user is being queried.
            candidate_loo_pos = global_pos - train_pos.loc[candidate]
            candidate_loo_n = global_n - train_n.loc[candidate]
            candidate_pop_p = (candidate_loo_pos + 0.5) / (candidate_loo_n + 1.0)
            candidate_p.loc[candidate] = (
                args.prior_strength * candidate_pop_p + train_pos.loc[candidate]
            ) / (args.prior_strength + train_n.loc[candidate])
        for mode in modes:
            selected = obs
            weights = np.ones(len(selected))
            if mode == "majority":
                selected = obs[~obs.minority]
                weights = np.ones(len(selected))
            elif mode == "minority":
                selected = obs[obs.minority]
                weights = np.ones(len(selected))
            elif mode == "surprisal_weighted":
                weights = selected.surprisal.to_numpy()
            if selected.empty:
                continue
            scores = {}
            for candidate in users:
                probs = np.array([candidate_p.loc[candidate, rid] for rid in selected.axis])
                ys = selected.y.to_numpy()
                ll = ys * np.log(np.clip(probs, EPS, 1-EPS)) + (1-ys) * np.log(np.clip(1-probs, EPS, 1-EPS))
                scores[candidate] = float(np.average(ll, weights=weights))
            ordered = sorted(scores, key=scores.get, reverse=True)
            rank = ordered.index(true_user) + 1
            retrieval_rows.append({
                "user_id": true_user, "mode": mode, "n_observations": len(selected),
                "rank": rank, "top1": rank == 1, "top5": rank <= 5,
                "reciprocal_rank": 1 / rank,
            })

    retrieval = pd.DataFrame(retrieval_rows)
    pred_summary = pd.DataFrame([{
        "n_users": len(predictions),
        "n_test_observations": len(observations),
        "population_log_loss": observations.population_log_loss.mean(),
        "hierarchical_user_log_loss": observations.user_log_loss.mean(),
        "observation_weighted_improvement": observations.population_log_loss.mean() - observations.user_log_loss.mean(),
        "mean_per_user_improvement": predictions.log_loss_improvement.mean(),
        "users_improved": int((predictions.log_loss_improvement > 0).sum()),
    }])
    lo, hi = bootstrap_user_metric(predictions, "log_loss_improvement", rng, args.n_boot)
    pred_summary["mean_per_user_improvement_ci95_low"] = lo
    pred_summary["mean_per_user_improvement_ci95_high"] = hi

    retrieval_summary = retrieval.groupby("mode").agg(
        n_users=("user_id", "nunique"),
        mean_observations=("n_observations", "mean"),
        top1_accuracy=("top1", "mean"),
        top5_accuracy=("top5", "mean"),
        mean_reciprocal_rank=("reciprocal_rank", "mean"),
        median_rank=("rank", "median"),
    ).reset_index()

    # Exploratory trust-gating sweep. These thresholds are evaluated on the
    # holdout and therefore describe the curve; choosing one for deployment
    # requires an inner validation period.
    trust_rows = []

    def add_trust_result(method: str, mask: np.ndarray, predicted: np.ndarray) -> None:
        evaluated = observations.copy()
        evaluated["method_loss"] = [
            log_loss(np.array([y]), np.array([p]))
            for y, p in zip(evaluated.y.to_numpy(), predicted)
        ]
        evaluated["improvement"] = evaluated.population_log_loss - evaluated.method_loss
        per_user_gain = evaluated.groupby("user_id").improvement.mean()
        per_user_personalized = pd.Series(mask, index=evaluated.index).groupby(evaluated.user_id).any()
        boot_means = []
        gain_values = per_user_gain.to_numpy()
        for _ in range(args.n_boot):
            boot_means.append(float(rng.choice(gain_values, size=len(gain_values), replace=True).mean()))
        trust_rows.append({
            "method": method,
            "n_observations": len(evaluated),
            "personalized_observations": int(mask.sum()),
            "personalized_coverage": float(mask.mean()),
            "log_loss": float(evaluated.method_loss.mean()),
            "improvement_over_global": float(evaluated.improvement.mean()),
            "mean_per_user_improvement": float(per_user_gain.mean()),
            "mean_per_user_improvement_ci95_low": float(np.percentile(boot_means, 2.5)),
            "mean_per_user_improvement_ci95_high": float(np.percentile(boot_means, 97.5)),
            "users_personalized": int(per_user_personalized.sum()),
            "users_improved": int((per_user_gain > 0).sum()),
        })

    pop_array = observations.population_p.to_numpy()
    user_array = observations.user_p.to_numpy()
    all_mask = np.ones(len(observations), dtype=bool)
    add_trust_result("always_personalize", all_mask, user_array)
    for minimum_support in [1, 2, 3, 5, 10, 20]:
        mask = observations.train_support.to_numpy() >= minimum_support
        add_trust_result(
            f"frequency_n>={minimum_support}", mask,
            np.where(mask, user_array, pop_array),
        )
    for minimum_support in [2, 3, 5, 10]:
        for minimum_consistency in [0.6, 0.7, 0.8, 0.9]:
            mask = (
                (observations.train_support.to_numpy() >= minimum_support)
                & (observations.train_consistency.to_numpy() >= minimum_consistency)
            )
            add_trust_result(
                f"n>={minimum_support},consistency>={minimum_consistency:.1f}", mask,
                np.where(mask, user_array, pop_array),
            )
    for k in [1, 2, 3, 5, 10, 20]:
        support = observations.train_support.to_numpy(dtype=float)
        trust = (support / (support + k)) * observations.train_directional_margin.to_numpy()
        predicted = (1 - trust) * pop_array + trust * user_array
        add_trust_result(f"continuous_k={k}", trust > 0, predicted)
    trust_summary = pd.DataFrame(trust_rows).sort_values(
        "improvement_over_global", ascending=False
    )

    observations.to_csv(os.path.join(args.out_dir, "future_observations.csv"), index=False)
    predictions.to_csv(os.path.join(args.out_dir, "future_prediction_per_user.csv"), index=False)
    pred_summary.to_csv(os.path.join(args.out_dir, "future_prediction_summary.csv"), index=False)
    retrieval.to_csv(os.path.join(args.out_dir, "user_retrieval_per_user.csv"), index=False)
    retrieval_summary.to_csv(os.path.join(args.out_dir, "user_retrieval_summary.csv"), index=False)
    trust_summary.to_csv(os.path.join(args.out_dir, "trust_gate_summary.csv"), index=False)
    print("Future prediction")
    print(pred_summary.to_string(index=False))
    print("\nUser retrieval")
    print(retrieval_summary.to_string(index=False))
    print("\nTrust-gating sweep (sorted by held-out improvement; exploratory)")
    print(trust_summary.to_string(index=False))


if __name__ == "__main__":
    main()
