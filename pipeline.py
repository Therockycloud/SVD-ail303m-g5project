#!/usr/bin/env python3
"""End-to-end training and evaluation pipeline for MovieLens recommendation."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_recall_at_k(predictions: list[tuple[int, int, float, float]], k: int, threshold: float) -> tuple[float, float]:
    user_est_true: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for uid, _iid, true_r, est in predictions:
        user_est_true[uid].append((est, true_r))

    precisions: list[float] = []
    recalls: list[float] = []
    for uid in user_est_true:
        user_ratings = sorted(user_est_true[uid], key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])
        precisions.append(n_rec_k / k if k > 0 else 0.0)
        recalls.append(n_rec_k / n_rel if n_rel > 0 else 0.0)

    return float(np.mean(precisions)), float(np.mean(recalls))


def ndcg_at_k(predictions: list[tuple[int, int, float, float]], k: int, threshold: float) -> float:
    user_est_true: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for uid, _iid, true_r, est in predictions:
        user_est_true[uid].append((est, true_r))

    ndcgs: list[float] = []
    for uid in user_est_true:
        user_ratings = sorted(user_est_true[uid], key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        dcg = sum((1.0 if true_r >= threshold else 0.0) / np.log2(i + 2) for i, (_, true_r) in enumerate(top_k))

        ideal_ratings = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k]
        idcg = sum((1.0 if true_r >= threshold else 0.0) / np.log2(i + 2) for i, (_, true_r) in enumerate(ideal_ratings))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs))


def mrr_at_k(predictions: list[tuple[int, int, float, float]], k: int, threshold: float) -> float:
    user_est_true: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for uid, _iid, true_r, est in predictions:
        user_est_true[uid].append((est, true_r))

    mrrs: list[float] = []
    for uid in user_est_true:
        top_k = sorted(user_est_true[uid], key=lambda x: x[0], reverse=True)[:k]
        rr = 0.0
        for i, (_est, true_r) in enumerate(top_k):
            if true_r >= threshold:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)

    return float(np.mean(mrrs))


def build_predictions(
    test_df: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    reconstructed: np.ndarray,
    global_mean: float,
) -> list[tuple[int, int, float, float]]:
    preds: list[tuple[int, int, float, float]] = []
    for row in test_df.itertuples(index=False):
        uid = int(row.userId)
        iid = int(row.movieId)
        true_r = float(row.rating)
        if uid in user_to_idx and iid in item_to_idx:
            est = float(reconstructed[user_to_idx[uid], item_to_idx[iid]])
        else:
            est = global_mean
        est = float(np.clip(est, 0.5, 5.0))
        preds.append((uid, iid, true_r, est))
    return preds


def recommend_top_n_for_user(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    reconstructed: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    if user_id not in user_to_idx:
        return pd.DataFrame(columns=["movieId", "title", "predicted_rating"])

    watched = set(ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].astype(int).tolist())
    user_idx = user_to_idx[user_id]
    user_scores = reconstructed[user_idx]

    candidates: list[tuple[int, float]] = []
    for item_idx, score in enumerate(user_scores):
        movie_id = idx_to_item[item_idx]
        if movie_id not in watched:
            candidates.append((movie_id, float(np.clip(score, 0.5, 5.0))))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[:top_n]
    top_df = pd.DataFrame(top, columns=["movieId", "predicted_rating"])
    return top_df.merge(movies_df[["movieId", "title"]], on="movieId", how="left")[["movieId", "title", "predicted_rating"]]


def run_pipeline(args: argparse.Namespace) -> None:
    ratings_path = Path(args.ratings_path)
    movies_path = Path(args.movies_path)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    recs_out = Path(args.recommendations_out) if args.recommendations_out else None

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    if recs_out is not None:
        recs_out.parent.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    train_df, test_df = train_test_split(ratings, test_size=args.test_size, random_state=args.random_state)
    train_df = train_df.copy()
    test_df = test_df.copy()

    global_mean = float(train_df["rating"].mean())

    baseline_pred = np.full(len(test_df), global_mean, dtype=float)
    baseline_rmse = rmse(test_df["rating"].to_numpy(float), baseline_pred)
    baseline_mae = mae(test_df["rating"].to_numpy(float), baseline_pred)

    train_matrix = train_df.pivot_table(index="userId", columns="movieId", values="rating")
    train_matrix_filled = train_matrix.fillna(global_mean)

    user_ids = train_matrix_filled.index.to_numpy(dtype=int)
    item_ids = train_matrix_filled.columns.to_numpy(dtype=int)
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    model = NMF(
        n_components=args.n_components,
        init="nndsvda",
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    W = model.fit_transform(train_matrix_filled.to_numpy())
    H = model.components_
    reconstructed = np.clip(np.dot(W, H), 0.5, 5.0)

    predictions = build_predictions(test_df, user_to_idx, item_to_idx, reconstructed, global_mean)
    y_true = np.array([x[2] for x in predictions], dtype=float)
    y_pred = np.array([x[3] for x in predictions], dtype=float)
    nmf_rmse = rmse(y_true, y_pred)
    nmf_mae = mae(y_true, y_pred)

    precision_k, recall_k = precision_recall_at_k(predictions, k=args.k, threshold=args.threshold)
    f1_k = float(2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k + recall_k) > 0 else 0.0
    ndcg_k = ndcg_at_k(predictions, k=args.k, threshold=args.threshold)
    mrr_k = mrr_at_k(predictions, k=args.k, threshold=args.threshold)

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "ratings_rows": int(len(ratings)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "users_train": int(len(user_ids)),
            "items_train": int(len(item_ids)),
        },
        "params": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_components": args.n_components,
            "max_iter": args.max_iter,
            "k": args.k,
            "threshold": args.threshold,
        },
        "metrics": {
            "baseline": {"rmse": baseline_rmse, "mae": baseline_mae},
            "nmf": {
                "rmse": nmf_rmse,
                "mae": nmf_mae,
                f"precision@{args.k}": precision_k,
                f"recall@{args.k}": recall_k,
                f"f1@{args.k}": f1_k,
                f"ndcg@{args.k}": ndcg_k,
                f"mrr@{args.k}": mrr_k,
            },
            "improvement_percent_vs_baseline": {
                "rmse": (1 - nmf_rmse / baseline_rmse) * 100.0,
                "mae": (1 - nmf_mae / baseline_mae) * 100.0,
            },
        },
    }

    with model_out.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "W": W,
                "H": H,
                "global_mean": global_mean,
                "user_ids": user_ids,
                "item_ids": item_ids,
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
            },
            f,
        )

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 62)
    print("PIPELINE SUMMARY")
    print("=" * 62)
    print(f"Baseline RMSE: {baseline_rmse:.4f} | Baseline MAE: {baseline_mae:.4f}")
    print(f"NMF RMSE:      {nmf_rmse:.4f} | NMF MAE:      {nmf_mae:.4f}")
    print(f"Precision@{args.k}: {precision_k:.4f}")
    print(f"Recall@{args.k}:    {recall_k:.4f}")
    print(f"F1@{args.k}:        {f1_k:.4f}")
    print(f"NDCG@{args.k}:      {ndcg_k:.4f}")
    print(f"MRR@{args.k}:       {mrr_k:.4f}")
    print("-" * 62)
    print(f"Model saved to:   {model_out}")
    print(f"Metrics saved to: {metrics_out}")

    if args.recommend_user is not None:
        top_df = recommend_top_n_for_user(
            user_id=args.recommend_user,
            ratings_df=ratings,
            movies_df=movies,
            user_to_idx=user_to_idx,
            idx_to_item=idx_to_item,
            reconstructed=reconstructed,
            top_n=args.top_n,
        if recs_out is not None:
            top_df.to_csv(recs_out, index=False)
            print(f"Recommendations for user {args.recommend_user} saved to: {recs_out}")
        print("-" * 62)
        print(f"Top-{args.top_n} recommendations for user {args.recommend_user}")
        print(top_df.head(args.top_n).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recommendation pipeline with baseline + NMF.")
    parser.add_argument("--ratings-path", default="data/ml-latest-small/ratings.csv")
    parser.add_argument("--movies-path", default="data/ml-latest-small/movies.csv")
    parser.add_argument("--model-out", default="models/nmf_pipeline.pkl")
    parser.add_argument("--metrics-out", default="reports/pipeline_metrics.json")
    parser.add_argument(
        "--recommendations-out",
        default=None,
        help="Optional output CSV path for demo recommendations (internal/temp).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=3.5)
    parser.add_argument(
        "--recommend-user",
        type=int,
        default=None,
        help="Optional user ID for demo recommendations; skip recommendation export when omitted.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
