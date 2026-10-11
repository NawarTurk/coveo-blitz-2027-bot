#!/usr/bin/env python3
"""
STEP 5-6 of the XGBoost shot scorer: train on ~950 games, test on ~100 games it never saw (split by GAME).

Models (saved to bot/models/scorer/):
  k1.json   P(shot kills 1+)
  k2.json   P(shot kills 2+)
  pts.json  expected points of the shot   <- what the bot would rank candidates by
Metrics on the unseen games (most shots miss, so plain accuracy means nothing):
  AUC                    how well it separates killing shots from misses (0.5 = coin flip, 1 = perfect)
  top 10% / 20% shots    kill rate + points of the shots the model likes most, vs N7's own score
                         (0.7 * CNN4 + 0.3 * trap table) on the same shots
  calibration            predicted P(kill) vs what really happened, in 10 buckets
  per mode               the same comparison inside each board mode
  feature importance     what the model relies on

Usage (repo root, needs: pip install xgboost pandas):
  v1 (N14):  python training/train_scorer.py --data data/scorer_local.csv.gz --features v1 --models bot/models/scorer
  v2 (N15):  python training/train_scorer.py --data data/scorer_local.csv.gz data/scorer_explore.csv.gz data/scorer_server.csv.gz \
                 --server data/scorer_server.csv.gz
             (v2 = 33 + 7 wave inputs, saved to bot/models/scorer_v2; --server also reports server games on their own.
              Server games in --data are split like the rest: the TEST games include some, never trained on.)
Writes the report to results/scorer_report.txt
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bot", "candidate_bots"))
import scorer_features as SF

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def auc(y, p):
    y = np.asarray(y)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def top(df, score, frac):
    k = max(1, int(len(df) * frac))
    t = df.iloc[np.argsort(-df[score].values)[:k]]
    return t.y1.mean(), t.y2.mean(), t.pts.mean()


def report(df, name):
    say(f"\n=== {name}: {len(df)} shots from {df.game.nunique()} games | real kill rate {df.y1.mean():.3f} "
        f"| 2+ rate {df.y2.mean():.4f} | avg points/shot {df.pts.mean():.2f}")
    say(f"AUC  kill 1+: XGB {auc(df.y1, df.p1):.3f} vs N7 score {auc(df.y1, df.base):.3f} | "
        f"kill 2+: XGB {auc(df.y2, df.p2):.3f} vs N7 {auc(df.y2, df.base):.3f}")
    for frac in (0.1, 0.2, 0.5):
        a, b = top(df, "ppts", frac), top(df, "base", frac)
        say(f"top {frac:.0%} shots   XGB: kill {a[0]:.3f} 2+ {a[1]:.4f} pts {a[2]:.2f}   |   "
            f"N7 score: kill {b[0]:.3f} 2+ {b[1]:.4f} pts {b[2]:.2f}")
    say("calibration (predicted P(kill) bucket -> real kill rate, shots):")
    b = pd.cut(df.p1, [0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0], include_lowest=True)
    for k, g in df.groupby(b, observed=True):
        say(f"  {str(k):14} -> {g.y1.mean():.3f}  ({len(g)})")
    say("per mode (kill rate of the top 20% shots: XGB vs N7 score):")
    for m, g in df.groupby("mode"):
        if len(g) < 200:
            continue
        a, c = top(g, "ppts", .2), top(g, "base", .2)
        say(f"  {SF.MODES[int(m)]:9} {len(g):7d} shots | real {g.y1.mean():.3f} | XGB top20% {a[0]:.3f} | N7 top20% {c[0]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/scorer_local.csv.gz"])
    ap.add_argument("--features", choices=["v1", "v2", "v2r"], default="v2",
                    help="v1 = 33 inputs (N14) | v2 = + 7 wave inputs (N15) | v2r = + 3 rule-model inputs (N15b)")
    ap.add_argument("--server", default=None, help="optional server-game dataset for an extra test")
    ap.add_argument("--test-games", type=int, default=100)
    ap.add_argument("--models", default=None, help="default: bot/models/scorer (v1) or bot/models/scorer_v2 (v2)")
    ap.add_argument("--report", default="results/scorer_report.txt")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in a.data], ignore_index=True)
    F = {"v1": SF.FEATURES, "v2": SF.FEATURES + SF.WAVE_FEATURES, "v2r": SF.ALL_FEATURES}[a.features]
    missing = [f for f in F if f not in df.columns]
    if missing:
        sys.exit(f"dataset has no {missing} -> rebuild it with the new build_scorer_dataset.py")
    if a.models is None:
        a.models = os.path.join(HERE, "..", "bot", "models", {"v1": "scorer", "v2": "scorer_v2", "v2r": "scorer_v2r"}[a.features])
    say(f"features {a.features}: {len(F)} inputs | data files {a.data}")
    df["base"] = 0.7 * df.cnn_score + 0.3 * df.prior            # N7's own ranking score
    games = np.array(sorted(df.game.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    test_g = set(games[:a.test_games])
    n_val = max(1, min(50, (len(games) - a.test_games) // 10))
    val_g = set(games[a.test_games:a.test_games + n_val])        # early stopping
    te = df[df.game.isin(test_g)].copy()
    va = df[df.game.isin(val_g)]
    tr = df[~df.game.isin(test_g | val_g)]
    say(f"data {len(df)} shots / {len(games)} games -> train {tr.game.nunique()} games ({len(tr)} shots), "
        f"early-stop {len(val_g)} games, TEST {len(test_g)} unseen games ({len(te)} shots)")

    P = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=5, tree_method="hist", early_stopping_rounds=40, n_jobs=-1)
    models = {
        "k1": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **P),
        "k2": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **P),
        "pts": xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **P),
    }
    target = {"k1": "y1", "k2": "y2", "pts": "pts"}
    for k, m in models.items():
        m.fit(tr[F], tr[target[k]], eval_set=[(va[F], va[target[k]])], verbose=False)
        say(f"trained {k}: {m.best_iteration + 1} trees")

    def predict(d):
        d["p1"] = models["k1"].predict_proba(d[F])[:, 1]
        d["p2"] = models["k2"].predict_proba(d[F])[:, 1]
        d["ppts"] = models["pts"].predict(d[F])
        return d

    report(predict(te), "TEST: unseen local games")
    if a.server:
        sv = pd.read_csv(a.server)
        if set(sv.game) & set(tr.game):
            sv = sv[~sv.game.isin(set(tr.game) | val_g)]          # only server games the model never trained on
        sv["base"] = 0.7 * sv.cnn_score + 0.3 * sv.prior
        if len(sv):
            report(predict(sv), "SERVER games (not trained on)")
        else:
            say("\nno unseen server games left to test (all were in training)")

    imp = pd.Series(models["pts"].get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    say("\nfeature importance (points model, gain): " + ", ".join(f"{k} {v:.0f}" for k, v in imp.head(15).items()))

    os.makedirs(a.models, exist_ok=True)
    for k, m in models.items():
        m.save_model(os.path.join(a.models, f"{k}.json"))
    json.dump({"features": F}, open(os.path.join(a.models, "features.json"), "w"))
    say(f"\nmodels saved to {os.path.normpath(a.models)}")
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    open(a.report, "w").write("\n".join(OUT) + "\n")
    print(f"report saved to {a.report}")


if __name__ == "__main__":
    main()
