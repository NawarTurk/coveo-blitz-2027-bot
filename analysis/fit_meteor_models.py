#!/usr/bin/env python3
"""
Does any heuristic add information BEYOND CNN4?  Fit a small Ridge correction on the launch table.

  target  = actual points of the meteor (and, separately, actual kills)
  base    = CNN4 only:         target ~ a * cnn_score + b * cnn_clear
  full    = CNN4 + heuristics: target ~ cnn_score + cnn_clear + heuristic features   (Ridge, standardized)
Cross-validation by WHOLE GAME (5 folds) so launches from one map never sit in both train and test.

Reports, out-of-fold:
  R2, correlation with the actual outcome, and "top-10%": average actual points of the 10% of launches
  each model rates best (what matters for picking spots)
Then the full-data coefficients in real units (points per +1 of the feature) and how stable their sign is across folds.
Saves the weights for the bot (only if the full model beats CNN4 alone): models/residual_weights.json

Usage:  python fit_meteor_models.py data/launch_table_cnn4.csv
        python fit_meteor_models.py data/launch_table_cnn4.csv --recent 500     (only the newest 500 games)
"""
import argparse, json, os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor

from meteor_features import BASE, HEUR

ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("--recent", type=int, default=0, help="only the newest N games (by file name)")
ap.add_argument("--out", default="models/meteor_models.json")
args = ap.parse_args()

df = pd.read_csv(args.table)
if args.recent:
    keep = sorted(df.game.unique())[-args.recent:]
    df = df[df.game.isin(keep)]
df = df.reset_index(drop=True)
print(f"{len(df):,} launches from {df.game.nunique()} games | avg kills {df.kills.mean():.3f}, "
      f"avg points {df.points.mean():.1f}, any kill {100 * df.any_kill.mean():.1f}%")
print(f"CNN4 calibration: predicted dinos in blast {df.cnn_E.mean():.3f} vs actual kills {df.kills.mean():.3f}")

groups = df.game.values
gkf = GroupKFold(n_splits=5)
alphas = np.logspace(-2, 4, 25)


def top10(score, y):
    n = max(1, len(y) // 10)
    return float(y[np.argsort(-score)[:n]].mean())


def tree_models():
    out = {"RF": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=50, max_features=0.5, n_jobs=-1,
                                               random_state=0)}
    try:
        from xgboost import XGBRegressor
        out["XGB"] = lambda: XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, min_child_weight=20, n_jobs=-1, verbosity=0)
    except Exception:
        print("(xgboost not installed: skipping XGB)")
    return out


TREES = tree_models()


def evaluate(target):
    y = df[target].values.astype(float)
    oof = {k: np.zeros(len(df)) for k in ["heur Ridge"] + [f"heur {t}" for t in TREES] + ["CNN4 only", "CNN4+heur Ridge"]
           + [f"CNN4+heur {t}" for t in TREES]}
    coefs = []
    Xb = df[BASE].values.astype(float)
    Xh = df[HEUR].values.astype(float)
    Xf = df[BASE + HEUR].values.astype(float)
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups)):
        oof["CNN4 only"][te] = LinearRegression().fit(Xb[tr], y[tr]).predict(Xb[te])
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
        mf = RidgeCV(alphas=alphas).fit((Xf[tr] - mu) / sd, y[tr])
        oof["CNN4+heur Ridge"][te] = mf.predict((Xf[te] - mu) / sd)
        coefs.append(mf.coef_ / sd)
        mh_, sh_ = Xh[tr].mean(0), Xh[tr].std(0) + 1e-9
        oof["heur Ridge"][te] = RidgeCV(alphas=alphas).fit((Xh[tr] - mh_) / sh_, y[tr]).predict((Xh[te] - mh_) / sh_)
        for name, mk in TREES.items():
            oof[f"heur {name}"][te] = mk().fit(Xh[tr], y[tr]).predict(Xh[te])
            oof[f"CNN4+heur {name}"][te] = mk().fit(Xf[tr], y[tr]).predict(Xf[te])
        print(f"  [{target}] fold {fold + 1}/5 done", flush=True)
    r2 = lambda p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    cor = lambda p: float(np.corrcoef(p, y)[0, 1])
    print(f"\n=== target: {target} (out-of-fold, by game)")
    print(f"  {'':<22}{'R2':>8}{'corr':>8}{'top-10% actual':>16}")
    for k, p in oof.items():
        print(f"  {k:<22}{r2(p):>8.4f}{cor(p):>8.4f}{top10(p, y):>16.2f}")
    print(f"  {'(perfect hindsight)':<22}{'':>16}{top10(y, y):>16.2f}")
    return np.array(coefs), r2(oof["CNN4 only"]), r2(oof["CNN4+heur Ridge"]), top10(oof["CNN4 only"], y), \
        top10(oof["CNN4+heur Ridge"], y), {k: top10(p, y) for k, p in oof.items()}


res = {t: evaluate(t) for t in ("points", "kills")}

# full-data fit on points -> weights for the bot
y = df.points.values.astype(float)
Xf = df[BASE + HEUR].values.astype(float)
mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
m = RidgeCV(alphas=alphas).fit((Xf - mu) / sd, y)
raw = m.coef_ / sd
coefs_cv = res["points"][0]
stable = (np.sign(coefs_cv) == np.sign(raw)).mean(0)
imp = np.abs(m.coef_)
print(f"\n=== points model, full data (ridge alpha {m.alpha_:.3g})  -- sorted by importance (effect of +1 std)")
print(f"  {'feature':<16}{'points per +1':>14}{'effect of +1 std':>18}{'sign stable':>13}")
for i in np.argsort(-imp):
    print(f"  {(BASE + HEUR)[i]:<16}{raw[i]:>14.3f}{m.coef_[i]:>18.2f}{100 * stable[i]:>12.0f}%")

_, b_r2, f_r2, b_top, f_top, tops = res["points"]
gain = f_top - b_top
print(f"\nVERDICT: top-10% shots {b_top:.1f} -> {f_top:.1f} points ({100 * gain / max(b_top, 1e-9):+.1f}%), "
      f"R2 {b_r2:.4f} -> {f_r2:.4f}")
best_name = max(tops, key=tops.get)
print(f"BEST by top-10% points: {best_name} ({tops[best_name]:.1f})  vs CNN4 only {tops['CNN4 only']:.1f}")
for name, mk in TREES.items():
    mt = mk().fit(df[HEUR].values.astype(float), y)
    fi = mt.feature_importances_
    print(f"\n{name} (heuristics only) feature importance:")
    for i in np.argsort(-fi)[:12]:
        print(f"  {HEUR[i]:<16}{fi[i]:.3f}")
# ---- save both ridge models for bot_znawar1.py (always; the verdict above says whether they help)
def ridge_full(cols):
    X = df[cols].values.astype(float)
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    mm = RidgeCV(alphas=alphas).fit((X - mu_) / sd_, y)
    return {"features": cols, "mean": mu_.tolist(), "std": sd_.tolist(), "coef_std": mm.coef_.tolist(),
            "intercept_std": float(mm.intercept_), "alpha": float(mm.alpha_)}


os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump({"heur": ridge_full(HEUR), "hybrid": ridge_full(BASE + HEUR), "table": args.table, "n": int(len(df)),
           "cv_top10": tops}, open(args.out, "w"), indent=1)
print(f"\nsaved {args.out} (heuristic model + hybrid model, ridge, target = points)")
if f_top > b_top * 1.03 and f_r2 > b_r2:
    print("-> heuristics add information beyond CNN4: zNawar1 should help")
else:
    print("-> heuristics add (almost) nothing beyond CNN4: zNawar1 will mostly reorder candidates (speed), not score better")
