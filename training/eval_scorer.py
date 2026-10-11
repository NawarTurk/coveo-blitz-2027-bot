#!/usr/bin/env python3
"""
Score an EXISTING shot model on a dataset, without retraining (e.g. how good is N14's model on server games?).
Same report as train_scorer.py: AUC vs N7's formula, top 10/20/50% shots, calibration, per mode.
Runs the model with xgb_numpy (exactly what the bot runs on the server).

Usage (repo root):
  python analysis/team_log_to_jsonl.py logs/server_game_logs/*.txt --out logs/server_jsonl
  python training/build_scorer_dataset.py --logs logs/server_jsonl --out data/scorer_server.csv.gz --tag srv_
  python training/eval_scorer.py --data data/scorer_server.csv.gz --models bot/models/scorer
Writes results/scorer_eval_<models folder>.txt
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "bot", "candidate_bots")]
import train_scorer as TS          # report() + say()
import xgb_numpy as XN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "bot", "models", "scorer"))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    names = json.load(open(os.path.join(a.models, "features.json")))["features"]
    df = pd.concat([pd.read_csv(p) for p in a.data], ignore_index=True)
    missing = [f for f in names if f not in df.columns]
    if missing:
        sys.exit(f"dataset has no {missing}: rebuild it with the new build_scorer_dataset.py")
    X = df[names].values.astype(np.float32)
    df["p1"] = XN.Forest(os.path.join(a.models, "k1.json")).predict(X)
    df["p2"] = XN.Forest(os.path.join(a.models, "k2.json")).predict(X)
    df["ppts"] = XN.Forest(os.path.join(a.models, "pts.json")).predict(X)
    df["base"] = 0.7 * df.cnn_score + 0.3 * df.prior
    TS.say(f"model {os.path.normpath(a.models)} ({len(names)} inputs) on {a.data}")
    TS.report(df, "EVAL")
    out = a.report or f"results/scorer_eval_{os.path.basename(os.path.normpath(a.models))}.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(TS.OUT) + "\n")
    print(f"report saved to {out}")


if __name__ == "__main__":
    main()
