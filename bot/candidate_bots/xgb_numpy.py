"""
Run a saved XGBoost model (save_model .json) with plain numpy - the bot needs no xgboost library on the server.
All trees are evaluated at once for a batch of rows (~1-2 ms for 60 candidates).
Supports binary:logistic (returns probability) and reg:squarederror (returns value).
"""
import json
import math

import numpy as np


class Forest:
    def __init__(self, path):
        m = json.load(open(path))
        lr = m["learner"]
        obj = lr["objective"]["name"]
        bs = lr["learner_model_param"]["base_score"].strip("[]")
        base = float(bs.split(",")[0])
        self.logistic = obj == "binary:logistic"
        if self.logistic:
            base = min(max(base, 1e-6), 1 - 1e-6)
        self.base = math.log(base / (1 - base)) if self.logistic else base
        trees = lr["gradient_booster"]["model"]["trees"]
        best = lr.get("attributes", {}).get("best_iteration")
        if best is not None:
            trees = trees[:int(best) + 1]                  # same trees as sklearn predict() after early stopping
        n = max(len(t["left_children"]) for t in trees)
        T = len(trees)
        self.left = np.full((T, n), -1, np.int32)
        self.right = np.full((T, n), -1, np.int32)
        self.feat = np.zeros((T, n), np.int32)
        self.cond = np.zeros((T, n), np.float32)
        self.dleft = np.zeros((T, n), bool)
        depth = 0
        for k, t in enumerate(trees):
            m_ = len(t["left_children"])
            self.left[k, :m_] = t["left_children"]
            self.right[k, :m_] = t["right_children"]
            self.feat[k, :m_] = t["split_indices"]
            self.cond[k, :m_] = t["split_conditions"]
            self.dleft[k, :m_] = np.asarray(t["default_left"], bool)
            depth = max(depth, int(t["tree_param"].get("max_depth", 0)) if "max_depth" in t["tree_param"] else 0)
        self.leaf = self.left < 0
        self.steps = self._max_depth()
        self.T = T

    def _max_depth(self):
        d = 0
        for k in range(self.left.shape[0]):
            stack = [(0, 0)]
            while stack:
                i, h = stack.pop()
                if self.left[k, i] < 0:
                    d = max(d, h)
                else:
                    stack += [(self.left[k, i], h + 1), (self.right[k, i], h + 1)]
        return d

    def predict(self, X):
        X = np.asarray(X, np.float32)
        B = X.shape[0]
        node = np.zeros((B, self.T), np.int32)
        ti = np.arange(self.T)[None, :]
        rows = np.arange(B)[:, None]
        for _ in range(self.steps):
            f = self.feat[ti, node]
            x = X[rows, f]
            go_left = np.where(np.isnan(x), self.dleft[ti, node], x < self.cond[ti, node])
            nxt = np.where(go_left, self.left[ti, node], self.right[ti, node])
            node = np.where(self.leaf[ti, node], node, nxt)
        margin = self.cond[ti, node].sum(1) + self.base
        return 1 / (1 + np.exp(-margin)) if self.logistic else margin
