#!/usr/bin/env python3
"""
metrics_utils.py

Metrics helpers for binary classification and regression targets used in
structure-scoring tasks. The class `Metrics` filters non-finite pairs,
computes confusion-matrix–based rates (sensitivity, specificity, etc.),
optionally computes regression scores, and exposes ranking utilities
(hitrate, success@k, AUC). All operations are guarded to avoid raising errors.
"""

import warnings
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score

# Silence known noisy deprecation warnings from dependencies.
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="'dropout_adj' is deprecated")

# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _to_np(x):
    """Convert to NumPy array of dtype float."""
    return np.asarray(x, dtype=float)

def _finite_pairmask(y, pred):
    """Return filtered (y, pred, mask) keeping only finite pairs."""
    y = _to_np(y)
    pred = _to_np(pred)
    mask = np.isfinite(y) & np.isfinite(pred)
    return y[mask], pred[mask], mask

def _safe_div(num, den):
    """Elementwise division with zero-denominator protection."""
    try:
        den = np.asarray(den, dtype=float)
        num = np.asarray(num, dtype=float)
        out = np.divide(num, den, where=(den != 0))
        if np.ndim(out) == 0:
            return float(out)
        return out
    except Exception:
        return None

def get_binary(values, threshold, target):
    """Map continuous or multiclass values to {0,1} based on target convention."""
    inverse = ["fnat", "bin_class", "dockq"]
    arr = _to_np(values)
    if target in inverse:
        values_binary = (arr > threshold).astype(int).tolist()
    else:
        values_binary = (arr < threshold).astype(int).tolist()
    return values_binary

def get_comparison(prediction, ground_truth, binary=True, classes=[0, 1]):
    """Compute FP, FN, TP, TN. Safe even if a class is missing."""
    try:
        CM = confusion_matrix(ground_truth, prediction, labels=classes)
        false_positive = CM.sum(axis=0) - np.diag(CM)
        false_negative = CM.sum(axis=1) - np.diag(CM)
        true_positive = np.diag(CM)
        true_negative = CM.sum() - (false_positive + false_negative + true_positive)
        if binary:
            return (false_positive[1], false_negative[1],
                    true_positive[1], true_negative[1])
        else:
            return false_positive, false_negative, true_positive, true_negative
    except Exception:
        if binary:
            return 0, 0, 0, 0
        else:
            z = np.zeros(len(classes), dtype=int)
            return z, z, z, z

# ---------------------------------------------------------------------
# metrics container
# ---------------------------------------------------------------------
class Metrics(object):
    def __init__(self, prediction, y, target, threshold=0.23, binary=True):
        """
        Metrics container. Never raises; returns None for unavailable metrics.

        Args:
            prediction: predicted scores or labels.
            y: ground-truth scores or labels.
            target (str): one of regression targets {"fnat","irmsd","lrmsd","dockq"}
                          or classification targets {"bin_class","capri_class"}.
            threshold (float): decision threshold for binarization when needed.
            binary (bool): if True, use binary evaluation; otherwise use multiclass
                           for known targets.
        """
        # filter out non-finite pairs up front
        y_f, pred_f, mask = _finite_pairmask(y, prediction)

        self.prediction = pred_f
        self.y = y_f
        self.binary = binary
        self.target = target
        self.threshold = threshold

        # empty after filtering: all metrics become None
        if self.y.size == 0 or self.prediction.size == 0:
            self._init_empty()
            return

        if self.binary:
            prediction_binary = get_binary(self.prediction, self.threshold, self.target)
            y_binary = get_binary(self.y, self.threshold, self.target)
            classes = [0, 1]
            (false_positive, false_negative, true_positive, true_negative) = get_comparison(
                prediction_binary, y_binary, self.binary, classes=classes
            )
        else:
            if self.target == "capri_class":
                classes = [1, 2, 3, 4, 5]
            elif self.target == "bin_class":
                classes = [0, 1]
            else:
                # unknown non-binary target: fall back to binary on threshold
                classes = [0, 1]
                self.binary = True
            (false_positive, false_negative, true_positive, true_negative) = get_comparison(
                self.prediction, self.y, self.binary, classes=classes
            )

        # guarded rates
        self.sensitivity = _safe_div(true_positive, (true_positive + false_negative))
        self.specificity = _safe_div(true_negative, (true_negative + false_positive))
        self.precision   = _safe_div(true_positive, (true_positive + false_positive))
        self.NPV         = _safe_div(true_negative, (true_negative + false_negative))
        self.FPR         = _safe_div(false_positive, (false_positive + true_negative))
        self.FNR         = _safe_div(false_negative, (true_positive + false_negative))
        self.FDR         = _safe_div(false_positive, (true_positive + false_positive))

        denom = (true_positive + false_positive + false_negative + true_negative)
        self.accuracy = _safe_div((true_positive + true_negative), denom)

        # defaults for regression
        self.explained_variance = None
        self.max_error = None
        self.mean_absolute_error = None
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.mean_squared_log_error = None
        self.median_squared_log_error = None
        self.r2_score = None

        # compute regression metrics only for regression targets and safe inputs
        if self.target in ["fnat", "irmsd", "lrmsd", "dockq"]:
            try:
                constant_pred = np.all(self.prediction == self.prediction[0])
                constant_true = np.all(self.y == self.y[0])
                #if not (constant_pred and constant_true):
                self.explained_variance = metrics.explained_variance_score(self.y, self.prediction)
                self.max_error = metrics.max_error(self.y, self.prediction)
                self.mean_absolute_error = metrics.mean_absolute_error(self.y, self.prediction) #mae
                self.mean_squared_error = metrics.mean_squared_error(self.y, self.prediction) #mse
                self.root_mean_squared_error = np.sqrt(self.mean_squared_error)  # rmse

                # MSLE/median abs error left disabled unless needed:
                # self.mean_squared_log_error = metrics.mean_squared_log_error(self.y, self.prediction)
                # self.median_squared_log_error = metrics.median_absolute_error(self.y, self.prediction)
                self.r2_score = metrics.r2_score(self.y, self.prediction)
            except Exception as e:
                # in case of any error, leave regression metrics as None
                print(f"Warning: regression metrics could not be computed: {e}")
                pass

    def _init_empty(self):
        """Initialize all fields for the empty case."""
        self.sensitivity = None
        self.specificity = None
        self.precision = None
        self.NPV = None
        self.FPR = None
        self.FNR = None
        self.FDR = None
        self.accuracy = None
        self.explained_variance = None
        self.max_error = None
        self.mean_absolute_error = None
        self.mean_squared_error = None
        self.root_mean_squared_error = None
        self.mean_squared_log_error = None
        self.median_squared_log_error = None
        self.r2_score = None

    # -----------------------------------------------------------------
    # ranking utilities
    # -----------------------------------------------------------------
    def format_score(self):
        """Return sorted indices and binary ground truth per the target convention."""
        idx = np.argsort(self.prediction)
        inverse = ["fnat", "bin_class", "dockq"]
        if self.target in inverse:
            idx = idx[::-1]
        ground_truth_bool = np.array(get_binary(self.y, self.threshold, self.target))
        return idx, ground_truth_bool

    def hitrate(self):
        """Cumulative hits along the sorted ranking."""
        idx, ground_truth_bool = self.format_score()
        return np.cumsum(ground_truth_bool[idx])

    def hitrate_at_k(self, k=10):
        """Fraction of hits in the top k."""
        idx, ground_truth_bool = self.format_score()
        n = len(ground_truth_bool)
        if n == 0:
            return float("nan")
        k = min(k, n)
        topk_hits = ground_truth_bool[idx][:k]
        return float(np.mean(topk_hits))

    def success_at_k(self, k=10):
        """1 if at least one hit appears in the top k, else 0."""
        idx, ground_truth_bool = self.format_score()
        n = len(ground_truth_bool)
        if n == 0:
            return float("nan")
        k = min(k, n)
        topk_hits = np.asarray(ground_truth_bool)[idx][:k]
        return float(np.any(topk_hits))

    def auc(self):
        """ROC-AUC if both classes are present after filtering; else None."""
        ground_truth_bool = np.array(get_binary(self.y, self.threshold, self.target))
        if ground_truth_bool.size == 0:
            return None
        has_pos = np.any(ground_truth_bool == 1)
        has_neg = np.any(ground_truth_bool == 0)
        if not (has_pos and has_neg):
            return None
        try:
            return roc_auc_score(ground_truth_bool, self.prediction)
        except Exception:
            return None
