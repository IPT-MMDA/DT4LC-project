"""Class amounts, transition matrices, Markov demand, allocation and the sentences that describe them.

Pure functions on class-index rasters (0 = no data, i = ``names[i - 1]``; ``names`` defaults to the Esri legend,
:data:`.features.ESRI_CLASSES`). The allocation is the
one of the DT4LC horizon backtest (lulc-minimal ``analyses/2026-10-28_dt-forecast-horizons``, ``f2_evaluate.py``):
per from-class, (cell, target) pairs by score, descending, while the target's quota is open.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import features

ESRI = features.ESRI_CLASSES
K = len(ESRI)
CELL_KM2 = features.RES * features.RES / 1e6  # km² per 30 m cell (0.0009)
MIN_SHARE = 0.01  # a class below 1 % of the area gets no sentence
MIN_NET_AREA = 0.01  # a net change below 1 % of the area ...
MIN_NET_CLASS = 0.05  # ... and below 5 % of the class is not worth a sentence
MAX_SENTENCES = 4


def counts(cls: np.ndarray, k: int = K) -> np.ndarray:
    """``(k,)`` cells per class (no data ignored)."""
    return np.bincount(cls[cls > 0].ravel(), minlength=k + 1)[1 : k + 1]


def matrix(a: np.ndarray, b: np.ndarray, k: int = K) -> np.ndarray:
    """``(k, k)`` cells per (class at the first date, class at the second), cells with data at both dates."""
    ok = (a > 0) & (b > 0)
    m = np.bincount((a[ok].astype(np.int64) - 1) * k + (b[ok] - 1), minlength=k * k)
    return m.reshape(k, k)


def majority(stack: np.ndarray, k: int = K) -> np.ndarray:
    """Per cell, the most frequent class of ``stack`` (years, ...); ties go to the latest year's class."""
    cnt = np.stack([(stack == c).sum(0) for c in range(1, k + 1)])
    best = cnt.max(0)
    out = np.zeros(stack.shape[1:], np.uint8)
    for y in range(stack.shape[0]):  # later years overwrite: ties go to the latest
        c = stack[y]
        own = np.take_along_axis(cnt, np.maximum(c.astype(np.int64) - 1, 0)[None], 0)[0]
        win = (c > 0) & (own == best)
        out[win] = c[win]
    return out


def rates(m: np.ndarray) -> np.ndarray:
    """Row-normalised transition matrix (a row without cells stays all zero)."""
    return m / np.maximum(m.sum(1, keepdims=True), 1)


def largest_remainder(x: np.ndarray, total: int) -> np.ndarray:
    n = np.floor(x).astype(int)
    rest = total - n.sum()
    if rest > 0:
        n[np.argsort(-(x - n), kind="stable")[:rest]] += 1
    return n


def demand(start: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """``(k, k)`` cells moving a -> b: each class's cells in ``start`` split by its row of ``rate`` (k x k).

    A class whose row is empty (absent from the trend pair) keeps all its cells.
    """
    k = rate.shape[0]
    n = counts(start, k)
    q = np.zeros((k, k), int)
    for a in range(k):
        if n[a] == 0:
            continue
        if rate[a].sum() > 0:
            q[a] = largest_remainder(n[a] * rate[a] / rate[a].sum(), int(n[a]))
        else:
            q[a, a] = n[a]
    return q


def allocate(start: np.ndarray, quota: np.ndarray, score: np.ndarray, seed: int = 0) -> np.ndarray:
    """The class map after the change ``quota`` (k x k cells) placed by ``score`` (cells, k): highest first.

    ``start`` is a flat class-index array (0 = no data, left as is). Ties in score are broken by a seeded draw.
    """
    k = quota.shape[0]
    rng = np.random.default_rng(seed)
    out = start.copy()
    for a in range(1, k + 1):
        idx = np.flatnonzero(start == a)
        q = quota[a - 1].copy()
        q[a - 1] = 0
        left = int(q.sum())
        if idx.size == 0 or left == 0:
            continue
        b = np.tile(np.arange(1, k + 1), idx.size)
        cell = np.repeat(idx, k)
        sc = np.nan_to_num(score[idx].ravel().astype(np.float64), nan=-1.0)
        keep = b != a
        b, cell, sc = b[keep], cell[keep], sc[keep]
        order = np.lexsort((rng.random(sc.size), -sc))
        # A target meets at most ``left`` of its pairs before its quota is full: the q[t] it takes and at most
        # left - q[t] whose cell another target took first. Its later pairs, and every pair of a target with no
        # quota, are never taken, so the loop below skips them.
        bo = b[order]
        rank = np.empty(bo.size, np.int64)
        for t in range(1, k + 1):
            at = bo == t
            rank[at] = np.arange(int(at.sum())) if q[t - 1] > 0 else left
        order = order[rank < left]
        taken = np.zeros(start.size, bool)
        for t, c in zip(b[order].tolist(), cell[order].tolist(), strict=True):
            if q[t - 1] > 0 and not taken[c]:
                taken[c] = True
                out[c] = t
                q[t - 1] -= 1
                left -= 1
                if left == 0:
                    break
    return out


def matrix_table(m: np.ndarray, names: Sequence[str] = ESRI) -> dict[str, Any]:
    """A transition matrix for the result: class names, km², and each row's shares in %."""
    names = list(names)
    keep = [i for i in range(len(names)) if m[i].sum() or m[:, i].sum()]
    rows = m[np.ix_(keep, keep)]
    return {
        "classes": [names[i] for i in keep],
        "km2": [[km2(v) for v in r] for r in rows],
        "row_percent": [[round(100.0 * float(v) / max(int(r.sum()), 1), 1) for v in r] for r in rows],
    }


def class_table(before: np.ndarray, after: np.ndarray, names: Sequence[str] = ESRI) -> list[dict[str, Any]]:
    """Per class present at either date: km² before and after, change in km² and in %."""
    out = []
    for i, name in enumerate(names):
        if before[i] == 0 and after[i] == 0:
            continue
        d = int(after[i]) - int(before[i])
        out.append(
            {
                "class": name,
                "before_km2": km2(before[i]),
                "after_km2": km2(after[i]),
                "change_km2": km2(d),
                "change_percent": round(100.0 * d / before[i], 1) if before[i] else None,
            }
        )
    return out


def km2(cells: float) -> float:
    """Area of ``cells`` 30 m cells in km², rounded to whole cells (4 decimals)."""
    return round(float(cells) * CELL_KM2, 4)


def _area(cells: float) -> str:
    return km2_text(float(cells) * CELL_KM2)


def km2_text(value: float) -> str:
    """km² as text: "0.36 km²" under 10, "48.3 km²" under 100, "1 204 km²" above; "<0.01 km²" for a trace."""
    v = abs(value)
    if 0 < v < 0.005:
        return "<0.01 km²"
    # Thresholds after rounding, so 9.996 reads "10.0 km²", not "10.00 km²".
    text = f"{v:.2f}" if v < 9.995 else f"{v:.1f}" if v < 99.95 else f"{v:,.0f}".replace(",", " ")
    return f"{text} km²"


def sentences(m: np.ndarray, verb: str, names: Sequence[str] = ESRI) -> list[str]:
    """Plain sentences for the class changes of transition matrix ``m`` (cells) worth reading.

    ``verb`` is "decreased" style past or "would decrease" style future; the caller states the period or the
    condition once, above the sentences. A sentence reads "<Class> <verb> by M % (net A km²); its largest loss is to
    <Y> (B km²)." B is a gross flow, so it can exceed the net change A when the class also gains elsewhere. A class
    is named when its net change is at least :data:`MIN_NET_AREA` of the area or :data:`MIN_NET_CLASS` of the class
    (and the class covers at least :data:`MIN_SHARE` of the area); if none is, one sentence says so.
    """
    before, after = m.sum(1), m.sum(0)
    total = max(int(before.sum()), 1)
    net = after.astype(int) - before.astype(int)
    out: list[str] = []
    for i in np.argsort(-np.abs(net), kind="stable"):
        if len(out) == MAX_SENTENCES or net[i] == 0:
            break
        if max(before[i], after[i]) < MIN_SHARE * total:
            continue
        if abs(net[i]) < MIN_NET_AREA * total and abs(net[i]) < MIN_NET_CLASS * max(before[i], 1):
            continue
        name = names[i]
        down = net[i] < 0
        flows = m[i].copy() if down else m[:, i].copy()
        flows[i] = 0
        j = int(np.argmax(flows))
        word = verb.format(dir="decrease" if down else "increase", past="decreased" if down else "increased")
        flow = "loss is to" if down else "gain is from"
        main = f"; its largest {flow} {names[j]} ({_area(flows[j])})" if flows[j] > 0 else ""
        if before[i]:
            pct = 100.0 * abs(net[i]) / before[i]
            out.append(f"{name} {word} by {pct:.0f} % (net {_area(abs(net[i]))}){main}.")
        else:
            out.append(f"{name} {word} from none to {_area(net[i])}{main}.")
    if not out:
        area, own = 100 * MIN_NET_AREA, 100 * MIN_NET_CLASS
        out.append(f"No class changes by more than {area:.0f} % of the area or {own:.0f} % of itself.")
    return out


def top_transitions(m: np.ndarray, n: int = 5, names: Sequence[str] = ESRI) -> list[dict[str, Any]]:
    """The ``n`` largest class-to-class flows of ``m`` (cells): from, to, km² and the share of the from-class."""
    off = m.astype(float).copy()
    np.fill_diagonal(off, 0)
    out = []
    for k in np.argsort(-off, axis=None, kind="stable")[:n]:
        a, b = divmod(int(k), len(names))
        if off[a, b] <= 0:
            break
        out.append(
            {
                "from": names[a],
                "to": names[b],
                "km2": km2(off[a, b]),
                "percent_of_from": round(100.0 * off[a, b] / max(int(m[a].sum()), 1), 1),
            }
        )
    return out


def changed_km2(m: np.ndarray) -> float:
    """Area of the cells whose class differs between the two dates of ``m``."""
    return km2(m.sum() - np.trace(m))
