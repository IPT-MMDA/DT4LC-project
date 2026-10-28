"""Class amounts, transition matrices, Markov demand, allocation and the sentences that describe them.

Pure functions on Esri class-index rasters (0 = no data, i = ``features.ESRI_CLASSES[i - 1]``). The allocation is the
one of the DT4LC horizon backtest (lulc-minimal ``analyses/2026-10-28_dt-forecast-horizons``, ``f2_evaluate.py``):
per from-class, (cell, target) pairs by score, descending, while the target's quota is open.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import features

K = len(features.ESRI_CLASSES)
CELL_HA = features.RES * features.RES / 1e4  # hectares per 30 m cell
MIN_SHARE = 0.01  # a class below 1 % of the area gets no sentence
MAX_SENTENCES = 4


def counts(cls: np.ndarray) -> np.ndarray:
    """``(K,)`` cells per class (no data ignored)."""
    return np.bincount(cls[cls > 0].ravel(), minlength=K + 1)[1:]


def matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(K, K)`` cells per (class at the first date, class at the second), cells with data at both dates."""
    ok = (a > 0) & (b > 0)
    m = np.bincount((a[ok].astype(np.int64) - 1) * K + (b[ok] - 1), minlength=K * K)
    return m.reshape(K, K)


def majority(stack: np.ndarray) -> np.ndarray:
    """Per cell, the most frequent class of ``stack`` (years, ...); ties go to the latest year's class."""
    cnt = np.stack([(stack == c).sum(0) for c in range(1, K + 1)])
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
    """``(K, K)`` cells moving a -> b: each class's cells in ``start`` split by its row of ``rate``.

    A class whose row is empty (absent from the trend pair) keeps all its cells.
    """
    n = counts(start)
    q = np.zeros((K, K), int)
    for a in range(K):
        if n[a] == 0:
            continue
        if rate[a].sum() > 0:
            q[a] = largest_remainder(n[a] * rate[a] / rate[a].sum(), int(n[a]))
        else:
            q[a, a] = n[a]
    return q


def allocate(start: np.ndarray, quota: np.ndarray, score: np.ndarray, seed: int = 0) -> np.ndarray:
    """The class map after the change ``quota`` (K x K cells) placed by ``score`` (cells, K): highest first.

    ``start`` is a flat class-index array (0 = no data, left as is). Ties in score are broken by a seeded draw.
    """
    rng = np.random.default_rng(seed)
    out = start.copy()
    for a in range(1, K + 1):
        idx = np.flatnonzero(start == a)
        q = quota[a - 1].copy()
        q[a - 1] = 0
        if idx.size == 0 or q.sum() == 0:
            continue
        b = np.tile(np.arange(1, K + 1), idx.size)
        cell = np.repeat(idx, K)
        sc = np.nan_to_num(score[idx].ravel().astype(np.float64), nan=-1.0)
        keep = b != a
        b, cell, sc = b[keep], cell[keep], sc[keep]
        taken = np.zeros(start.size, bool)
        for j in np.lexsort((rng.random(sc.size), -sc)):
            if q.sum() == 0:
                break
            if q[b[j] - 1] > 0 and not taken[cell[j]]:
                taken[cell[j]] = True
                out[cell[j]] = b[j]
                q[b[j] - 1] -= 1
    return out


def matrix_table(m: np.ndarray) -> dict[str, Any]:
    """A transition matrix for the result: class names, hectares, and each row's shares in %."""
    names = list(features.ESRI_CLASSES)
    keep = [i for i in range(K) if m[i].sum() or m[:, i].sum()]
    rows = m[np.ix_(keep, keep)]
    return {
        "classes": [names[i] for i in keep],
        "hectares": [[round(float(v) * CELL_HA, 2) for v in r] for r in rows],
        "row_percent": [[round(100.0 * float(v) / max(int(r.sum()), 1), 1) for v in r] for r in rows],
    }


def class_table(before: np.ndarray, after: np.ndarray) -> list[dict[str, Any]]:
    """Per class present at either date: hectares before and after, change in hectares and in %."""
    out = []
    for i, name in enumerate(features.ESRI_CLASSES):
        if before[i] == 0 and after[i] == 0:
            continue
        d = int(after[i]) - int(before[i])
        out.append(
            {
                "class": name,
                "before_ha": round(float(before[i]) * CELL_HA, 2),
                "after_ha": round(float(after[i]) * CELL_HA, 2),
                "change_ha": round(d * CELL_HA, 2),
                "change_percent": round(100.0 * d / before[i], 1) if before[i] else None,
            }
        )
    return out


def _ha(cells: float) -> str:
    return ha_text(cells * CELL_HA)


def ha_text(ha: float) -> str:
    """Hectares as text: "412 ha", or km² from 1 000 ha ("12.5 km²")."""
    if ha >= 1000:
        return f"{ha / 100:,.1f} km²".replace(",", " ")
    return f"{ha:,.0f} ha".replace(",", " ")


def sentences(m: np.ndarray, lead: str, verb: str) -> list[str]:
    """Plain sentences for the largest net class changes of transition matrix ``m`` (cells).

    ``lead`` opens each sentence ("Between 2021 and 2025,"), ``verb`` is "decreased" style past or "is expected to
    decrease" style future: the sentence reads "<lead> <Class> <verb> by M % (A ha), mainly to <Y> (B ha)."
    """
    before, after = m.sum(1), m.sum(0)
    total = max(int(before.sum()), 1)
    net = after.astype(int) - before.astype(int)
    out = []
    for i in np.argsort(-np.abs(net), kind="stable"):
        if len(out) == MAX_SENTENCES or net[i] == 0:
            break
        if max(before[i], after[i]) < MIN_SHARE * total:
            continue
        name = features.ESRI_CLASSES[i]
        down = net[i] < 0
        flows = m[i].copy() if down else m[:, i].copy()
        flows[i] = 0
        j = int(np.argmax(flows))
        word = verb.format(dir="decrease" if down else "increase", past="decreased" if down else "increased")
        where = "to" if down else "from"
        main = f", mainly {where} {features.ESRI_CLASSES[j]} ({_ha(flows[j])})" if flows[j] > 0 else ""
        if before[i]:
            out.append(f"{lead} {name} {word} by {100.0 * abs(net[i]) / before[i]:.0f} % ({_ha(abs(net[i]))}){main}.")
        else:
            out.append(f"{lead} {name} {word} from none to {_ha(net[i])}{main}.")
    return out
