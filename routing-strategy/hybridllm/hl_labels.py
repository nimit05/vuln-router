#!/usr/bin/env python3
"""Hybrid LLM's two label rules, kept byte-identical to upstream.

Anchor: Ding, Mallick, Wang, Sim, Mukherjee, Ruhle, Lakshmanan & Awadallah,
*Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*, **ICLR 2024**.
Code: `third_party/HybridLLM` at UPSTREAM_COMMIT.txt.

The router head is a 2-class classifier over (route to SMALL, route to LARGE).
Everything that makes the method the method lives in how the target for that
head is built, so those two expressions are copied rather than paraphrased:

* deterministic (`det_2cls`), `hybrid_llm/pair_ranker/ranker.py:122-127`

      true_labels = [0 if _[0] >= _[1] - self.t else 1 for _ in scores...]

  class 0 means "the small model is good enough", and `t` is the quality-gap
  tolerance the paper trades quality for cost with.

* probabilistic (`prob_2cls`), `ranker.py:79-96` and `:128-134`

      match_prob = self._match_prob(s[0], s[1], t=self.t)
      target     = [p, 1 - p]

  `_match_prob` is the fraction of (small sample, large sample) pairs in which
  the small model is within `t` of the large one. With one sample each it
  collapses to the deterministic rule, which is why the sampled quality in
  h1 matters: it is what separates the two variants.

Upstream computes both against a `scores` tensor of shape
(batch, 2, n_samples, 1). Here the same numbers arrive as plain lists, so the
body of `_match_prob` is reproduced verbatim -- `sum(i >= score_large - t)` is
a numpy count on an array exactly as it is a torch count on a tensor.
"""
from __future__ import annotations
import numpy as np


def match_prob(score_small, score_large, t: float = 0.0) -> float:
    """Verbatim from ranker.py:79-96, with the tensors as numpy arrays.

    Fraction of sample pairs in which the small model is within `t` of the
    large one. Upstream's `prob_2cls` target is `[p, 1 - p]`.
    """
    score_small = np.asarray(score_small, dtype=float)
    score_large = np.asarray(score_large, dtype=float)
    len_s = len(score_small)
    len_l = len(score_large)

    return sum([sum(i >= score_large - t) / len_l for i in score_small]) / len_s


def det_label(score_small, score_large, t: float = 0.0) -> int:
    """Verbatim from ranker.py:122-127. 0 = route small, 1 = route large.

    Upstream indexes one scalar per model; with sampled quality we take the
    mean, which is the same number whenever n_samples == 1.
    """
    s = float(np.mean(score_small))
    l = float(np.mean(score_large))
    return 0 if s >= l - t else 1


def _selfcheck() -> None:
    """The two rules must agree when there is one sample each -- that identity
    is the reason `prob_2cls` is a relaxation of `det_2cls` and not a different
    method. Cheap enough to assert on import of the CLI."""
    for s, l, t in [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                    (0.4, 0.9, 0.5), (0.4, 0.9, 0.4), (0.3, 0.9, 0.5)]:
        p = match_prob([s], [l], t)
        d = det_label([s], [l], t)
        assert (p == 1.0) == (d == 0), (s, l, t, p, d)


if __name__ == "__main__":
    _selfcheck()
    print("hl_labels: det_2cls and prob_2cls agree at n_samples=1")
