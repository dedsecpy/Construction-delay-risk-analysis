"""Exact inference for the factor → severe_delay BN (binary nodes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bn.cpt import bernoulli_table, noisy_or_severe_table
from src.bn.graph import BayesianNetwork, default_states, load_structure


@dataclass
class Posterior:
    p_severe: float
    factor_marginals: dict[str, float]


class DelayRiskBN:
    def __init__(
        self,
        factor_priors: dict[str, float] | None = None,
        p_cause: float = 0.35,
        leak: float = 0.02,
    ):
        self.graph = load_structure()
        self.factors = [n for n in self.graph.nodes if n != "severe_delay"]
        self.states = default_states(self.factors)
        self.p_cause = float(p_cause)
        self.leak = float(leak)
        if factor_priors is None:
            factor_priors = {f: 0.2 for f in self.factors}
        self.factor_priors = {f: float(factor_priors.get(f, 0.2)) for f in self.factors}
        self.factor_cpts = {f: bernoulli_table(self.factor_priors[f]) for f in self.factors}
        self.severe_cpt = noisy_or_severe_table(len(self.factors), leak=self.leak, p_cause=self.p_cause)
        self.evidence: dict[str, int] = {}

    def clone_fresh(self) -> "DelayRiskBN":
        twin = DelayRiskBN(
            factor_priors=self.factor_priors,
            p_cause=self.p_cause,
            leak=self.leak,
        )
        return twin

    def reset(self) -> None:
        self.evidence = {}

    def observe(self, factor: str, present: bool | int) -> None:
        if factor not in self.factors:
            raise KeyError(factor)
        self.evidence[factor] = 1 if present else 0

    def _cfg_prob(self, cfg: int) -> float:
        p = 1.0
        for i, f in enumerate(self.factors):
            bit = 1 if (cfg & (1 << i)) else 0
            if f in self.evidence and self.evidence[f] != bit:
                return 0.0
            p *= self.factor_cpts[f][bit]
        return p

    def query(self) -> Posterior:
        n = len(self.factors)
        n_cfg = 1 << n
        z = 0.0
        p_sev = 0.0
        marg = {f: 0.0 for f in self.factors}
        for cfg in range(n_cfg):
            prior = self._cfg_prob(cfg)
            if prior == 0.0:
                continue
            p_yes = self.severe_cpt[cfg, 1]
            joint = prior
            z += joint
            p_sev += joint * p_yes
            for i, f in enumerate(self.factors):
                if cfg & (1 << i):
                    marg[f] += joint
        if z <= 0:
            return Posterior(0.5, {f: 0.5 for f in self.factors})
        return Posterior(p_severe=p_sev / z, factor_marginals={f: marg[f] / z for f in self.factors})

    def as_network(self) -> BayesianNetwork:
        return BayesianNetwork(graph=self.graph, cpts={}, states=self.states)
