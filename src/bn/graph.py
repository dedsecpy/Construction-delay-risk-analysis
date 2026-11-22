"""Discrete Bayesian network over 8 risk factors → severe_delay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BayesianNetwork:
    graph: nx.DiGraph
    cpts: dict[str, np.ndarray]
    states: dict[str, list[str]] = field(default_factory=dict)

    def parents(self, node: str) -> list[str]:
        return list(self.graph.predecessors(node))

    def nodes(self) -> list[str]:
        return list(self.graph.nodes)


def load_structure() -> nx.DiGraph:
    edges = pd.read_csv(ROOT / "data" / "processed" / "edges.csv")
    g = nx.DiGraph()
    factors = pd.read_csv(ROOT / "data" / "processed" / "factors.csv")["factor_id"].tolist()
    for f in factors:
        g.add_node(f)
    g.add_node("severe_delay")
    for _, row in edges.iterrows():
        g.add_edge(str(row["parent"]), str(row["child"]))
    return g


def default_states(factors: list[str]) -> dict[str, list[str]]:
    states = {f: ["absent", "present"] for f in factors}
    states["severe_delay"] = ["no", "yes"]
    return states
