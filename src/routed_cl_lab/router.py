"""The router: which cluster, and which memory system, for each instance.

The router sees only the current instance, the memory-system
descriptions, and the cluster summaries -- never the source-task label.
It scores each cluster by cosine similarity between the instance's skill
vector and the cluster's summary vector, plus a noisy estimate of how
well the cluster's memory system fits the instance's schema (the router
reads the system descriptions, but its reading is imperfect). If the best
score is below the creation threshold, it opens a new cluster and picks
the memory system with the best noisy fit estimate.

Ground-truth assignment is a privileged diagnostic (paper Table 8): it
uses the source-task identity, which the router never sees, to measure
how often the router agrees with the ideal partition and system choice.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from routed_cl_lab.agent import SYSTEMS, fidelity
from routed_cl_lab.memory import Cluster, make_system
from routed_cl_lab.stream import FAMILIES, Instance

# Similarity below this -> open a new cluster instead of reusing one.
# Calibrated between the cross-family similarity ceiling (~0.45, the
# blended score of an instance against a cluster of another family) and
# the same-family floor (~0.75): a threshold in that gap separates
# "this cluster is about my kind of problem" from "it is not".
CREATE_THRESHOLD = 0.58
MAX_CLUSTERS = 14  # 2x the number of source tasks; rarely binding


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


class Router:
    def __init__(
        self,
        rng: np.random.Generator,
        noise: float = 0.10,
        create_threshold: float = CREATE_THRESHOLD,
        max_clusters: int = MAX_CLUSTERS,
        allowed_systems: list[str] | None = None,
    ) -> None:
        self.rng = rng
        self.noise = noise
        self.create_threshold = create_threshold
        self.max_clusters = max_clusters
        # Restrict the library, e.g. ["icl"] for the Routed ICL condition.
        self.allowed_systems = allowed_systems if allowed_systems is not None else SYSTEMS
        self.clusters: list[Cluster] = []
        # Agreement accounting vs ground truth (paper Table 8).
        self.n_routed = 0
        self.n_system_agree = 0
        self.n_partition_agree = 0
        self.n_both_agree = 0

    def _system_fit_estimate(self, system: str, instance: Instance) -> float:
        """Noisy estimate of how well a system fits the instance's schema."""
        return fidelity(system, instance.affinity) + self.rng.normal(0.0, self.noise)

    def _cluster_score(self, cluster: Cluster, instance: Instance) -> float:
        summary = cluster.summary_vector()
        if summary is None:
            # Empty cluster: no evidence either way; neutral score.
            sim = 0.5
        else:
            sim = _cosine(instance.skill, summary)
        fit = self._system_fit_estimate(cluster.system_name, instance)
        # The score blends "is this cluster about my kind of problem" with
        # "does this cluster's memory system fit my kind of problem".
        return 0.7 * sim + 0.3 * fit + self.rng.normal(0.0, self.noise)

    def _new_cluster(self, instance: Instance, step: int) -> Cluster:
        # Pick the memory system with the best noisy fit estimate.
        best_system = max(
            self.allowed_systems, key=lambda s: self._system_fit_estimate(s, instance)
        )
        system = make_system(best_system, self.rng)
        cluster = Cluster(system_name=best_system, system=system, created_at=step)
        self.clusters.append(cluster)
        return cluster

    def route(self, instance: Instance, step: int) -> Cluster:
        """Assign an instance to a cluster, creating one if warranted."""
        if not self.clusters:
            cluster = self._new_cluster(instance, step)
        else:
            scores = [self._cluster_score(c, instance) for c in self.clusters]
            best_idx = int(np.argmax(scores))
            if scores[best_idx] < self.create_threshold and len(self.clusters) < self.max_clusters:
                cluster = self._new_cluster(instance, step)
            else:
                cluster = self.clusters[best_idx]
        self._record_agreement(instance, cluster)
        return cluster

    def _record_agreement(self, instance: Instance, cluster: Cluster) -> None:
        """Compare the routing decision against ground truth."""
        self.n_routed += 1
        system_ok = cluster.system_name == instance.affinity
        # Partition agreement: the cluster's members are (mostly) the same
        # family as this instance. A fresh cluster trivially agrees; an
        # established cluster agrees if its majority family matches.
        if not cluster.members:
            partition_ok = True
        else:
            families = [m.family for m in cluster.members]
            # Counter.most_common breaks ties by first-seen order, which
            # is deterministic; max() over a set is not (hash order).
            majority = Counter(families).most_common(1)[0][0]
            partition_ok = majority == instance.family
        self.n_system_agree += int(system_ok)
        self.n_partition_agree += int(partition_ok)
        self.n_both_agree += int(system_ok and partition_ok)

    def agreement_rates(self) -> dict[str, float]:
        if self.n_routed == 0:
            return {"system": 0.0, "partition": 0.0, "both": 0.0}
        return {
            "system": self.n_system_agree / self.n_routed,
            "partition": self.n_partition_agree / self.n_routed,
            "both": self.n_both_agree / self.n_routed,
        }


def ground_truth_cluster(
    clusters: dict[str, Cluster], instance: Instance, rng: np.random.Generator, step: int
) -> Cluster:
    """Privileged diagnostic: one cluster per family, ideal system.

    The router never sees this; it exists only to measure headroom, as in
    the paper's Ground-truth assignment condition.
    """
    key = instance.family
    if key not in clusters:
        system = make_system(instance.affinity, rng)
        clusters[key] = Cluster(
            system_name=instance.affinity, system=system, created_at=step
        )
    return clusters[key]
