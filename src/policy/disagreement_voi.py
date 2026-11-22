from __future__ import annotations

from src.policy.consensus_voi import ConsensusVoIPolicy


class DisagreementVoIPolicy(ConsensusVoIPolicy):
    """Same VoI machinery; intended to be paired with disagreement priors in the BN."""

    name = "disagreement_voi"
