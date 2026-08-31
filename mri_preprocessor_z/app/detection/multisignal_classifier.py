"""
Multi-signal modality classifier: aggregates evidence from multiple
SignalSource instances into a single ClassificationResult with weighted
confidence scoring.

Confidence model
----------------
For each candidate modality, we sum the weights of all signals that
voted for it. Then:

    confidence = top_score / (top_score + sum_of_other_scores)

This is bounded [0, 1]:
  - All signals agree on one modality -> top_score = total, others = 0
    -> confidence = 1.0
  - Signals split 50/50 between two modalities -> confidence = 0.5
  - One modality gets 3 votes (weights sum 3.0), another gets 1 vote
    (weight 1.0) -> confidence = 3.0 / (3.0 + 1.0) = 0.75

Manual-review threshold
-----------------------
If the top modality's confidence is below `confidence_threshold`
(default 0.5), the result's `modality` is set to None. The signals
and alternatives are still populated, so the UI can show "we found
evidence for X (0.4) and Y (0.3) -- please confirm".

The threshold is configurable per-classifier instance, so different
deployment contexts (research vs clinical) can tune it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.data_model.schema import ClassificationResult, ClassificationSignal
from app.detection.signals import SignalSource, default_signals


class MultiSignalClassifier:
    """Aggregates evidence from multiple signal sources.

    Default configuration uses the standard signal set
    (FilenameSignal, PathSignal, SidecarSignal, DimensionSignal,
    HeaderSignal). Callers may pass a custom list to add or replace
    signals, e.g.::

        classifier = MultiSignalClassifier(
            signals=[FilenameSignal(weight=2.0), SidecarSignal()],
            confidence_threshold=0.6,
        )
    """

    def __init__(
        self,
        signals: Optional[List[SignalSource]] = None,
        confidence_threshold: float = 0.5,
    ):
        self.signals: List[SignalSource] = (
            signals if signals is not None else default_signals()
        )
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        filepath: str,
        filename: str,
        sidecars: dict,
        metadata: dict,
    ) -> ClassificationResult:
        # 1. Run each signal source. Each may return a ClassificationSignal
        # voting for one modality, or None to abstain.
        signals: List[ClassificationSignal] = []
        for source in self.signals:
            sig = source.evaluate(filepath, filename, sidecars, metadata)
            if sig is not None:
                signals.append(sig)

        # 2. Sum weights per modality.
        scores: Dict[str, float] = {}
        for sig in signals:
            scores[sig.modality] = scores.get(sig.modality, 0.0) + sig.weight

        # No signals fired -> fully unclassified.
        if not scores:
            return ClassificationResult(
                modality=None,
                confidence=0.0,
                signals=[],
                alternatives=[],
            )

        # 3. Sort modalities by total score, descending. Ties broken by
        #    modality name (deterministic) so the output is stable.
        sorted_modalities = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        top_modality, top_score = sorted_modalities[0]
        total_score = sum(scores.values())

        # 4. Compute confidence: top_score / (top_score + sum_of_others).
        #    - One modality only -> confidence = 1.0
        #    - Split between two -> ratio = top / total
        #    - Total 0 is unreachable here (we early-returned above).
        if len(sorted_modalities) == 1:
            confidence = 1.0
        else:
            other_score = total_score - top_score
            confidence = top_score / (top_score + other_score)

        # 5. Alternatives: every other modality that got at least one vote,
        #    sorted by score descending. Returned as a list of
        #    (modality, score) tuples.
        alternatives = [(m, s) for m, s in sorted_modalities[1:]]

        # 6. Threshold check. If confidence is below the threshold, we
        #    return modality=None (flagged for manual review) but keep
        #    the signals + alternatives so the UI can explain why.
        if confidence < self.confidence_threshold:
            return ClassificationResult(
                modality=None,
                confidence=confidence,
                signals=signals,
                alternatives=alternatives,
            )

        return ClassificationResult(
            modality=top_modality,
            confidence=confidence,
            signals=signals,
            alternatives=alternatives,
        )


# Default singleton, used by the convenience function and the builder.
_default_classifier = MultiSignalClassifier()


def classify_acquisition(
    filepath: str,
    filename: str,
    sidecars: dict,
    metadata: dict,
) -> ClassificationResult:
    """Convenience function: classify using the default singleton.

    This is the entry point called by `app.data_model.builder._classify_acquisition`.
    """
    return _default_classifier.classify(filepath, filename, sidecars, metadata)
