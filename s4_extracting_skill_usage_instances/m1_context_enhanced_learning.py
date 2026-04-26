"""
s4.m1 — Context-Enhanced Learning Framework (PROJECT_SPECS §4 Method 1).

Training paradigm from the Arora talk: place 'helpful information' (a
'textbook' / phrasebook / few-shot examples) in context during SFT, but do
NOT compute loss over it. Apply dropout (e.g. 20%) to the helpful-context
tokens at train time; at test time the helpful context is absent entirely.

Structure of a training example:
    [ helpful context ] [ question ] [ answer ]
    loss mask:          no            no         yes

Cited outcome: enabled internalization of multi-layer translation skills that
vanilla SFT provably cannot learn from input/output pairs alone.

STATUS: out of scope for this scaffold -- requires a training harness
(HuggingFace Trainer, accelerate, or similar), model weights, and GPU access.
Leaving as a stub so the PROJECT_SPECS 1:1 mapping is preserved.
"""

from __future__ import annotations


def build_context_enhanced_dataset(*args, **kwargs):
    raise NotImplementedError(
        "s4.m1 out of scope — requires a training harness (HF Trainer / accelerate) "
        "and GPU access. See docstring for the structure this method would produce."
    )
