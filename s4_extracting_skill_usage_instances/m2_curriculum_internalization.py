"""
s4.m2 — Curriculum-Based Internalization (PROJECT_SPECS §4 Method 2).

Two-phase curriculum on top of s4.m1:

  Phase 1 (warm-up): train on many random phrasebooks + productions so the
  model learns to pattern-match against its context.

  Phase 2 (target): train on the single target phrasebook with partial
  dropout (e.g. 20%) on the contextual 'textbook' tokens. Teaches the model:
  'don't rely on that information -- sometimes it's missing -- so store it.'

At test time: 100% dropout of the textbook; the model still solves the task,
indicating the skill was internalized.

STATUS: out of scope -- requires training infrastructure. See s4.m1.
"""

from __future__ import annotations


def warmup_phase(*args, **kwargs):
    raise NotImplementedError("s4.m2 out of scope — training harness required")


def target_phase(*args, **kwargs):
    raise NotImplementedError("s4.m2 out of scope — training harness required")
