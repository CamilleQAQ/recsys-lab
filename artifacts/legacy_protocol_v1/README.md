# Historical protocol v1 — not current release results

These files preserve the originally reported MovieLens/KuaiRec checkpoints and
metrics. They must not be evaluated against the v2 prepared data or cited as
v2 results.

- MovieLens v1 sorted equal-second events by item index and held out the last
  two rows even when their real order was unknown.
- KuaiRec v1 combined a pair's maximum watch ratio with its last playback
  timestamp. The label and time could come from different raw events.
- v1 seen-item filtering used positive train history (and, for standard
  sequential test, train + validation), not all prior raw exposures.

Protocol v2 uses strict timestamp boundaries, an actual first-play event for
KuaiRec labels, and all prior observed exposures for candidate filtering.
Revised checkpoints and metrics live outside this directory.
