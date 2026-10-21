# UrbanTrace AI — Technical Documentation

This folder is the in-depth documentation for UrbanTrace AI. The top-level
[README](../README.md) is the quick start; [USAGE.md](USAGE.md) is the original
long-form user guide. These documents explain **what the system is, why it is
built the way it is, and how well it works**.

| # | Document | Answers |
|---|---|---|
| 1 | [Solution, innovation and novelty](01-solution-and-novelty.md) | What problem is solved, what is new, how it differs from a plain ANPR system |
| 2 | [Technology choices](02-technology-choices.md) | Why each technology was chosen, what was considered instead, why not |
| 3 | [Models and measured accuracy](03-models-and-accuracy.md) | Every model, its training, its measured precision/recall/mAP/exact-match, its limits |
| 4 | [Tracking and identity resolution](04-tracking-and-identity.md) | Why ByteTrack, how plates bind to tracks, how one vehicle is linked across cameras |
| 5 | [Database design (DBMS)](05-database-dbms.md) | PostgreSQL + PostGIS, schema, partitioning, indexing, pooling, transactions |
| 6 | [Networking and load balancing](06-networking-and-load-balancing.md) | Streams, protocols, nginx, worker assignment, failover, adaptive load shedding |
| 7 | [Concurrency and thread safety](07-concurrency-and-thread-safety.md) | Every lock, every shared structure, and what is deliberately not locked |
| 8 | [Security](08-security.md) | Authentication, input validation, secrets, container hardening, known gaps |
| 9 | [Verification, defects found, and limitations](09-verification-and-limitations.md) | What was tested live, what was fixed, what is still unproven |

## How to read the numbers

Every figure in these documents is one of three kinds, and each is labelled:

* **Measured** — produced by running code in this repository (the accuracy
  notebook `testing_accuracy.ipynb`, `scripts/bench_fleet.py`, the test suite, or
  the live checks described in document 9). The source is named next to it.
* **Recorded** — read from a model checkpoint's own training record. It describes
  the training run, not this project's footage.
* **Design rationale** — an argument for why a choice is sensible. It is *not* a
  benchmark. Where this document compares against an alternative that was not
  actually run here, it says so.

Nothing is rounded up. Where a number is weak (OCR exact-match is 71.3 %), the
documents say so and explain why.
