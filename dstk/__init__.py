"""dstk: small, dependency-light helpers for rigorous data science work.

Modules
-------
stats        bootstrap / Wilson intervals, paired method comparison, A/B sample size
ranking      Precision@k, Recall@k, MRR, AP/MAP@k, NDCG@k, coverage
leakage      duplicate, group, temporal and label-shuffle checks, adversarial validation
drift        PSI (numeric + categorical) and KS drift reports
contracts    data contract / schema validation
preprocess   pipeline-safe outlier handling (fit on train only)
parity      train/serve parity (features, batch vs single, save/load, column order)
timeseries   seasonal naive, MASE/sMAPE/WAPE, rolling-origin splits
experiments  two-proportion test, SRM check, CUPED, Holm / BH corrections
explog       append-only experiment log (JSONL) + Markdown table
latency      p50/p95/p99 latency measurement
report_lint  anti-hype and unverified-number checks for READMEs and reports
"""
__version__ = "1.0.0"
