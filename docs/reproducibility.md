# Reproducibility and v0 limits

## Verified numerical path

From the repository root, with Python 3.10 or later:

```bash
python3 tools/reproduce_results.py --results results
```

This standard-library command checks SHA-256 evidence hashes, validation-sample coverage, paired action-noise identities, and the reported diagnostic decision. It recomputes action aggregates, forecast interventions, human forecast fitting, and the increment over retention-only. Expected decision: `pause_current_approach`; expected mean increment: approximately `0.024788%`.

The curated tables contain 1,152 action samples, 960 intervention samples, 144 human-fit rows, and 420 optimizer-update rows. They were exported after verifying 13 experiment reports against saved archive/report/checkpoint receipts. Hashes establish file integrity, not independent verification of how experiments were run. Evidence pins, configurations, and the original decision are preserved in [summary.json](../results/summary.json).

To regenerate the main figure with Matplotlib and NumPy installed:

```bash
python3 tools/plot_submission.py
```

## GPU stages

The numeric path does not rerun GPU training or simulation. Existing entry points and stage contracts are documented in [source ingestion](source_ingestion.md), [native integration](egosteer_integration.md), [human adaptation](human_adaptation.md), [action evaluation](human_action_panel.md), and [decision diagnostics](decision_diagnostics.md). These detailed documents retain historical stage plans; the executed outcome is summarized in the [research report](research_report.md).

Verified runs used an A100 80 GB, Python 3.12.14, Torch 2.10.0+cu130, Transformers 5.6.2, and FlashAttention 2.8.3. Model revisions and evidence hashes are recorded in the summary. Model and dataset access and upstream licenses apply. Raw recordings and checkpoints are not included in v0. Asset names in the summary identify intended media outputs; the video files are not yet packaged.

A clean-machine end-to-end training recipe and portable video inputs remain pending. The original checkpoint chain and pinned dependencies are required to rerun the historical GPU comparisons. Numeric reproduction is the verified path in this version. Passing software tests does not establish successful robot transfer.
