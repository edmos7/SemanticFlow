# Project information and constraints

Prepared for Humanoid's [Internship, Robot Learning Research application](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221). The challenge allows two weeks and specifies a deadline of **16 October 2026, 23:59 BST**. It asks for personally collected real data used in a robot-learning project, reproducible instructions, example outputs, and clear design choices and findings.

## Hardware and scope

**No physical robot hardware was available. All robot experiments were simulation-only.** Human data came from three phone recordings held near the chest. There was no robot data collection, hardware deployment, or real-world robot evaluation. The world predictions are learned scene features, not rendered future RGB videos.

The project reached human-video adaptation and offline action diagnostics. It did not demonstrate successful human-adapted control on the target compositions. An earlier source-policy simulation test completed 17 trials with zero successes before timing out; the planned rollout matrix was not completed.

## Time and compute

The two-week submission window included integration, source calibration, simulator debugging, data ingestion, adaptation, and recovery diagnostics. Available time and cloud compute limited the final experiment to three clips, one adaptation seed, 60 updates per clip and method, and a bounded source validation panel. The planned three training seeds and 50 target initial states were not run. No held-out human recording was available.

Measured final stages ran on an **NVIDIA A100-SXM4-80GB**:

| Stage | Updates | Stage time | Peak allocated CUDA memory |
|---|---:|---:|---:|
| Standard human adaptation, three clips | 180 | 306.5 s | 7.27 GiB |
| SemFlow human adaptation, three clips | 180 | 279.6 s | 7.27 GiB |
| Retention-only control | 60 | 111.5 s | 5.96 GiB |

Stage times include setup and caching within each stage. The three human update loops themselves took about 162 seconds per method in total. These measurements are not the end-to-end project time and exclude earlier calibration, retries, idle runtime, and other evaluation stages. Peak allocated memory is not total GPU memory usage or a verified minimum GPU specification.

[Stage measurements](../results/stages.csv) record measured runtime and memory. [Compute records](../results/compute.csv) retain available displayed cloud compute-unit differences, including unsuccessful sessions. Missing values mean missing measurements, not free compute. Balance differences are not precise billing records; a complete project cost cannot be reconstructed from them.

These constraints explain the pilot's scale. They do not turn its negative results into evidence of successful transfer. The decision was to stop this interface and training setup before spending more compute on target rollouts.
