#!/usr/bin/env python3
"""Regenerate the v0 action and intervention figure from verified scalar evidence."""
import argparse
from pathlib import Path
from reproduce_results import analyze, TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('assets/figures/action_diagnostics.png'))
    args = parser.parse_args()
    data = analyze(args.results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    labels = ['Frozen', 'Retention', 'Std box', 'Std drawer', 'Std cups', 'SF box', 'SF drawer', 'SF cups']
    keys = ['frozen', 'retention'] + [a + ':' + t for a in ('standard', 'semflow') for t in TARGETS]
    values = [data['actions']['10'][k]['first_eight'] for k in keys]
    axes[0].bar(labels, values, color=['#505f79', '#87939f'] + ['#dba24f'] * 3 + ['#448c8a'] * 3)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_title('Human adaptation did not improve source actions')
    axes[0].set_ylabel('Normalized first-eight-action MSE (lower is better)')
    interventions = ['none', 'zero', 'static', 'shuffle', 'bypass']
    x = np.arange(len(interventions))
    for i, (model, label) in enumerate(zip(('frozen', *TARGETS), ('Frozen', 'SF box', 'SF drawer', 'SF cups'))):
        y = [data['ablations'][model]['all'][v]['first_eight']['mse'] for v in interventions]
        axes[1].bar(x + (i - 1.5) * .19, y, .19, label=label)
    axes[1].set_xticks(x, ['Predicted', 'Zero', 'Static', 'Shuffle', 'Bypass'])
    axes[1].set_title('Whole-conditioner bypass substantially lowers error')
    for ax in axes:
        ax.axhline(data['baseline']['first_eight'], color='#9b3038', linestyle='--', label='Source-mean baseline')
        ax.set_ylim(0, 1.85)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].legend(loc='lower left', fontsize=8)
    fig.supxlabel('12 source-validation windows × 4 paired noise seeds; 10 sampling steps. One adaptation seed.\nBypass removes current and future scene injection; these are offline diagnostics, not rollout success.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
