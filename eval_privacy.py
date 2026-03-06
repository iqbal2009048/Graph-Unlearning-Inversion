#!/usr/bin/env python
"""eval_privacy.py — Comparison evaluation script for the privacy-preserving framework.

Runs both the **baseline** experiment (no privacy flags) and the **proposed**
experiment (all privacy flags enabled) back-to-back and prints a formatted
comparison table of attack AUC, attack F1 and node classification accuracy.

Usage::

    python eval_privacy.py --dataset_name cora --target_model GCN \
        --method GIF --attack_method trend_mia \
        --is_gen_unlearn_request True --is_gen_unlearned_probs True

All standard ``main.py`` arguments are accepted; the privacy flags
(``--concept_leakage``, ``--privacy_mask``, ``--adversarial_training``) are
managed internally by this script and should *not* be passed on the command
line.
"""

import logging
import os
import sys
import io
from typing import Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Inline import — re-use the same code path as main.py
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parameter_parser import parameter_parser


class _MetricsHandler(logging.Handler):
    """Logging handler that extracts key metrics from log records."""

    def __init__(self):
        super().__init__()
        self.metrics = {'auc': 'N/A', 'f1': 'N/A', 'node_acc': 'N/A'}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        # Combined attack metrics line:  "AUC: {auc}, Prec: ..."
        for key in ('auc', 'f1'):
            tag = f'{key}:'
            if tag in msg.lower():
                parts = msg.lower().split(tag)
                if len(parts) > 1:
                    val = parts[1].strip().split()[0].rstrip(',')
                    if self.metrics[key] == 'N/A':
                        self.metrics[key] = val
        # Node classification F1
        if 'Final Test F1' in msg:
            parts = msg.strip().split('Final Test F1:')
            if len(parts) > 1:
                val = parts[1].strip().split()[0]
                self.metrics['node_acc'] = val


def run_experiment(args: Dict, enable_privacy: bool) -> Dict[str, str]:
    """Run a single experiment and return the key metrics.

    Args:
        args: Parsed argument dictionary from :func:`parameter_parser`.
        enable_privacy: Whether to enable all privacy framework flags.

    Returns:
        Dictionary with keys ``auc``, ``f1``, ``node_acc`` (all strings).
    """
    import random

    from exp.exp_unlearn_inv import ExpUnlearningInversion

    # Override privacy flags on a copy so the original dict is not mutated
    args = dict(args)
    args['concept_leakage'] = enable_privacy
    args['privacy_mask'] = enable_privacy
    args['adversarial_training'] = enable_privacy

    handler = _MetricsHandler()
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        torch.manual_seed(20221012)
        np.random.seed(20221012)
        random.seed(20221012)

        ExpUnlearningInversion(args)

    except (RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
        logging.warning('Experiment raised an error: %s', exc)
    finally:
        root_logger.removeHandler(handler)

    return handler.metrics


def print_table(rows):
    """Print a formatted comparison table to stdout."""
    header = ('Dataset', 'Model', 'Method', 'Attack AUC', 'Attack F1', 'Node Acc')
    col_w = [10, 6, 10, 12, 10, 10]
    fmt = '  '.join(f'{{:<{w}}}' for w in col_w)
    sep = '  '.join('-' * w for w in col_w)

    print()
    print('Privacy Framework Comparison Table')
    print('=' * (sum(col_w) + 2 * (len(col_w) - 1)))
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print('=' * (sum(col_w) + 2 * (len(col_w) - 1)))
    print()


def main():
    logging.basicConfig(
        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
        level=logging.INFO,
        stream=sys.stdout,
    )

    args = parameter_parser()

    dataset = args.get('dataset_name', 'N/A')
    model_name = args.get('target_model', 'N/A')

    logging.info('Running BASELINE experiment (no privacy flags)...')
    baseline_metrics = run_experiment(args, enable_privacy=False)

    logging.info('Running PROPOSED experiment (privacy flags enabled)...')
    proposed_metrics = run_experiment(args, enable_privacy=True)

    rows = [
        (
            dataset, model_name, 'Baseline',
            baseline_metrics.get('auc', 'N/A'),
            baseline_metrics.get('f1', 'N/A'),
            baseline_metrics.get('node_acc', 'N/A'),
        ),
        (
            dataset, model_name, 'Proposed',
            proposed_metrics.get('auc', 'N/A'),
            proposed_metrics.get('f1', 'N/A'),
            proposed_metrics.get('node_acc', 'N/A'),
        ),
    ]

    print_table(rows)

    logging.info('Baseline metrics: %s', baseline_metrics)
    logging.info('Proposed  metrics: %s', proposed_metrics)


if __name__ == '__main__':
    main()
