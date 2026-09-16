#!/usr/bin/env python3
"""Prepare a completed-outcome credit checkpoint branch without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pylatro_agent.training.outcome_credit import prepare_outcome_credit_branch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="Trusted complete-cohort full PPO checkpoint")
    parser.add_argument("--destination", type=Path, required=True, help="New branch checkpoint path; never overwritten")
    args = parser.parse_args(argv)
    result = prepare_outcome_credit_branch(args.parent, args.destination)
    # Full source hashes and configuration history are embedded in the file.
    print(json.dumps({key: value for key, value in result.items() if key != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
