#!/usr/bin/env bash
set -euo pipefail

# Build thesis_proposal.pdf using latexmk
latexmk -pdf -interaction=nonstopmode thesis_proposal.tex
