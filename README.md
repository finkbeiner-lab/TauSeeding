# TauSeeding

HEK and mouse neuron seeding experiments with PFFs, VVD-Tau, and AggFluor.

## Live Analysis Report

**[View the interactive report →](https://finkbeiner-lab.github.io/TauSeeding/)**

## Repository Structure

```
index.html              ← Interactive analysis report (GitHub Pages)
code/
  fret_pipeline.py      ← Reusable FRET analysis pipeline
  test_fret_pipeline.py ← Unit tests (pytest)
  calibration.py        ← FRET correction factor calibration
  fret.py               ← Core FRET computation module
  FRET_analysis_protocol.md  ← Written protocol
figures/                ← High-resolution figures (for publications)
```

## FRET Analysis

Sensitized emission FRET using Tau-RD-CFP / Tau-RD-YFP biosensors:

- **Corrected FRET:** Fc = I_DA − α·I_DD − δ·I_AA (α=0.536, δ=0.350)
- **FRET Efficiency:** E = Fc / (Fc + G·I_DD) (G=0.65)

## Data Pipeline

Thinking Microscope → Nextflow processing → NAS → Local analysis

## Authors

Jeremy Linsley, Austin Holub, Shijie Wang
