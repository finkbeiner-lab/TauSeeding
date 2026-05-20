# FRET Analysis Protocol — Tau-RD Biosensor HEK Cells

## Overview

This protocol describes the sensitized emission FRET imaging analysis used to detect tau aggregation in HEK cells expressing Tau-RD-CFP and Tau-RD-YFP biosensors. Tau aggregation brings CFP and YFP into proximity, increasing FRET signal.

## Cell Lines and Constructs

- **Tau-Halo-P2a-mScarlet3** ("Tau-Halo"): Tau-RD fused to HaloTag, with P2a-linked mScarlet3 for expression normalization
- **Tau-VVD-Halo-P2a-mScarlet3** ("VVD-Tau"): Tau-RD with VVD optogenetic domain and HaloTag

Both cell lines co-express Tau-RD-CFP and Tau-RD-YFP for FRET-based detection of tau aggregation.

## Seeding Conditions

- **K18**: Recombinant tau K18 fragment (aggregation-competent seed)
- **FL**: Full-length recombinant tau seed
- **Unseeded-lipo**: Lipofectamine only (negative control, no tau seeds)

## Imaging Channels

Three primary FRET channels are acquired per cell:

| Channel Name | Excitation | Emission | Abbreviation |
|---|---|---|---|
| CFP-DMD | CFP excitation | CFP emission | I_DD |
| CFP-FRET | CFP excitation | YFP emission | I_DA |
| YFP-DMD | YFP excitation | YFP emission | I_AA |

Additional channels:

- **RFP1**: mScarlet3 signal, used for expression-level normalization
- **Cy5cleanup**: HaloTag dye signal (Aggfluor or JF646), used to measure tau-HaloTag labeling

## Light Stimulation

Tiles receive different stimulation durations (in ms): 0, 500, 2000, 10000. The tile-to-stimulation mapping is:

| Tile | Stim (ms) | Tile | Stim (ms) |
|---|---|---|---|
| 1 | 0 | 9 | 2000 |
| 2 | 500 | 10 | 10000 |
| 3 | 2000 | 11 | 0 |
| 4 | 10000 | 12 | 500 |
| 5 | 10000 | 13 | 500 |
| 6 | 0 | 14 | 2000 |
| 7 | 500 | 15 | 10000 |
| 8 | 2000 | 16 | 0 |

## FRET Computation

### Step 1: Corrected FRET Signal (Fc)

The raw CFP-FRET channel (I_DA) contains bleedthrough from both the donor (CFP) and direct excitation of the acceptor (YFP). These are subtracted using empirically determined correction factors:

$$F_c = I_{DA} - \alpha \cdot I_{DD} - \delta \cdot I_{AA}$$

Where:

- **I_DA** = CFP-FRET intensity (sensitized emission channel)
- **I_DD** = CFP-DMD intensity (donor channel)
- **I_AA** = YFP-DMD intensity (acceptor channel)
- **α = 0.536** = donor bleedthrough coefficient (fraction of CFP signal leaking into the FRET channel)
- **δ = 0.350** = direct excitation coefficient (fraction of YFP directly excited by CFP excitation wavelength)

Fc represents the true FRET signal after removing spectral contamination. Positive Fc indicates energy transfer (proximity of CFP and YFP), which in this system reports on tau aggregation.

### Step 2: FRET Efficiency (E)

FRET efficiency quantifies the fraction of donor excitation energy transferred to the acceptor:

$$E = \frac{F_c}{F_c + G \cdot I_{DD}}$$

Where:

- **G = 0.65** = instrument-specific G-factor that accounts for differences in detection efficiency and quantum yield between donor and acceptor

FRET efficiency ranges from 0 (no transfer) to 1 (complete transfer). In practice, aggregation-positive cells show elevated E values.

### Key Metrics Used

For population-level analysis, per-well or per-tile statistics are computed at each timepoint:

- **Median Fc**: Robust central tendency of corrected FRET, less sensitive to outliers
- **Mean Fc**: Arithmetic mean of corrected FRET
- **95th percentile Fc**: Captures the high-FRET tail of the distribution, most sensitive to detecting seeded aggregation since only a subset of cells may be aggregate-positive
- **Median/Mean/95th percentile YFP-DMD**: Acceptor channel signal, used to monitor YFP expression over time

The 95th percentile Fc was found to be the most sensitive metric for distinguishing seeded from unseeded conditions.

## Normalizations

### ΔF/F (Fold Change from Baseline)

Per-well normalization to the first timepoint:

$$\Delta F/F = \frac{V(t) - V(t_0)}{V(t_0)}$$

Where V(t) is the metric value at timepoint t and V(t₀) is the value at the first available timepoint. This removes well-to-well baseline differences and shows fractional change over time.

### Z-Score Normalization

Used when comparing signals of very different magnitudes (e.g., Aggfluor vs JF646 Cy5cleanup). Per-dye Z-scoring:

$$Z = \frac{V - \mu_{dye}}{\sigma_{dye}}$$

Where μ_dye and σ_dye are the global mean and standard deviation of all well-timepoint values for that dye type. This places both dyes on a common unitless scale (standard deviations from mean).

### ΔZ-Score (Baseline-Normalized Z-Score)

Combines Z-scoring with first-timepoint normalization:

1. Z-score the metric per dye (as above)
2. Subtract each well's first-timepoint Z-score: ΔZ(t) = Z(t) − Z(t₀)

All traces start at 0, showing change from baseline on the Z-scored scale.

## Cy5cleanup / HaloTag Dye Analysis

The Cy5cleanup channel measures HaloTag dye binding to the tau-HaloTag construct:

- **Well columns 8, 10**: Aggfluor dye (P1h Aggfluor)
- **Well columns 9, 11**: JF646 dye

### Cy5cleanup/RFP1 Ratio

Per-cell normalization for expression level:

$$\text{Ratio} = \frac{\text{Cy5cleanup}_{\text{cell}}}{\text{RFP1}_{\text{cell}}}$$

This controls for differences in tau-HaloTag expression between cells, since RFP1 (mScarlet3) is co-expressed via P2a with the tau construct.

### Integrated Density

For area-weighted signal:

$$\text{Integrated Density} = \text{Mean Intensity} \times \text{Cell Area}$$

## Data Aggregation Hierarchy

1. **Per-cell**: Raw measurements from segmented cells (celldata)
2. **Per-tile**: Cells grouped by imaging tile within a well
3. **Per-well**: Cells grouped by well (biological replicate unit)
4. **Per-condition**: Wells grouped by cell line × seed type × stim duration

Error bars throughout use **standard error of the mean (SEM)** computed across wells (biological replicates), not across individual cells.

## CSV Data Structure

Source data comes from the imaging pipeline as relational tables:

- **celldata.csv**: Per-cell measurements (id, area, stimulate, randomcellid)
- **channeldata.csv**: Channel definitions (id, welldata_id, channel name)
- **intensitycelldata.csv**: Per-cell-per-channel intensities
- **tiledata.csv**: Tile metadata (tile number, timepoint, welldata_id)
- **welldata.csv**: Well identifiers
- **dosagedata.csv**: Condition metadata (cell line name, seed type)

These are merged by joining on shared keys (celldata_id, channeldata_id, welldata_id, tiledata_id), then pivoting channels into columns to produce a single per-cell row with all channel intensities.
