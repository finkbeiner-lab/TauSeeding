#!/usr/bin/env python3
"""
FRET Analysis Pipeline — Tau-RD Biosensor
==========================================
Reusable script for processing FRET imaging data from CSV exports.

Usage:
    python fret_pipeline.py --csvs /path/to/CSVS --out /path/to/output

    # With Cy5 data:
    python fret_pipeline.py --csvs /path/to/CSVS4 --out /path/to/output --cy5

    # Custom correction factors:
    python fret_pipeline.py --csvs /path/to/CSVS --out /path/to/output \
        --alpha 0.536 --delta 0.350 --G 0.65

Author: Jeremy Linsley, Bhatt Lab, Gladstone/UCSF
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# DEFAULT PARAMETERS
# ============================================================
DEFAULT_ALPHA = 0.536   # Donor bleedthrough
DEFAULT_DELTA = 0.350   # Direct excitation
DEFAULT_G = 0.65        # G-factor

# Tile → stimulation duration mapping
DEFAULT_STIM = {
    1:0, 2:500, 3:2000, 4:10000, 5:10000, 6:0, 7:500, 8:2000,
    9:2000, 10:10000, 11:0, 12:500, 13:500, 14:2000, 15:10000, 16:0
}

# Cell line name mapping
DEFAULT_CELL_LINES = {
    'Tau-Halo-P2a-mScarlet3': 'Tau-Halo',
    'Tau-VVD-Halo-P2a-mScarlet3': 'VVD-Tau',
}

# Well column → dye mapping (for Cy5 experiments)
DEFAULT_DYE_COLS = {8: 'Aggfluor', 9: 'JF646', 10: 'Aggfluor', 11: 'JF646'}

# Plot style
PLOT_PARAMS = {
    'font.size': 32,
    'axes.titlesize': 38,
    'axes.labelsize': 32,
    'legend.fontsize': 26,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'lines.linewidth': 3,
    'lines.markersize': 11,
}


# ============================================================
# DATA LOADING
# ============================================================
def load_and_merge_csvs(base_path, fret_channels=None):
    """
    Merge relational CSV tables from imaging pipeline into per-cell dataframe.

    Parameters
    ----------
    base_path : str
        Path to CSVS directory containing celldata.csv, channeldata.csv, etc.
    fret_channels : list, optional
        Channel names to include. Default: CFP-DMD, CFP-FRET, RFP1, YFP-DMD

    Returns
    -------
    pd.DataFrame
        Per-cell dataframe with one column per channel, plus metadata.
    """
    if fret_channels is None:
        fret_channels = ['CFP-DMD', 'CFP-FRET', 'RFP1', 'YFP-DMD']

    print(f"Loading data from {base_path}...")

    # Load tables
    icd = pd.read_csv(f'{base_path}/intensitycelldata.csv')
    ch = pd.read_csv(f'{base_path}/channeldata.csv')[['id', 'welldata_id', 'channel']]
    ch.rename(columns={'id': 'channeldata_id'}, inplace=True)

    # Check for Cy5cleanup
    available_channels = ch['channel'].unique()
    if 'Cy5cleanup' in available_channels and 'Cy5cleanup' not in fret_channels:
        fret_channels = fret_channels + ['Cy5cleanup']
        print("  Found Cy5cleanup channel — including it")

    # Merge intensity with channel info
    m = icd.merge(ch, on=['channeldata_id', 'welldata_id'], how='inner')
    m = m[m['channel'].isin(fret_channels)]
    print(f"  Intensity records after channel filter: {len(m):,}")

    # Merge cell data
    cd = pd.read_csv(f'{base_path}/celldata.csv')[['id', 'tiledata_id', 'randomcellid', 'area', 'stimulate']]
    cd.rename(columns={'id': 'celldata_id'}, inplace=True)
    m = m.merge(cd, on=['celldata_id', 'tiledata_id'], how='inner')

    # Merge tile data
    td = pd.read_csv(f'{base_path}/tiledata.csv')[['id', 'welldata_id', 'tile', 'timepoint']]
    td.rename(columns={'id': 'tiledata_id'}, inplace=True)
    td_dedup = td.drop_duplicates(subset=['tiledata_id'])
    m = m.merge(td_dedup, on='tiledata_id', how='inner', suffixes=('', '_td'))

    # Merge well data
    wd = pd.read_csv(f'{base_path}/welldata.csv')[['id', 'well']]
    wd.rename(columns={'id': 'welldata_id'}, inplace=True)
    m = m.merge(wd, on='welldata_id', how='inner')

    # Merge dosage/condition data
    dd = pd.read_csv(f'{base_path}/dosagedata.csv')
    if 'welldata_id' in dd.columns and 'name' in dd.columns:
        dd_cols = ['welldata_id']
        if 'name' in dd.columns:
            dd_cols.append('name')
        if 'kind' in dd.columns:
            dd_cols.append('kind')
        if 'condition' in dd.columns:
            dd_cols.append('condition')
        dd_sub = dd[dd_cols].drop_duplicates()
        m = m.merge(dd_sub, on='welldata_id', how='inner')

    # Pivot channels to columns
    id_cols = [c for c in m.columns if c not in ['channeldata_id', 'channel', 'intensity',
                                                   'welldata_id_td', 'welldata_id']]
    # Keep welldata_id for later use
    if 'welldata_id' not in id_cols:
        id_cols.append('welldata_id')

    pivot = m.pivot_table(index=['celldata_id'], columns='channel', values='intensity',
                          aggfunc='first').reset_index()

    # Merge back metadata
    meta_cols = ['celldata_id', 'well', 'tile', 'timepoint', 'randomcellid', 'area', 'stimulate']
    if 'name' in m.columns:
        meta_cols.append('name')
    if 'kind' in m.columns:
        meta_cols.append('kind')
    meta = m[meta_cols].drop_duplicates(subset=['celldata_id'])

    df = pivot.merge(meta, on='celldata_id', how='inner')

    print(f"  Final merged dataset: {len(df):,} cells, {df['timepoint'].nunique()} timepoints")
    print(f"  Channels: {[c for c in fret_channels if c in df.columns]}")

    return df


def compute_fret(df, alpha=DEFAULT_ALPHA, delta=DEFAULT_DELTA, G=DEFAULT_G):
    """
    Compute corrected FRET (Fc), normalized Fc, and FRET efficiency.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: CFP-DMD, CFP-FRET, YFP-DMD
    alpha, delta, G : float
        FRET correction factors

    Returns
    -------
    pd.DataFrame
        Input dataframe with added Fc, Fc_norm, E columns
    """
    df = df.copy()
    df['Fc'] = df['CFP-FRET'] - alpha * df['CFP-DMD'] - delta * df['YFP-DMD']
    df['Fc_norm'] = df['Fc'] / df['CFP-DMD'].replace(0, np.nan)
    df['E'] = df['Fc'] / (df['Fc'] + G * df['CFP-DMD']).replace(0, np.nan)

    print(f"  Fc range: [{df['Fc'].quantile(0.01):.1f}, {df['Fc'].quantile(0.99):.1f}]")
    print(f"  E range: [{df['E'].quantile(0.01):.4f}, {df['E'].quantile(0.99):.4f}]")

    return df


def add_metadata(df, stim_map=None, cell_line_map=None, dye_col_map=None):
    """Add derived metadata columns: stim_dur, cell_line, dye."""
    df = df.copy()

    if stim_map is None:
        stim_map = DEFAULT_STIM
    if cell_line_map is None:
        cell_line_map = DEFAULT_CELL_LINES

    df['stim_dur'] = df['tile'].map(stim_map)

    if 'name' in df.columns:
        df['cell_line'] = df['name'].map(cell_line_map)

    if dye_col_map is not None and 'well' in df.columns:
        df['well_col'] = df['well'].str.extract(r'(\d+)').astype(int)
        df['dye'] = df['well_col'].map(dye_col_map)

    return df


# ============================================================
# PLOTTING HELPERS
# ============================================================
def get_condition_stats(df, filters, metric_func, timepoints, group_col='well'):
    """
    Compute per-timepoint mean±SEM for a filtered subset.

    Parameters
    ----------
    df : pd.DataFrame
    filters : dict
        Column → value pairs to filter on (e.g., {'cell_line': 'Tau-Halo', 'kind': 'K18'})
    metric_func : callable
        Function applied to each group (e.g., lambda g: g['Fc'].quantile(0.95))
    timepoints : list
        Timepoints to include
    group_col : str
        Column defining biological replicates (default: 'well')

    Returns
    -------
    means, sems : np.array or (None, None) if no data
    """
    sub = df.copy()
    for col, val in filters.items():
        sub = sub[sub[col] == val]

    if len(sub) == 0:
        return None, None

    well_tp = sub.groupby([group_col, 'timepoint']).apply(
        metric_func, include_groups=False
    ).reset_index(name='val')

    tp_stats = well_tp.groupby('timepoint')['val'].agg(['mean', 'sem']).reindex(timepoints)
    return tp_stats['mean'].values, tp_stats['sem'].values


def get_deltaFF_stats(df, filters, metric_func, timepoints, group_col='well', t0=0):
    """Like get_condition_stats but with ΔF/F normalization to t0."""
    sub = df.copy()
    for col, val in filters.items():
        sub = sub[sub[col] == val]

    if len(sub) == 0:
        return None, None

    well_tp = sub.groupby([group_col, 'timepoint']).apply(
        metric_func, include_groups=False
    ).reset_index(name='val')

    wells = well_tp[group_col].unique()
    dff_records = []
    for w in wells:
        wdata = well_tp[well_tp[group_col] == w].set_index('timepoint')['val']
        if t0 not in wdata.index or wdata.loc[t0] == 0:
            continue
        t0_val = wdata.loc[t0]
        for tp in timepoints:
            if tp in wdata.index:
                dff_records.append({group_col: w, 'timepoint': tp, 'dff': (wdata.loc[tp] - t0_val) / t0_val})

    if not dff_records:
        return None, None

    dff_df = pd.DataFrame(dff_records)
    tp_stats = dff_df.groupby('timepoint')['dff'].agg(['mean', 'sem']).reindex(timepoints)
    return tp_stats['mean'].values, tp_stats['sem'].values


def plot_timecourse(ax, timepoints, conditions_data, title, ylabel, legend_kwargs=None):
    """
    Plot multiple conditions on a single axis with error bars.

    Parameters
    ----------
    conditions_data : list of dicts
        Each dict: {'means', 'sems', 'label', 'color', 'linestyle', 'marker'}
    """
    for cond in conditions_data:
        if cond['means'] is None:
            continue
        ax.errorbar(timepoints, cond['means'], yerr=cond['sems'],
                    color=cond['color'], linestyle=cond.get('linestyle', '-'),
                    marker=cond.get('marker', 'o'),
                    label=cond['label'], capsize=4, capthick=1.5)

    ax.set_xlabel('Timepoint')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if legend_kwargs is None:
        legend_kwargs = dict(fontsize=22, ncol=2, loc='upper center',
                            bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    ax.legend(**legend_kwargs)


# ============================================================
# STANDARD ANALYSIS SUITE
# ============================================================
def run_standard_fret_analysis(df, out_dir, exclude_last_tp=True):
    """
    Run the standard suite of FRET plots.

    Generates:
    - Fc time courses (median, mean, p95) — raw and ΔF/F
    - YFP-DMD time courses — raw and ΔF/F
    - Per cell-line × seed type breakdowns
    """
    plt.rcParams.update(PLOT_PARAMS)
    os.makedirs(out_dir, exist_ok=True)

    tps = sorted(df['timepoint'].unique())
    if exclude_last_tp:
        tps = tps[:-1]
        df = df[df['timepoint'].isin(tps)].copy()

    cell_lines = [cl for cl in df['cell_line'].dropna().unique()]
    kinds = [k for k in df['kind'].dropna().unique()]
    stim_durs = sorted(df['stim_dur'].dropna().unique())

    # Color palette
    colors = ['#1976D2', '#E53935', '#43A047', '#8E24AA', '#0D47A1', '#B71C1C',
              '#1B5E20', '#4A148C', '#90CAF9', '#EF9A9A', '#64B5F6', '#E57373']
    markers = ['o', 's', 'v', '^', 'D', 'p']
    linestyles = ['-', '--', ':', '-.']

    # Metric functions
    metrics = [
        (lambda g: g['Fc'].median(), 'Median Fc', 'Fc_median'),
        (lambda g: g['Fc'].mean(), 'Mean Fc', 'Fc_mean'),
        (lambda g: g['Fc'].quantile(0.95), '95th %ile Fc', 'Fc_p95'),
        (lambda g: g['YFP-DMD'].median(), 'Median YFP-DMD', 'YFP_median'),
        (lambda g: g['YFP-DMD'].quantile(0.95), '95th %ile YFP-DMD', 'YFP_p95'),
    ]

    for metric_func, metric_label, metric_tag in metrics:
        # --- Raw ---
        fig, ax = plt.subplots(figsize=(28, 16))
        cond_data = []
        ci = 0
        for cl in cell_lines:
            for kind in kinds:
                for sd in [0, 10000]:  # Just 0ms and 10s for overview
                    means, sems = get_condition_stats(
                        df, {'cell_line': cl, 'kind': kind, 'stim_dur': sd},
                        metric_func, tps
                    )
                    cond_data.append({
                        'means': means, 'sems': sems,
                        'label': f'{cl} {kind} {sd}ms',
                        'color': colors[ci % len(colors)],
                        'linestyle': linestyles[ci % len(linestyles)],
                        'marker': markers[ci % len(markers)],
                    })
                    ci += 1

        plot_timecourse(ax, tps, cond_data, f'{metric_label} — All Conditions', metric_label,
                       legend_kwargs=dict(fontsize=20, ncol=3, loc='upper center',
                                         bbox_to_anchor=(0.5, -0.08), framealpha=0.9))
        plt.tight_layout(rect=[0, 0.14, 1, 1])
        fig.savefig(f'{out_dir}/{metric_tag}_raw.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {metric_tag}_raw.png")

        # --- ΔF/F ---
        fig, ax = plt.subplots(figsize=(28, 16))
        cond_data_dff = []
        ci = 0
        for cl in cell_lines:
            for kind in kinds:
                for sd in [0, 10000]:
                    means, sems = get_deltaFF_stats(
                        df, {'cell_line': cl, 'kind': kind, 'stim_dur': sd},
                        metric_func, tps, t0=tps[0]
                    )
                    cond_data_dff.append({
                        'means': means, 'sems': sems,
                        'label': f'{cl} {kind} {sd}ms',
                        'color': colors[ci % len(colors)],
                        'linestyle': linestyles[ci % len(linestyles)],
                        'marker': markers[ci % len(markers)],
                    })
                    ci += 1

        ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
        plot_timecourse(ax, tps, cond_data_dff, f'ΔF/F {metric_label}', f'ΔF/F ({metric_label})',
                       legend_kwargs=dict(fontsize=20, ncol=3, loc='upper center',
                                         bbox_to_anchor=(0.5, -0.08), framealpha=0.9))
        plt.tight_layout(rect=[0, 0.14, 1, 1])
        fig.savefig(f'{out_dir}/{metric_tag}_dFF.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {metric_tag}_dFF.png")

    print(f"\nStandard FRET analysis complete. Output in {out_dir}/")


def run_cy5_analysis(df, out_dir, dye_col_map=None):
    """
    Run Cy5cleanup-specific analyses (requires Cy5cleanup column).

    Generates:
    - Cy5 time courses per dye
    - Z-scored dye comparison
    - Cy5/RFP1 ratio plots
    """
    if 'Cy5cleanup' not in df.columns:
        print("No Cy5cleanup column — skipping Cy5 analysis")
        return

    plt.rcParams.update(PLOT_PARAMS)
    os.makedirs(out_dir, exist_ok=True)

    if dye_col_map is None:
        dye_col_map = DEFAULT_DYE_COLS

    cy5_df = df.dropna(subset=['Cy5cleanup']).copy()
    if len(cy5_df) == 0:
        print("No Cy5 data available — skipping")
        return

    if 'dye' not in cy5_df.columns:
        cy5_df['well_col'] = cy5_df['well'].str.extract(r'(\d+)').astype(int)
        cy5_df['dye'] = cy5_df['well_col'].map(dye_col_map)
        cy5_df = cy5_df[cy5_df['dye'].notna()].copy()

    tps = sorted(cy5_df['timepoint'].unique())

    # Cy5/RFP1 ratio
    cy5_df['cy5_rfp_ratio'] = cy5_df['Cy5cleanup'] / cy5_df['RFP1'].replace(0, np.nan)

    metrics = [
        (lambda g: g['Cy5cleanup'].median(), 'Median Cy5cleanup', 'cy5_median'),
        (lambda g: g['Cy5cleanup'].mean(), 'Mean Cy5cleanup', 'cy5_mean'),
        (lambda g: g['Cy5cleanup'].quantile(0.95), '95th %ile Cy5cleanup', 'cy5_p95'),
    ]

    for dye_name in ['Aggfluor', 'JF646']:
        dye_sub = cy5_df[cy5_df['dye'] == dye_name]
        if len(dye_sub) == 0:
            continue

        for metric_func, metric_label, metric_tag in metrics:
            fig, ax = plt.subplots(figsize=(28, 16))
            cond_data = []
            colors_iter = ['#1976D2', '#E53935', '#43A047', '#8E24AA', '#0D47A1', '#B71C1C']
            ci = 0

            for cl in dye_sub['cell_line'].dropna().unique():
                for kind in dye_sub['kind'].dropna().unique():
                    means, sems = get_condition_stats(
                        dye_sub, {'cell_line': cl, 'kind': kind, 'stim_dur': 0},
                        metric_func, tps
                    )
                    cond_data.append({
                        'means': means, 'sems': sems,
                        'label': f'{cl} {kind} 0ms',
                        'color': colors_iter[ci % len(colors_iter)],
                        'marker': 'o',
                    })
                    ci += 1

            plot_timecourse(ax, tps, cond_data,
                          f'{metric_label} — {dye_name}', metric_label)
            plt.tight_layout(rect=[0, 0.12, 1, 1])
            fig.savefig(f'{out_dir}/{metric_tag}_{dye_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {metric_tag}_{dye_name}.png")

    print(f"\nCy5 analysis complete. Output in {out_dir}/")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='FRET Analysis Pipeline')
    parser.add_argument('--csvs', required=True, help='Path to CSVS directory')
    parser.add_argument('--out', required=True, help='Output directory for plots')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help=f'Donor bleedthrough (default: {DEFAULT_ALPHA})')
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA, help=f'Direct excitation (default: {DEFAULT_DELTA})')
    parser.add_argument('--G', type=float, default=DEFAULT_G, help=f'G-factor (default: {DEFAULT_G})')
    parser.add_argument('--cy5', action='store_true', help='Run Cy5cleanup analysis')
    parser.add_argument('--no-fret', action='store_true', help='Skip standard FRET plots')
    parser.add_argument('--save-csv', action='store_true', help='Save merged CSV')

    args = parser.parse_args()

    # Load and process
    df = load_and_merge_csvs(args.csvs)
    df = compute_fret(df, alpha=args.alpha, delta=args.delta, G=args.G)
    df = add_metadata(df, dye_col_map=DEFAULT_DYE_COLS if args.cy5 else None)

    if args.save_csv:
        csv_path = f'{args.out}/fret_cells_merged.csv'
        os.makedirs(args.out, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved merged CSV: {csv_path}")

    # Run analyses
    if not args.no_fret:
        run_standard_fret_analysis(df, args.out)

    if args.cy5:
        run_cy5_analysis(df, args.out)

    print("\n=== Pipeline complete ===")


if __name__ == '__main__':
    main()
