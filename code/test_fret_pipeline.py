#!/usr/bin/env python3
"""
Unit tests for FRET analysis pipeline.

Tests:
1. FRET computation correctness (Fc, E values)
2. CSV merge integrity (no lost cells, correct columns)
3. Normalization math (ΔF/F, Z-score)

Run: pytest test_fret_pipeline.py -v
"""
import pytest
import pandas as pd
import numpy as np

# ============================================================
# FRET COMPUTATION CONSTANTS
# ============================================================
ALPHA = 0.536
DELTA = 0.350
G = 0.65


# ============================================================
# 1. FRET COMPUTATION TESTS
# ============================================================
class TestFcComputation:
    """Test corrected FRET signal: Fc = IDA - α·IDD - δ·IAA"""

    def test_fc_basic(self):
        """Fc with known values."""
        IDD, IDA, IAA = 1000.0, 800.0, 500.0
        Fc = IDA - ALPHA * IDD - DELTA * IAA
        expected = 800.0 - 0.536 * 1000.0 - 0.350 * 500.0
        assert Fc == pytest.approx(expected, rel=1e-10)
        assert Fc == pytest.approx(89.0, rel=1e-10)

    def test_fc_zero_signal(self):
        """Fc should be zero when IDA equals bleedthrough."""
        IDD, IAA = 1000.0, 500.0
        IDA = ALPHA * IDD + DELTA * IAA  # Exact bleedthrough
        Fc = IDA - ALPHA * IDD - DELTA * IAA
        assert Fc == pytest.approx(0.0, abs=1e-10)

    def test_fc_negative_when_no_fret(self):
        """Fc can be negative (noise or overcorrection)."""
        IDD, IDA, IAA = 1000.0, 100.0, 500.0
        Fc = IDA - ALPHA * IDD - DELTA * IAA
        assert Fc < 0, "Fc should be negative when IDA < bleedthrough"

    def test_fc_increases_with_ida(self):
        """Fc should increase monotonically with IDA (all else equal)."""
        IDD, IAA = 1000.0, 500.0
        Fc_low = 500.0 - ALPHA * IDD - DELTA * IAA
        Fc_high = 1500.0 - ALPHA * IDD - DELTA * IAA
        assert Fc_high > Fc_low

    def test_fc_vectorized(self):
        """Fc computation works on pandas Series."""
        df = pd.DataFrame({
            'CFP-DMD': [1000.0, 2000.0, 500.0],
            'CFP-FRET': [800.0, 1200.0, 300.0],
            'YFP-DMD': [500.0, 600.0, 200.0],
        })
        df['Fc'] = df['CFP-FRET'] - ALPHA * df['CFP-DMD'] - DELTA * df['YFP-DMD']
        expected = [
            800.0 - 0.536 * 1000.0 - 0.350 * 500.0,
            1200.0 - 0.536 * 2000.0 - 0.350 * 600.0,
            300.0 - 0.536 * 500.0 - 0.350 * 200.0,
        ]
        np.testing.assert_allclose(df['Fc'].values, expected, rtol=1e-10)


class TestFretEfficiency:
    """Test FRET efficiency: E = Fc / (Fc + G·IDD)"""

    def test_e_basic(self):
        """E with known Fc and IDD."""
        Fc, IDD = 100.0, 1000.0
        E = Fc / (Fc + G * IDD)
        expected = 100.0 / (100.0 + 0.65 * 1000.0)
        assert E == pytest.approx(expected, rel=1e-10)

    def test_e_range_positive_fc(self):
        """E should be between 0 and 1 for positive Fc and IDD."""
        for Fc in [1, 10, 100, 1000, 10000]:
            for IDD in [100, 1000, 5000]:
                E = Fc / (Fc + G * IDD)
                assert 0 < E < 1, f"E={E} out of range for Fc={Fc}, IDD={IDD}"

    def test_e_zero_fc(self):
        """E should be 0 when Fc is 0."""
        E = 0.0 / (0.0 + G * 1000.0)
        assert E == pytest.approx(0.0, abs=1e-10)

    def test_e_high_fret(self):
        """E approaches 1 for very high Fc relative to IDD."""
        Fc, IDD = 1e6, 1.0
        E = Fc / (Fc + G * IDD)
        assert E > 0.99

    def test_e_negative_fc(self):
        """E can be negative or >1 for negative Fc — this is expected noise."""
        Fc, IDD = -100.0, 1000.0
        E = Fc / (Fc + G * IDD)
        assert E < 0, "Negative Fc should give negative E"

    def test_e_undefined_zero_denominator(self):
        """E is undefined when Fc + G·IDD = 0."""
        Fc = -G * 1000.0  # Makes denominator zero
        IDD = 1000.0
        denom = Fc + G * IDD
        assert denom == pytest.approx(0.0, abs=1e-10)


class TestNormalizedFc:
    """Test donor-normalized Fc: Fc_norm = Fc / IDD"""

    def test_fc_norm_basic(self):
        Fc, IDD = 89.0, 1000.0
        Fc_norm = Fc / IDD
        assert Fc_norm == pytest.approx(0.089, rel=1e-10)

    def test_fc_norm_zero_idd(self):
        """Should handle zero IDD gracefully (NaN or inf)."""
        Fc, IDD = 89.0, 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            Fc_norm = Fc / IDD if IDD != 0 else np.nan
        assert np.isnan(Fc_norm)


# ============================================================
# 2. CSV MERGE INTEGRITY TESTS
# ============================================================
class TestCSVMergeIntegrity:
    """Test that CSV merging preserves data integrity."""

    @pytest.fixture
    def merged_fret_csv(self):
        """Load the actual merged FRET CSV if available."""
        path = '/sessions/epic-zen-dirac/fret_results/csv_fret/fret_cells_merged.csv'
        try:
            return pd.read_csv(path, nrows=10000)
        except FileNotFoundError:
            pytest.skip("Merged FRET CSV not available")

    @pytest.fixture
    def merged_cy5_csv(self):
        """Load the actual merged Cy5 CSV if available."""
        path = '/sessions/epic-zen-dirac/fret_results/csv_fret/fret_cells_cy5_alltp.csv'
        try:
            return pd.read_csv(path, nrows=10000)
        except FileNotFoundError:
            pytest.skip("Merged Cy5 CSV not available")

    def test_fret_required_columns(self, merged_fret_csv):
        """Merged FRET CSV must have all required channel columns."""
        required = ['CFP-DMD', 'CFP-FRET', 'YFP-DMD', 'RFP1',
                     'well', 'tile', 'timepoint', 'name', 'kind']
        for col in required:
            assert col in merged_fret_csv.columns, f"Missing required column: {col}"

    def test_cy5_required_columns(self, merged_cy5_csv):
        """Merged Cy5 CSV must have Cy5cleanup column."""
        required = ['Cy5cleanup', 'RFP1', 'well', 'timepoint']
        for col in required:
            assert col in merged_cy5_csv.columns, f"Missing required column: {col}"

    def test_no_duplicate_cells_per_timepoint(self, merged_fret_csv):
        """Each cell should appear at most once per timepoint."""
        if 'celldata_id' in merged_fret_csv.columns:
            dupes = merged_fret_csv.groupby(['celldata_id', 'timepoint']).size()
            assert (dupes <= 1).all(), "Found duplicate celldata_id within same timepoint"

    def test_channel_values_non_negative(self, merged_fret_csv):
        """Raw channel intensities should be non-negative."""
        for ch in ['CFP-DMD', 'CFP-FRET', 'YFP-DMD', 'RFP1']:
            if ch in merged_fret_csv.columns:
                valid = merged_fret_csv[ch].dropna()
                assert (valid >= 0).all(), f"{ch} has negative values"

    def test_no_all_nan_channels(self, merged_fret_csv):
        """No channel should be entirely NaN (indicates broken merge)."""
        for ch in ['CFP-DMD', 'CFP-FRET', 'YFP-DMD', 'RFP1']:
            if ch in merged_fret_csv.columns:
                assert not merged_fret_csv[ch].isna().all(), f"{ch} is all NaN"

    def test_timepoints_sequential(self, merged_fret_csv):
        """Timepoints should be sequential integers."""
        tps = sorted(merged_fret_csv['timepoint'].unique())
        # Check they're integers
        assert all(isinstance(t, (int, np.integer)) for t in tps)
        # Check no huge gaps (max gap should be 1)
        gaps = np.diff(tps)
        assert gaps.max() <= 1, f"Non-sequential timepoints: max gap = {gaps.max()}"

    def test_well_format(self, merged_fret_csv):
        """Wells should match pattern like A1, B10, etc."""
        wells = merged_fret_csv['well'].dropna().unique()
        import re
        pattern = re.compile(r'^[A-Z]\d{1,2}$')
        for w in wells:
            assert pattern.match(w), f"Unexpected well format: {w}"

    def test_stim_mapping_coverage(self, merged_fret_csv):
        """All tiles should map to a known stimulation duration."""
        STIM = {1:0, 2:500, 3:2000, 4:10000, 5:10000, 6:0, 7:500, 8:2000,
                9:2000, 10:10000, 11:0, 12:500, 13:500, 14:2000, 15:10000, 16:0}
        tiles = merged_fret_csv['tile'].dropna().unique()
        for t in tiles:
            assert t in STIM, f"Tile {t} not in stimulation mapping"


# ============================================================
# 3. NORMALIZATION MATH TESTS
# ============================================================
class TestDeltaFF:
    """Test ΔF/F normalization: (V(t) - V(t0)) / V(t0)"""

    def test_dff_at_baseline(self):
        """ΔF/F at t0 should be 0."""
        t0_val = 100.0
        dff = (t0_val - t0_val) / t0_val
        assert dff == pytest.approx(0.0, abs=1e-10)

    def test_dff_doubled(self):
        """Signal doubling should give ΔF/F = 1.0."""
        t0_val = 100.0
        t1_val = 200.0
        dff = (t1_val - t0_val) / t0_val
        assert dff == pytest.approx(1.0, rel=1e-10)

    def test_dff_halved(self):
        """Signal halving should give ΔF/F = -0.5."""
        t0_val = 100.0
        t1_val = 50.0
        dff = (t1_val - t0_val) / t0_val
        assert dff == pytest.approx(-0.5, rel=1e-10)

    def test_dff_zero_baseline(self):
        """ΔF/F should be NaN/inf for zero baseline."""
        t0_val = 0.0
        t1_val = 100.0
        with np.errstate(divide='ignore', invalid='ignore'):
            dff = (t1_val - t0_val) / t0_val if t0_val != 0 else np.nan
        assert np.isnan(dff)

    def test_dff_per_well(self):
        """ΔF/F computed per well independently."""
        data = pd.DataFrame({
            'well': ['A1', 'A1', 'A1', 'B1', 'B1', 'B1'],
            'timepoint': [0, 1, 2, 0, 1, 2],
            'val': [100, 150, 200, 50, 100, 75],
        })
        results = []
        for w in data['well'].unique():
            wdata = data[data['well'] == w].set_index('timepoint')['val']
            t0 = wdata.loc[0]
            for tp in [0, 1, 2]:
                results.append({'well': w, 'timepoint': tp,
                               'dff': (wdata.loc[tp] - t0) / t0})
        result_df = pd.DataFrame(results)

        a1_dff = result_df[result_df['well'] == 'A1']['dff'].values
        np.testing.assert_allclose(a1_dff, [0.0, 0.5, 1.0], rtol=1e-10)

        b1_dff = result_df[result_df['well'] == 'B1']['dff'].values
        np.testing.assert_allclose(b1_dff, [0.0, 1.0, 0.5], rtol=1e-10)


class TestZScore:
    """Test Z-score normalization: Z = (V - μ) / σ"""

    def test_zscore_basic(self):
        """Z-score of the mean should be 0."""
        vals = np.array([10, 20, 30, 40, 50], dtype=float)
        mu, sigma = vals.mean(), vals.std()
        z_mean = (mu - mu) / sigma
        assert z_mean == pytest.approx(0.0, abs=1e-10)

    def test_zscore_one_std_above(self):
        """Value one σ above mean should have Z = 1 (with ddof=0)."""
        vals = np.array([10, 20, 30, 40, 50], dtype=float)
        mu, sigma = vals.mean(), vals.std()
        z = (mu + sigma - mu) / sigma
        assert z == pytest.approx(1.0, rel=1e-10)

    def test_zscore_preserves_ordering(self):
        """Z-scoring should preserve relative ordering."""
        vals = np.array([5, 10, 15, 20, 25], dtype=float)
        mu, sigma = vals.mean(), vals.std()
        z_vals = (vals - mu) / sigma
        assert all(z_vals[i] < z_vals[i + 1] for i in range(len(z_vals) - 1))

    def test_zscore_per_dye(self):
        """Z-scoring per dye group should normalize each independently."""
        data = pd.DataFrame({
            'dye': ['Agg', 'Agg', 'Agg', 'JF', 'JF', 'JF'],
            'val': [10, 20, 30, 1000, 2000, 3000],
        })
        for dye in ['Agg', 'JF']:
            sub = data[data['dye'] == dye]['val']
            mu, sigma = sub.mean(), sub.std()
            z_vals = (sub - mu) / sigma
            assert z_vals.mean() == pytest.approx(0.0, abs=1e-10)

    def test_zscore_zero_std(self):
        """Z-score should handle zero std gracefully."""
        vals = np.array([5.0, 5.0, 5.0])
        mu, sigma = vals.mean(), vals.std()
        assert sigma == 0.0
        # Should not compute Z-score with zero std


class TestDeltaZScore:
    """Test ΔZ-score: Z(t) - Z(t0) per well."""

    def test_delta_zscore_at_baseline(self):
        """ΔZ at first timepoint should be 0 for all wells."""
        z_vals = {'A1': {0: 0.5, 1: 1.2, 2: 1.8},
                  'B1': {0: -0.3, 1: 0.1, 2: 0.4}}
        for well, tps in z_vals.items():
            t0 = tps[0]
            dz_t0 = tps[0] - t0
            assert dz_t0 == pytest.approx(0.0, abs=1e-10)

    def test_delta_zscore_change(self):
        """ΔZ should reflect Z-score change from baseline."""
        z_t0 = 0.5
        z_t1 = 1.2
        z_t2 = -0.1
        assert (z_t1 - z_t0) == pytest.approx(0.7, rel=1e-10)
        assert (z_t2 - z_t0) == pytest.approx(-0.6, rel=1e-10)


# ============================================================
# 4. CALIBRATION FACTOR SANITY CHECKS
# ============================================================
class TestCalibrationFactors:
    """Sanity checks on calibration values."""

    def test_alpha_in_range(self):
        """α (donor bleedthrough) should be between 0 and 1."""
        assert 0 < ALPHA < 1, f"α={ALPHA} out of expected range"

    def test_delta_in_range(self):
        """δ (direct excitation) should be between 0 and 1."""
        assert 0 < DELTA < 1, f"δ={DELTA} out of expected range"

    def test_g_factor_in_range(self):
        """G-factor should be between 0 and 2 (typical range)."""
        assert 0 < G < 2, f"G={G} out of expected range"

    def test_alpha_from_calibration(self):
        """α should match the calibration result."""
        import json
        try:
            with open('/sessions/epic-zen-dirac/mnt/TM Tau-FRET/tau_fret_pipeline/calibration_results/fret_calibration.json') as f:
                cal = json.load(f)
            assert ALPHA == pytest.approx(cal['alpha'], rel=1e-3)
        except FileNotFoundError:
            pytest.skip("Calibration file not available")

    def test_delta_from_calibration(self):
        """δ should match the calibration result."""
        import json
        try:
            with open('/sessions/epic-zen-dirac/mnt/TM Tau-FRET/tau_fret_pipeline/calibration_results/fret_calibration.json') as f:
                cal = json.load(f)
            assert DELTA == pytest.approx(cal['delta'], rel=1e-2)
        except FileNotFoundError:
            pytest.skip("Calibration file not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
