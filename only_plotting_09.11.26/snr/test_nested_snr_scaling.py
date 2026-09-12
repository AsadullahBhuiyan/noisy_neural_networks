"""Small synthetic checks; no real models or downloaded data required."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import analyze_nested_snr_scaling as analysis


def fixture(root, n=10, side=2, noise="replacement"):
    d = side**2
    folder = root / f"mlp_ensemble__{noise}__N={n}__d={side}x{side}"
    folder.mkdir()
    cfg = dict(dataset="synthetic", noise_type=noise, noise_probability=.97,
               num_classes=3, img_hw=[side, side], num_noise_datasets=3,
               num_models_per_noise_dataset=2, test_outputs_filename="test_outputs.csv.gz")
    (folder / "metadata.json").write_text(json.dumps(dict(
        config=cfg, dataset="synthetic", num_train=n, num_test=7, feature_dim=d,
        test_is_clean=True)))
    rows = []
    for k in range(3):
        for m in range(2):
            for t in range(7):
                # The +/- init term cancels; dataset offsets have sample var 1/(Nd).
                z = t + 1 + (k - 1) / np.sqrt(n*d) + (2*m - 1)*.25
                row = dict(test_index=t, noise_dataset_index=k, init_index=m,
                           model_index=k*2+m, true_label=t % 3, train_size=n, feature_dim=d)
                row.update({f"logit_{c}": z if c == t % 3 else .1*z for c in range(3)})
                rows.append(row)
    pd.DataFrame(rows).to_csv(folder / "test_outputs.csv.gz", index=False)
    return folder


class SNRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = fixture(self.root)
        self.run = analysis.load_run(self.folder)
        self.indices = np.array([0, 2, 6])
        self.args = argparse.Namespace(output="true_class", class_index=1, chunk_rows=11)

    def tearDown(self):
        self.temp.cleanup()

    def test_nested_formula_and_outputs(self):
        for mode in ("true_class", "margin", "class"):
            self.args.output = mode
            values, labels = analysis.read_selected(self.run, self.indices, self.args)
            stats = analysis.snr_statistics(values)
            factor = (np.full(3, .9) if mode == "margin" else
                      np.where(labels == 1, 1., .1) if mode == "class" else np.ones(3))
            np.testing.assert_allclose(stats["signal_mean"], (self.indices + 1) * factor)
            np.testing.assert_allclose(stats["dataset_variance"], factor**2 / 40)
            np.testing.assert_allclose(stats["snr"], (self.indices + 1)**2 * 40)
            np.testing.assert_allclose(stats["estimated_init_variance_in_dataset_mean"], factor**2 * .0625)

    def test_missing_duplicate_and_wrong_label_rejected(self):
        path = self.run["path"]
        original = pd.read_csv(path)
        for kind in ("missing", "duplicate", "label"):
            changed = original.copy()
            if kind == "missing":
                changed = changed.iloc[1:]
            elif kind == "duplicate":
                changed = pd.concat([changed, changed.iloc[[0]]])
            else:
                changed.loc[0, "true_label"] = 1
            changed.to_csv(path, index=False)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                analysis.read_selected(self.run, self.indices, self.args)

    def test_plain_csv_and_reproducible_sampling(self):
        path = self.run["path"]
        with gzip.open(path, "rt") as f:
            path.with_suffix("").write_text(f.read())
        path.unlink()
        run = analysis.load_run(self.folder)
        analysis.read_selected(run, self.indices, self.args)
        a = analysis.sampled_indices("synthetic", 10000, 500, 12)
        np.testing.assert_array_equal(a, analysis.sampled_indices("synthetic", 10000, 500, 12))
        self.assertEqual(len(np.unique(a)), 500)

    def test_zero_cases(self):
        v = np.zeros((3, 2, 3))
        v[:, :, 1] = 1
        v[:, :, 2] = np.array([-1, 0, 1])[:, None]
        stats = analysis.snr_statistics(v)
        self.assertEqual(list(stats["status"]),
                         ["zero_signal_and_variance", "zero_variance", "zero_signal"])
        self.assertFalse(stats["valid_log"].any())

    def test_full_cli_recovers_scaling_and_replots(self):
        for noise in ("replacement", "additive_gaussian"):
            for n in (10, 30):
                for side in (2, 4):
                    if (noise, n, side) != ("replacement", 10, 2):
                        fixture(self.root, n, side, noise)
        out = self.root / "results"
        cmd = [sys.executable, str(Path(analysis.__file__)), "--root", str(self.root),
               "--outdir", str(out), "--num-test-points", "5", "--chunk-rows", "11"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        points = pd.read_csv(out / "snr_points.csv")
        self.assertEqual(len(points), 40)
        self.assertEqual(points.groupby("folder").test_index.apply(tuple).nunique(), 1)
        fits = pd.read_csv(out / "snr_scaling_fits.csv")
        self.assertEqual(len(fits), 2)
        np.testing.assert_allclose(fits.slope, 1, atol=1e-12)
        np.testing.assert_allclose(fits.separate_N_exponent, 1, atol=1e-12)
        np.testing.assert_allclose(fits.separate_d_exponent, 1, atol=1e-12)
        self.assertEqual(len(list(out.glob("*.png"))), 2)
        self.assertEqual(len(list(out.glob("*.pdf"))), 2)
        result = subprocess.run([sys.executable, str(Path(analysis.__file__)),
                                 "--plot-only", str(out / "snr_points.csv")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
