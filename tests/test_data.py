"""The data layer: ingest and verify-data, profile-data, make-synthetic, convert and the store dataset, and
estimate. Everything runs in a temporary repository on files the test writes."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.support import TempRepo, load_catalog, needs_module


def cli(*argv) -> int:
    from sweepkit import cli as module

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return module.main(list(argv))


class TestIngest(unittest.TestCase):
    def test_resumable_copy_manifest_and_verify(self):
        load_catalog()
        with TempRepo() as root, tempfile.TemporaryDirectory() as d:
            source = Path(d) / "received"
            for i in range(12):
                p = source / f"part{i % 3}" / f"file{i}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"row_id,value\nr{i},{i}\n", encoding="utf-8")
            codes = []
            for _ in range(60):
                code = cli("ingest", "--name", "batch1", "--source", str(source), "--max-seconds", "0")
                codes.append(code)
                if code == 0:
                    break
            self.assertEqual(codes[-1], 0)
            self.assertIn(3, codes)                      # zero budget: it stopped and continued
            summary = json.loads((root / "data" / "manifests" / "batch1.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_files"], 12)
            self.assertEqual((root / "data" / "received" / "batch1" / "part1" / "file4.csv").read_text(),
                             "row_id,value\nr4,4\n")
            self.assertEqual(cli("ingest", "--name", "batch1", "--source", str(source), "--max-seconds", "5"), 0)
            self.assertEqual(cli("verify-data", "--name", "batch1"), 0)
            (root / "data" / "received" / "batch1" / "part0" / "file0.csv").write_text("changed", encoding="utf-8")
            self.assertEqual(cli("verify-data", "--name", "batch1"), 1)

    def test_refuses_a_folder_inside_data(self):
        from sweepkit.data.ingest import ingest

        with TempRepo() as root:
            inside = root / "data" / "x"
            inside.mkdir(parents=True)
            (inside / "a.txt").write_text("a")
            with self.assertRaisesRegex(ValueError, "inside data"):
                ingest("again", inside, 5)


class TestProfile(unittest.TestCase):
    def test_folder_leftovers_and_collisions(self):
        from sweepkit.data.profile import profile

        with TempRepo() as root:
            folder = root / "data" / "raw"
            (folder / ".ipynb_checkpoints").mkdir(parents=True)
            (folder / ".ipynb_checkpoints" / "a-checkpoint.csv").write_text("x")
            (folder / "a.csv").write_text("row_id,v\n1,2\n")
            (folder / "A.CSV").write_text("row_id,v\n1,2\n")
            (folder / "empty.csv").write_text("")
            (folder / "~$sheet.xlsx").write_text("lock")
            (folder / ".DS_Store").write_bytes(b"\0")
            report = profile("raw", None, None, None, None, 3)
            self.assertEqual(report["kind"], "folder")
            self.assertEqual(report["empty_files"], ["empty.csv"])
            self.assertIn(".ipynb_checkpoints/", report["leftovers"])
            self.assertIn("~$sheet.xlsx", report["leftovers"])
            self.assertIn(".DS_Store", report["leftovers"])
            self.assertEqual([sorted(c) for c in report["case_collisions"]], [["A.CSV", "a.csv"]])

    def test_table_columns_ids_and_join(self):
        from sweepkit.data.profile import profile

        with TempRepo() as root:
            (root / "data").mkdir()
            rows = ["row_id,age,site,score"]
            for i in range(20):
                score = "7.5" if i != 3 else '"7,5"'          # one value with a decimal comma, quoted
                age = "" if i == 5 else str(20 + i)
                rows.append(f"r{i if i != 9 else 8},{age},s{i % 2},{score}")
            rows.append("r99,30,s1,8,5")                      # unquoted: one field too many
            (root / "data" / "t.csv").write_text("\n".join(rows) + "\n")
            (root / "data" / "labels.csv").write_text("id,label\n" + "\n".join(f"r{i},x" for i in range(2, 25)) + "\nr2,y\n")
            report = profile("t.csv", "row_id", "labels.csv", "id", None, 2)
            age = report["by_column"]["age"]
            self.assertEqual((age["type"], age["missing"], age["min"], age["max"]), ("number", 1, 20.0, 39.0))
            self.assertIn("score", report["nearly_numeric_text_columns"])
            self.assertEqual(report["by_column"]["site"]["distinct"], 2)
            self.assertEqual(report["row_id"]["repeated"], 1)
            self.assertEqual(report["rows"], 20)
            self.assertEqual(report["bad_lines"]["count"], 1)
            self.assertEqual(report["bad_lines"]["first"], [{"line": 22, "fields": 5}])
            join = report["join"]
            self.assertEqual(join["rows_without_label"], 2)        # r0, r1
            self.assertEqual(join["labels_without_row"], 6)        # r9 (r8 repeated in rows) and r19..r24 minus r19
            self.assertEqual(join["labels_given_twice"], 1)

    def test_cli_saves_a_report(self):
        load_catalog()
        with TempRepo() as root:
            (root / "data").mkdir()
            (root / "data" / "t.csv").write_text("a,b\n1,x\n2,y\n")
            self.assertEqual(cli("profile-data", "--target", "t.csv", "--row-id", "none", "--labels", "none",
                                 "--label-key", "none", "--max-rows", "all", "--top-values", "3", "--hide", "none"), 0)
            self.assertEqual(len(list((root / "artifacts" / "checks").iterdir())), 1)

    def test_a_hidden_column_shows_no_values(self):
        load_catalog()
        from sweepkit.data.profile import profile, render

        with TempRepo() as root:
            (root / "data").mkdir()
            rows = ["id,x,truth"] + [f"r{i},{i * 1.5},{'grp_alpha' if i % 3 else 'grp_beta'}" for i in range(30)]
            (root / "data" / "t.csv").write_text("\n".join(rows) + "\n")
            report = profile("t.csv", "id", None, None, None, 5, ("truth",))
            self.assertEqual(report["by_column"]["truth"], {"hidden": True, "missing": 0, "present": 30,
                                                             "constant": False})
            text = render(report) + json.dumps(report)
            self.assertNotIn("grp_", text)                           # no value of the hidden column anywhere
            self.assertIn("truth: values hidden (--hide); 30 present, 0 missing", render(report))
            self.assertIn("range 0 to 43.5", render(report))         # the other columns as usual
            self.assertEqual(cli("profile-data", "--target", "t.csv", "--row-id", "id", "--labels", "none",
                                 "--label-key", "none", "--max-rows", "all", "--top-values", "3", "--hide", "truth"), 0)
            saved = next((root / "artifacts" / "checks").iterdir())
            self.assertNotIn("grp_", (saved / "report.json").read_text() + (saved / "report.txt").read_text())
            with self.assertRaises(ValueError):
                profile("t.csv", "id", None, None, None, 5, ("no_such_column",))


class TestSyntheticConvertStore(unittest.TestCase):
    def test_same_settings_same_bytes_and_written_once(self):
        load_catalog()
        args = ["make-synthetic", "--generator", "mixed", "--n-rows", "500", "--n-features", "6", "--n-groups", "3",
                "--noise", "0.5", "--seed", "4", "--name", "m", "--chunk-rows", "120", "--sample-rows", "40"]
        digests = []
        for _ in range(2):
            with TempRepo() as root:
                self.assertEqual(cli(*args), 0)
                record = json.loads((root / "data" / "synthetic" / "m.json").read_text())
                digests.append((record["csv"]["sha256"], record["sample"]["sha256"]))
                sample = (root / "data" / "synthetic" / "m_sample.csv").read_text().strip().splitlines()
                self.assertEqual(len(sample), 41)
                with self.assertRaises(FileExistsError):
                    from sweepkit.data.make import write_synthetic

                    write_synthetic("mixed", 500, 6, 3, 0.5, 4, "m", 120, 40, lambda m: None)
        self.assertEqual(digests[0], digests[1])

    def test_convert_to_store_and_run_a_job_on_it(self):
        load_catalog()
        with TempRepo() as root:
            self.assertEqual(cli("make-synthetic", "--generator", "blobs", "--n-rows", "3000", "--n-features", "5",
                                 "--n-groups", "3", "--noise", "0.5", "--seed", "1", "--name", "b", "--chunk-rows",
                                 "1000", "--sample-rows", "100"), 0)
            self.assertEqual(cli("convert", "--source", "synthetic/b.csv", "--name", "b_store", "--sep", "comma",
                                 "--row-id", "row_id", "--features", "*", "--exclude", "none", "--keep", "truth",
                                 "group", "published_split",
                                 "--chunk-rows", "1000", "--to", "store"), 0)
            from sweepkit.data.convert import read_store

            store = read_store("b_store")
            self.assertEqual(store["values"].shape, (3000, 5))
            import numpy as np

            original = np.loadtxt(root / "data" / "synthetic" / "b.csv", delimiter=",", skiprows=1,
                                  usecols=range(1, 6), dtype=np.float32)
            self.assertTrue(np.array_equal(np.asarray(store["values"]), original))

    @needs_module("pyarrow")
    def test_convert_to_parquet(self):
        load_catalog()
        with TempRepo() as root:
            cli("make-synthetic", "--generator", "blobs", "--n-rows", "2000", "--n-features", "3", "--n-groups", "2",
                "--noise", "0.5", "--seed", "1", "--name", "p", "--chunk-rows", "1000", "--sample-rows", "10")
            self.assertEqual(cli("convert", "--source", "synthetic/p.csv", "--name", "p_parquet", "--sep", "comma",
                                 "--row-id", "row_id", "--features", "*", "--exclude", "none", "--keep", "truth",
                                 "group", "published_split",
                                 "--chunk-rows", "1000", "--to", "parquet"), 0)
            import pandas as pd

            frame = pd.read_parquet(root / "data" / "stores" / "p_parquet" / "table.parquet")
            # the row IDs, the three inputs and the three kept columns (the test expected 5 until it first ran, on
            # the node, 2026-10-27)
            self.assertEqual(list(frame.columns), ["row_id", "f0", "f1", "f2", "truth", "group", "published_split"])
            self.assertEqual(frame.shape, (2000, 7))


class TestEstimate(unittest.TestCase):
    def test_parts_grow_and_predictions_are_written(self):
        load_catalog()
        from tests.sweep_fixtures import cluster_base, sweep

        with TempRepo() as root:
            base = cluster_base()
            base["data"]["n_rows"] = 4000
            s = sweep(base, [{"name": "base", "kind": "base", "priority": 1, "seeds": [1, 2, 3]}], sweep_id="e1")
            (root / "e1.json").write_text(json.dumps(s))
            from sweepkit.data.estimate import estimate

            result = estimate(root / "e1.json", "base", "cpu", [0.05, 0.1, 0.2], 2, lambda m: None)
            self.assertEqual([p["rows"] for p in result["parts"]], [160, 320, 640])
            self.assertEqual(result["full_rows"], 3200)
            self.assertEqual(result["jobs_in_sweep"], 6)
            self.assertGreater(result["per_job"]["seconds"], 0)
            self.assertTrue((Path(result["folder"]) / "report.txt").is_file())
            with self.assertRaises(ValueError):
                estimate(root / "e1.json", "base", "cpu", [0.2, 0.1], 1, lambda m: None)


if __name__ == "__main__":
    unittest.main()
