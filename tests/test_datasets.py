"""The datasets: what each one loads (rows, row IDs, per-row columns, kind, feature names, inputs) and how it refuses
bad settings or missing files with a message that says what to change.

"synthetic" needs no file. The file-based datasets read files written inside a temporary repository
(tests.support.TempRepo), so nothing lands in the tree: "table" reads a CSV written by make-synthetic and small
hand-written TSV and Excel files, "store" a store made by convert, "items" and "graph" the folders make-example-data
writes (plus a few hand-written items: text, a CSV array, a 16-bit wav), and "anndata" an .h5ad file when anndata is
installed (the singlecell profile; the test skips elsewhere, and fails where SWEEPKIT_REQUIRE names anndata).
"""
import types
import unittest

from tests.support import TempRepo, load_catalog, needs_module


def load(raw: dict):
    """The Data a data section loads, as a job's runner loads it."""
    from sweepkit.registry import DATASETS

    entry, settings = DATASETS.validate_section(raw, "data")
    return entry.target().load(settings)


def check(raw: dict) -> None:
    """The dataset's own check of its settings (the hook check_job calls with the whole job)."""
    from sweepkit.registry import DATASETS

    entry, settings = DATASETS.validate_section(raw, "data")
    hook = getattr(entry.target, "check_job", None)
    if hook is not None:
        hook(types.SimpleNamespace(data=settings))


def quiet(message: str) -> None:
    pass


def table_section(path: str, **changes) -> dict:
    return dict({"name": "table", "path": path, "format": "csv", "sheet_or_key": None, "row_id": "row_id",
                 "features": ["*"], "exclude": [], "keep": ["truth", "group", "published_split"],
                 "categorical": "codes", "filter_column": None, "filter_values": []}, **changes)


class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def section(self, generator: str, **changes) -> dict:
        return dict({"name": "synthetic", "generator": generator, "n_rows": 120, "n_features": 4, "n_groups": 3,
                     "noise": 0.5, "seed": 1}, **changes)

    def test_tabular_rows_ids_and_columns(self):
        import numpy as np

        from sweepkit.data.synthetic import make

        data = load(self.section("blobs"))
        self.assertEqual((data.kind, len(data.table), data.x.shape), ("tabular", 120, (120, 4)))
        self.assertEqual(data.table.ids[:2].tolist(), ["r0000000", "r0000001"])
        self.assertEqual(data.feature_names, ("f0", "f1", "f2", "f3"))
        self.assertEqual(sorted(data.table.columns), ["group", "published_split", "truth"])
        made = make("blobs", 120, 4, 3, 0.5, 1)
        self.assertTrue(np.array_equal(data.x, made["x"]))
        self.assertEqual(data.fingerprint, {"synthetic": {k: v for k, v in self.section("blobs").items()
                                                          if k != "name"}})
        again = load(self.section("blobs"))
        self.assertTrue(np.array_equal(again.x, data.x))                 # the same settings give the same rows

    def test_mixed_columns_keep_their_category_codes(self):
        data = load(self.section("mixed", n_features=6))
        self.assertEqual(data.feature_names, ("num_0", "num_1", "num_2", "cat_0", "cat_1", "cat_2"))
        self.assertEqual(data.categorical, {3: ["a", "b", "c", "d"], 4: ["a", "b", "c", "d"], 5: ["a", "b", "c", "d"]})

    def test_items_and_graph_kinds(self):
        sequences = load(self.section("sequences", n_features=3))
        self.assertEqual((sequences.kind, len(sequences.items)), ("sequence", 120))
        self.assertEqual(sequences.items.get(0).shape[1], 3)
        self.assertEqual(sequences.items.length(0), sequences.items.get(0).shape[0])
        self.assertIsNone(sequences.x)
        text = load(self.section("text"))
        self.assertEqual(text.kind, "text")
        self.assertEqual(text.items.length(0), len(text.items.get(0)))
        self.assertEqual(load(self.section("images", n_features=8)).items.get(0).shape, (1, 8, 8))
        self.assertEqual(load(self.section("audio", n_rows=6)).kind, "audio")
        graph = load(self.section("graph"))
        self.assertEqual((graph.kind, graph.x.shape[0], graph.graph.n_nodes), ("graph", 120, 120))
        self.assertGreater(graph.graph.edges.shape[1], 0)

    def test_bad_settings_are_refused(self):
        for changes in ({"n_rows": 3}, {"n_features": 0}, {"n_groups": 0}, {"noise": -1.0}):
            with self.assertRaisesRegex(ValueError, "n_rows >= 4", msg=str(changes)):
                check(self.section("blobs", **changes))
        from sweepkit.settings import SettingsError

        with self.assertRaises(SettingsError):
            load(self.section("no_such_generator"))


class TestTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def setUp(self):
        from sweepkit.data.make import write_synthetic

        self.repo = TempRepo()
        self.root = self.repo.__enter__()
        self.record = write_synthetic("mixed", 200, 6, 3, 0.8, 1, "mixed_table", 50, 20, quiet)

    def tearDown(self):
        self.repo.__exit__(None, None, None)

    def test_a_csv_from_make_synthetic(self):
        import numpy as np
        import pandas as pd

        from sweepkit.data.synthetic import make
        from sweepkit.hashing import sha256_file

        data = load(table_section("synthetic/mixed_table.csv"))
        self.assertEqual((data.kind, len(data.table), data.x.shape), ("tabular", 200, (200, 6)))
        self.assertEqual(data.table.ids[:2].tolist(), ["r000", "r001"])
        self.assertEqual(data.feature_names, ("num_0", "num_1", "num_2", "cat_0", "cat_1", "cat_2"))
        self.assertEqual(data.categorical, {3: ["a", "b", "c", "d"], 4: ["a", "b", "c", "d"], 5: ["a", "b", "c", "d"]})
        made = make("mixed", 200, 6, 3, 0.8, 1)
        self.assertTrue(np.allclose(data.x[:, :3], made["x"][:, :3], rtol=1e-6, atol=1e-6))   # 7 digits in the CSV
        self.assertTrue(np.array_equal(data.x[:, 3:], made["x"][:, 3:]))   # the codes come back as they were made
        for column in ("truth", "group", "published_split"):
            self.assertEqual(data.table.column(column).tolist(), made["columns"][column].tolist())
        path = self.root / "data" / "synthetic" / "mixed_table.csv"
        self.assertEqual(data.fingerprint, {"path": "synthetic/mixed_table.csv",
                                            **dict(zip(("bytes", "sha256"), sha256_file(path)))})
        frame = pd.read_csv(path)
        self.assertEqual(frame.columns.tolist(), self.record["columns"])

    def test_categorical_columns_four_ways(self):
        with self.assertRaisesRegex(ValueError, "'cat_0' is not numeric"):
            load(table_section("synthetic/mixed_table.csv", categorical="error"))
        dropped = load(table_section("synthetic/mixed_table.csv", categorical="drop"))
        self.assertEqual(dropped.feature_names, ("num_0", "num_1", "num_2"))
        self.assertEqual(dropped.categorical, {})
        onehot = load(table_section("synthetic/mixed_table.csv", categorical="onehot"))
        self.assertEqual(onehot.feature_names[3:7], ("cat_0=a", "cat_0=b", "cat_0=c", "cat_0=d"))
        self.assertEqual(onehot.x.shape, (200, 3 + 12))
        self.assertTrue((onehot.x[:, 3:7].sum(axis=1) == 1).all())      # one category per row and column

    def test_choosing_rows_and_columns(self):
        import numpy as np

        data = load(table_section("synthetic/mixed_table.csv", filter_column="published_split",
                                  filter_values=["train"]))
        self.assertEqual(len(data.table), 120)                           # 60 percent of 200
        self.assertEqual(set(data.table.column("published_split").tolist()), {"train"})
        regex = load(table_section("synthetic/mixed_table.csv", features=["re:^num_"], keep=["truth"]))
        self.assertEqual(regex.feature_names, ("num_0", "num_1", "num_2"))
        excluded = load(table_section("synthetic/mixed_table.csv", exclude=["num_1"], categorical="drop",
                                      keep=["truth"]))
        self.assertEqual(excluded.feature_names, ("num_0", "num_2"))      # the text columns are dropped
        numbered = load(table_section("synthetic/mixed_table.csv", row_id=None, categorical="drop"))
        self.assertEqual(numbered.table.ids[:3].tolist(), ["0", "1", "2"])
        self.assertTrue(np.array_equal(numbered.x, load(table_section("synthetic/mixed_table.csv",
                                                                      categorical="drop")).x))

    def test_missing_values_and_other_formats(self):
        import numpy as np
        import pandas as pd

        folder = self.root / "data" / "tables"
        folder.mkdir(parents=True)
        (folder / "small.tsv").write_text("id\ta\tb\tlabel\nx1\t1.5\t\tyes\nx2\t\t2\t\nx3\t3\t4\tno\n",
                                          encoding="utf-8")
        tsv = load(table_section("tables/small.tsv", format="tsv", row_id="id", keep=["label"]))
        self.assertEqual(tsv.table.ids.tolist(), ["x1", "x2", "x3"])
        self.assertTrue(np.isnan(tsv.x[0, 1]) and np.isnan(tsv.x[1, 0]))  # missing inputs stay NaN for impute
        self.assertEqual(tsv.table.column("label").tolist(), ["yes", "", "no"])   # missing kept values become ""
        frame = pd.DataFrame({"id": ["a", "b"], "v": [1.0, 2.0], "label": ["p", "q"]})
        frame.to_excel(folder / "book.xlsx", sheet_name="first", index=False)
        book = load(table_section("tables/book.xlsx", format="excel", sheet_or_key="first", row_id="id",
                                  keep=["label"]))
        self.assertEqual((book.table.ids.tolist(), book.x[:, 0].tolist()), (["a", "b"], [1.0, 2.0]))

    def test_refusals_say_what_to_change(self):
        with self.assertRaisesRegex(ValueError, "must stay inside data/"):
            check(table_section("../outside.csv"))
        with self.assertRaisesRegex(ValueError, "must stay inside data/"):
            check(table_section("/etc/outside.csv"))
        with self.assertRaisesRegex(ValueError, "filter_column together with filter_values"):
            check(table_section("synthetic/mixed_table.csv", filter_column="truth"))
        with self.assertRaisesRegex(ValueError, "takes sheet_or_key None"):
            check(table_section("synthetic/mixed_table.csv", sheet_or_key="Sheet1"))
        with self.assertRaisesRegex(ValueError, "needs sheet_or_key"):
            check(table_section("tables/x.h5", format="hdf5"))
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load(table_section("synthetic/absent.csv"))
        with self.assertRaisesRegex(ValueError, "'nothing' matches no column"):
            load(table_section("synthetic/mixed_table.csv", keep=["nothing"]))
        with self.assertRaisesRegex(ValueError, "no row has truth in"):
            load(table_section("synthetic/mixed_table.csv", filter_column="truth", filter_values=["c9"]))
        (self.root / "data" / "twice.csv").write_text("row_id,a\nr1,1\nr1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "row IDs repeat"):
            load(table_section("twice.csv", keep=[]))


class TestStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def setUp(self):
        from sweepkit.data.convert import convert
        from sweepkit.data.make import write_synthetic

        self.repo = TempRepo()
        self.root = self.repo.__enter__()
        write_synthetic("blobs", 300, 4, 3, 1.0, 2, "b", 100, 30, quiet)
        self.record = convert("synthetic/b.csv", "b_store", ",", "row_id", ("*",), (),
                              ("truth", "group", "published_split"), 1000, "store", quiet)

    def tearDown(self):
        self.repo.__exit__(None, None, None)

    def section(self, **changes) -> dict:
        return dict({"name": "store", "store": "b_store", "features": ["*"], "keep": ["truth", "published_split"],
                     "filter_column": None, "filter_values": []}, **changes)

    def test_rows_come_back_in_the_order_asked(self):
        import numpy as np
        import pandas as pd

        frame = pd.read_csv(self.root / "data" / "synthetic" / "b.csv")
        data = load(self.section())
        self.assertEqual((data.kind, len(data.table), data.x.shape), ("tabular", 300, (300, 4)))
        self.assertEqual(data.feature_names, ("f0", "f1", "f2", "f3"))
        self.assertEqual(data.table.ids.tolist(), frame["row_id"].tolist())
        self.assertEqual(data.table.column("truth").tolist(), frame["truth"].tolist())
        values = frame[["f0", "f1", "f2", "f3"]].to_numpy(dtype=np.float32)
        rows = np.array([205, 3, 77, 3])                                  # unsorted, with a repeat
        self.assertTrue(np.array_equal(data.x[rows], values[rows]))
        self.assertEqual(data.x[rows].dtype, np.float32)
        self.assertEqual(data.fingerprint["source_sha256"], self.record["sha256"])

    def test_a_subset_of_columns_and_rows(self):
        import numpy as np
        import pandas as pd

        frame = pd.read_csv(self.root / "data" / "synthetic" / "b.csv")
        two = load(self.section(features=["re:^f[02]$"]))
        self.assertEqual(two.feature_names, ("f0", "f2"))
        rows = np.array([9, 1, 4])
        self.assertTrue(np.array_equal(two.x[rows], frame[["f0", "f2"]].to_numpy(dtype=np.float32)[rows]))
        test = load(self.section(filter_column="published_split", filter_values=["test"]))
        expected = frame["published_split"] == "test"
        self.assertEqual(len(test.table), int(expected.sum()))
        self.assertEqual(test.table.ids.tolist(), frame.loc[expected, "row_id"].tolist())
        self.assertTrue(np.array_equal(test.x, frame.loc[expected, ["f0", "f1", "f2", "f3"]].to_numpy(np.float32)))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "filter_column must be one of the kept columns"):
            check(self.section(filter_column="group", filter_values=["g0"]))
        with self.assertRaisesRegex(ValueError, "filter_column together with filter_values"):
            check(self.section(filter_column="truth"))
        with self.assertRaisesRegex(FileNotFoundError, "no finished store"):
            load(self.section(store="absent"))
        with self.assertRaisesRegex(ValueError, "no row has published_split in"):
            load(self.section(filter_column="published_split", filter_values=["none"]))


class TestItemsAndGraph(unittest.TestCase):
    """The folders make-example-data writes, and a few hand-written items."""

    @classmethod
    def setUpClass(cls):
        load_catalog()
        from sweepkit.data.examples import make_example_data

        cls.repo = TempRepo()
        cls.root = cls.repo.__enter__()
        make_example_data(quiet)

    @classmethod
    def tearDownClass(cls):
        cls.repo.__exit__(None, None, None)

    def items_section(self, **changes) -> dict:
        return dict({"name": "items", "index": "synthetic/example_items/index.csv", "path_column": "path",
                     "row_id": "row_id", "keep": ["truth"], "kind": "sequence", "loader": "npy",
                     "sample_rate": None}, **changes)

    def graph_section(self, **changes) -> dict:
        return dict({"name": "graph", "nodes": "synthetic/example_graph/nodes.csv",
                     "edges": "synthetic/example_graph/edges.csv", "row_id": "row_id", "features": ["*"],
                     "keep": ["truth"], "source": "source", "target": "target", "undirected": False}, **changes)

    def test_items_one_array_file_per_row(self):
        import numpy as np

        data = load(self.items_section())
        self.assertEqual((data.kind, len(data.table), len(data.items)), ("sequence", 120, 120))
        self.assertEqual(data.table.ids[:2].tolist(), ["r000", "r001"])
        first = data.items.get(0)
        on_disk = np.load(self.root / "data" / "synthetic" / "example_items" / "r000.npy")
        self.assertTrue(np.array_equal(first, on_disk))
        self.assertEqual((first.dtype, first.shape[1]), (np.float32, 3))
        self.assertEqual(data.items.length(0), first.shape[0])
        self.assertEqual(sorted(set(data.table.column("truth").tolist())), ["c0", "c1", "c2"])
        self.assertEqual(data.fingerprint["items"], 120)

    def test_items_text_csv_and_wav(self):
        import wave

        import numpy as np

        folder = self.root / "data" / "hand"
        folder.mkdir(exist_ok=True)
        (folder / "a.txt").write_text("one two three", encoding="utf-8")
        (folder / "b.txt").write_text("four", encoding="utf-8")
        (folder / "text.csv").write_text("id,file,label\nt1,hand/a.txt,x\nt2,hand/b.txt,y\n", encoding="utf-8")
        text = load(self.items_section(index="hand/text.csv", path_column="file", row_id="id", keep=["label"],
                                       kind="text", loader="txt"))
        self.assertEqual((text.items.get(0), text.items.length(1)), ("one two three", 4))
        np.savetxt(folder / "s.csv", np.arange(12, dtype=np.float32).reshape(4, 3), delimiter=",")
        (folder / "arrays.csv").write_text("id,file\ns1,hand/s.csv\n", encoding="utf-8")
        arrays = load(self.items_section(index="hand/arrays.csv", path_column="file", row_id="id", keep=[],
                                         loader="csv"))
        self.assertEqual(arrays.items.get(0).tolist(), np.arange(12, dtype=np.float32).reshape(4, 3).tolist())
        samples = (np.sin(np.arange(1600) / 10) * 16000).astype("<i2")
        with wave.open(str(folder / "w.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(samples.tobytes())
        (folder / "audio.csv").write_text("id,file\nw1,hand/w.wav\n", encoding="utf-8")
        audio = load(self.items_section(index="hand/audio.csv", path_column="file", row_id="id", keep=[],
                                        kind="audio", loader="wav", sample_rate=16000))
        self.assertTrue(np.allclose(audio.items.get(0), samples / 32768.0))
        self.assertEqual(audio.items.length(0), 1600)
        wrong = load(self.items_section(index="hand/audio.csv", path_column="file", row_id="id", keep=[],
                                        kind="audio", loader="wav", sample_rate=8000))
        with self.assertRaisesRegex(ValueError, "16000 Hz, the settings say 8000"):
            wrong.items.get(0)

    def test_items_refusals(self):
        with self.assertRaisesRegex(ValueError, "wav takes a sample_rate"):
            check(self.items_section(kind="audio", loader="wav"))
        with self.assertRaisesRegex(ValueError, "wav takes a sample_rate"):
            check(self.items_section(sample_rate=16000))
        with self.assertRaisesRegex(ValueError, "loader npy gives"):
            check(self.items_section(kind="text"))
        with self.assertRaisesRegex(ValueError, "must stay inside data/"):
            check(self.items_section(index="../index.csv"))
        with self.assertRaisesRegex(ValueError, "'missing' is not a column"):
            load(self.items_section(keep=["missing"]))
        folder = self.root / "data" / "hand"
        folder.mkdir(exist_ok=True)
        (folder / "gone.csv").write_text("row_id,path\nq1,hand/nowhere.npy\n", encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "1 files listed in hand/gone.csv are missing"):
            load(self.items_section(index="hand/gone.csv", keep=[]))

    def test_graph_nodes_and_edges(self):
        import numpy as np

        from sweepkit.data.synthetic import make

        data = load(self.graph_section())
        made = make("graph", 150, 4, 3, 0.5, 1)
        self.assertEqual((data.kind, len(data.table), data.x.shape), ("graph", 150, (150, 4)))
        self.assertEqual(data.feature_names, ("f0", "f1", "f2", "f3"))
        self.assertEqual(data.table.ids[:2].tolist(), ["n000", "n001"])
        self.assertTrue(np.array_equal(data.graph.edges, made["edges"]))  # positions, in the file's order
        self.assertTrue(np.allclose(data.x, made["x"], rtol=1e-6, atol=1e-6))
        doubled = load(self.graph_section(undirected=True))
        self.assertEqual(doubled.graph.edges.shape[1], 2 * data.graph.edges.shape[1])

    def test_graph_edge_to_an_unknown_node(self):
        folder = self.root / "data" / "hand"
        folder.mkdir(exist_ok=True)
        (folder / "nodes.csv").write_text("row_id,f0,truth\na,1,c0\nb,2,c1\n", encoding="utf-8")
        (folder / "edges.csv").write_text("source,target\na,b\nb,z\n", encoding="utf-8")
        with self.assertRaisesRegex(KeyError, "'z' is not in the dataset"):
            load(self.graph_section(nodes="hand/nodes.csv", edges="hand/edges.csv"))


@needs_module("anndata")
class TestAnnData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_catalog()

    def test_cells_by_genes(self):
        import anndata
        import numpy as np
        import pandas as pd

        with TempRepo() as root:
            folder = root / "data" / "cells"
            folder.mkdir(parents=True)
            x = np.arange(20, dtype=np.float32).reshape(5, 4)
            obs = pd.DataFrame({"cell_type": ["t", "t", "b", "b", "nk"]}, index=[f"c{i}" for i in range(5)])
            var = pd.DataFrame({"highly_variable": [True, False, True, False]}, index=["g0", "g1", "g2", "g3"])
            adata = anndata.AnnData(X=x, obs=obs, var=var)
            adata.layers["counts"] = x * 10
            adata.write_h5ad(folder / "cells.h5ad")
            section = {"name": "anndata", "path": "cells/cells.h5ad", "layer": None, "keep": ["cell_type"],
                       "var_filter": None}
            data = load(section)
            self.assertEqual((data.kind, len(data.table), data.x.shape), ("tabular", 5, (5, 4)))
            self.assertEqual(data.table.ids.tolist(), ["c0", "c1", "c2", "c3", "c4"])
            self.assertEqual(data.feature_names, ("g0", "g1", "g2", "g3"))
            self.assertEqual(data.table.column("cell_type").tolist(), ["t", "t", "b", "b", "nk"])
            self.assertTrue(np.array_equal(data.x, x))
            chosen = load(dict(section, layer="counts", var_filter="highly_variable"))
            self.assertEqual(chosen.feature_names, ("g0", "g2"))
            self.assertTrue(np.array_equal(chosen.x, (x * 10)[:, [0, 2]]))


if __name__ == "__main__":
    unittest.main()
