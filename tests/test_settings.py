"""Settings validation: a missing, extra or mistyped field must raise; added fields keep old hashes; scalar and
registry-section fields."""
import dataclasses
import unittest
from typing import Literal

from sweepkit.settings import SettingsError, check_settings_class, settings_hash, to_plain, validate


@dataclasses.dataclass(frozen=True)
class Inner:
    kind: Literal["mean", "max"]
    size: int


@dataclasses.dataclass(frozen=True)
class Example:
    n_mels: int
    lr: float
    use_bn: bool
    name: str
    channels: tuple[int, ...]
    kernel: tuple[int, int]
    pooling: Literal["mean", "max", "attention"]
    dropout: float | None
    extra: dict[str, float]
    inner: Inner


def good() -> dict:
    return {"n_mels": 64, "lr": 0.001, "use_bn": True, "name": "cnn", "channels": [32, 64],
            "kernel": [3, 3], "pooling": "mean", "dropout": None, "extra": {"a": 1.5},
            "inner": {"kind": "max", "size": 2}}


class TestValidate(unittest.TestCase):
    def test_good_settings_validate(self):
        s = validate(Example, good(), "model")
        self.assertEqual(s.channels, (32, 64))
        self.assertEqual(s.inner.kind, "max")
        self.assertIsNone(s.dropout)

    def test_missing_field_raises(self):
        raw = good()
        del raw["lr"]
        with self.assertRaisesRegex(SettingsError, "missing \\['lr'\\]"):
            validate(Example, raw, "model")

    def test_unknown_field_raises(self):
        raw = good()
        raw["n_mel"] = 64
        with self.assertRaisesRegex(SettingsError, "unknown \\['n_mel'\\]"):
            validate(Example, raw, "model")

    def test_number_as_text_raises(self):
        # A config edit by a regular expression once left "64# comment" as text and failed every model.
        raw = good()
        raw["n_mels"] = "64# comment"
        with self.assertRaisesRegex(SettingsError, "model.n_mels: expected a whole number"):
            validate(Example, raw, "model")
        raw = good()
        raw["lr"] = "0.001"
        with self.assertRaises(SettingsError):
            validate(Example, raw, "model")

    def test_bool_is_not_a_number_and_number_is_not_bool(self):
        raw = good()
        raw["n_mels"] = True
        with self.assertRaises(SettingsError):
            validate(Example, raw, "model")
        raw = good()
        raw["use_bn"] = 1
        with self.assertRaises(SettingsError):
            validate(Example, raw, "model")

    def test_float_accepts_int_and_stores_float(self):
        raw = good()
        raw["lr"] = 1
        s = validate(Example, raw, "model")
        self.assertIsInstance(s.lr, float)
        raw2 = good()
        raw2["lr"] = 1.0
        self.assertEqual(settings_hash(s), settings_hash(validate(Example, raw2, "model")))

    def test_int_refuses_float(self):
        raw = good()
        raw["n_mels"] = 64.0
        with self.assertRaises(SettingsError):
            validate(Example, raw, "model")

    def test_nan_and_inf_refused(self):
        for bad in (float("nan"), float("inf")):
            raw = good()
            raw["lr"] = bad
            with self.assertRaises(SettingsError):
                validate(Example, raw, "model")

    def test_literal_and_nested_errors_name_the_field(self):
        raw = good()
        raw["pooling"] = "avg"
        with self.assertRaisesRegex(SettingsError, "model.pooling"):
            validate(Example, raw, "model")
        raw = good()
        raw["inner"] = {"kind": "max"}
        with self.assertRaisesRegex(SettingsError, "model.inner"):
            validate(Example, raw, "model")

    def test_fixed_tuple_length(self):
        raw = good()
        raw["kernel"] = [3, 3, 3]
        with self.assertRaises(SettingsError):
            validate(Example, raw, "model")

    def test_list_items_checked(self):
        raw = good()
        raw["channels"] = [32, "64"]
        with self.assertRaisesRegex(SettingsError, "channels\\[1\\]"):
            validate(Example, raw, "model")

    def test_hash_ignores_key_order_and_sees_values(self):
        a = validate(Example, good(), "model")
        reordered = dict(reversed(list(good().items())))
        b = validate(Example, reordered, "model")
        self.assertEqual(settings_hash(a), settings_hash(b))
        changed = good()
        changed["n_mels"] = 128
        self.assertNotEqual(settings_hash(a), settings_hash(validate(Example, changed, "model")))

    def test_to_plain_round_trip(self):
        s = validate(Example, good(), "model")
        self.assertEqual(to_plain(s)["channels"], [32, 64])
        self.assertEqual(validate(Example, to_plain(s), "model"), s)


@dataclasses.dataclass(frozen=True)
class Grown:
    """A settings class that gained fields twice after runs existed."""
    n_mels: int
    pooling: Literal["mean", "max"]
    speeds: tuple[float, ...]
    alpha: float
    head: int | None

    ADDED_FIELDS = (("2027-01-10", {"speeds": [1.0], "alpha": 0.0}), ("2027-02-20", {"head": None}))


class TestAddedFields(unittest.TestCase):
    """Fields added after runs exist: older forms stay valid and keep their hash (settings.py)."""

    def test_every_form_validates_and_older_forms_get_the_listed_values(self):
        oldest = validate(Grown, {"n_mels": 64, "pooling": "mean"}, "model")
        middle = validate(Grown, {"n_mels": 64, "pooling": "mean", "speeds": [1.0], "alpha": 0}, "model")
        newest = validate(Grown, {"n_mels": 64, "pooling": "mean", "speeds": [1.0], "alpha": 0.0, "head": None},
                          "model")
        self.assertEqual(oldest, newest)
        self.assertEqual(middle, newest)
        self.assertEqual(oldest.speeds, (1.0,))
        self.assertIsInstance(oldest.alpha, float)

    def test_listed_values_hash_as_the_older_form(self):
        old_hash = settings_hash(validate(Grown, {"n_mels": 64, "pooling": "mean"}, "model"))
        from sweepkit.hashing import sha256_json

        self.assertEqual(old_hash, sha256_json({"n_mels": 64, "pooling": "mean"}))
        full = {"n_mels": 64, "pooling": "mean", "speeds": [1.0], "alpha": 0.0, "head": None}
        self.assertEqual(settings_hash(validate(Grown, full, "model")), old_hash)
        for changed in ({"speeds": [0.9, 1.0, 1.1]}, {"alpha": 0.2}, {"head": 16}):
            other = validate(Grown, {**full, **changed}, "model")
            self.assertNotEqual(settings_hash(other), old_hash, changed)
        middle = validate(Grown, {"n_mels": 64, "pooling": "mean", "speeds": [0.9, 1.1], "alpha": 0.0}, "model")
        self.assertEqual(settings_hash(middle), sha256_json({"n_mels": 64, "pooling": "mean", "speeds": [0.9, 1.1]}))
        self.assertEqual(to_plain(middle)["head"], None, "to_plain shows every field")

    def test_partial_forms_are_refused(self):
        for raw in ({"n_mels": 64, "pooling": "mean", "speeds": [1.0]},              # alpha missing
                    {"n_mels": 64, "pooling": "mean", "head": None},                 # newest field alone
                    {"n_mels": 64, "pooling": "mean", "speeds": [1.0], "alpha": 0.0, "head": None, "x": 1}):
            with self.assertRaises(SettingsError):
                validate(Grown, raw, "model")

    def test_bad_declarations_fail_at_registration(self):
        def grown(added):
            @dataclasses.dataclass(frozen=True)
            class G:
                n: int
                m: float

                ADDED_FIELDS = added
            return G

        check_settings_class(grown((("2027-01-10", {"m": 0.0}),)))
        for added in ((("2027-01-10", {"x": 0.0}),),                                # not a field
                      (("2027-01-10", {"m": "zero"}),),                             # wrong type
                      (("2027-02-01", {"m": 0.0}), ("2027-01-01", {"n": 1})),      # dates out of order
                      (("27 Sep", {"m": 0.0}),),                                    # not a date
                      (("2027-01-10", {"m": 0.0}), ("2027-02-01", {"m": 1.0})),    # listed twice
                      [("2027-01-10", {"m": 0.0})]):                                # not a tuple
            with self.assertRaises(TypeError, msg=str(added)):
                check_settings_class(grown(added))


class TestDeclarations(unittest.TestCase):
    def test_defaults_refused(self):
        @dataclasses.dataclass(frozen=True)
        class WithDefault:
            n: int = 3

        with self.assertRaisesRegex(TypeError, "has a default"):
            check_settings_class(WithDefault)

    def test_mutable_refused(self):
        @dataclasses.dataclass
        class Mutable:
            n: int

        with self.assertRaisesRegex(TypeError, "frozen"):
            check_settings_class(Mutable)

    def test_unsupported_type_refused(self):
        @dataclasses.dataclass(frozen=True)
        class Odd:
            n: set

        with self.assertRaisesRegex(TypeError, "unsupported"):
            check_settings_class(Odd)


class TestScalarAndSections(unittest.TestCase):
    def test_scalar_keeps_type_and_refuses_containers(self):
        from sweepkit.settings import Scalar

        @dataclasses.dataclass(frozen=True)
        class WithExtra:
            extra: dict[str, Scalar]

        s = validate(WithExtra, {"extra": {"a": 1, "b": 0.5, "c": "x", "d": True}}, "model")
        self.assertEqual(s.extra, {"a": 1, "b": 0.5, "c": "x", "d": True})
        self.assertIs(type(s.extra["a"]), int)
        for bad in ([1], {"x": 1}, None, float("nan")):
            with self.assertRaises(SettingsError):
                validate(WithExtra, {"extra": {"a": bad}}, "model")

    def test_registry_section_validates_against_the_named_component(self):
        from tests.support import load_catalog
        from sweepkit.settings import Selected, section_type

        load_catalog()

        @dataclasses.dataclass(frozen=True)
        class Step:
            method: section_type("model", ("embedder",))

        raw = {"method": {"name": "pca", "n_components": 2, "solver": "full", "whiten": False, "batch_size": None,
                          "extra": {}}}
        s = validate(Step, raw, "prep[0]")
        self.assertIsInstance(s.method, Selected)
        self.assertEqual(to_plain(s), raw)
        self.assertEqual(validate(Step, to_plain(s), "prep[0]"), s)
        wrong_family = {"method": {"name": "kmeans", "n_clusters": 2, "init": "random", "n_init": 1, "max_iter": 10,
                                   "tol": 0.0, "extra": {}}}
        with self.assertRaisesRegex(SettingsError, "takes embedder"):
            validate(Step, wrong_family, "prep[0]")
        missing = {"method": {"name": "pca", "n_components": 2}}
        with self.assertRaisesRegex(SettingsError, "missing"):
            validate(Step, missing, "prep[0]")


if __name__ == "__main__":
    unittest.main()
