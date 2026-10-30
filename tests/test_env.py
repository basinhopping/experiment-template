"""The environment profiles: every requirement file reads, pins exact versions, and belongs to a profile; every
component names a profile that exists and import names that some profile installs; docs/PROFILES.md is current."""
import unittest

from tests.support import REPO, load_catalog

# the only pins that are not exact, each with its reason (docs/PROFILES.md repeats it)
NOT_EXACT = {("automl", "setuptools"): "TPOT's dependency stopit imports pkg_resources, which setuptools 82 removed"}


class TestProfiles(unittest.TestCase):
    def test_every_requirement_file_reads_and_pins_exact_versions(self):
        from sweepkit.env import PROFILES, pins

        files = sorted(p.name for p in (REPO / "env").glob("requirements-*.txt"))
        self.assertEqual(files, sorted(f"requirements-{p}.txt" for p in PROFILES))
        for profile in PROFILES:
            found = pins(profile)
            self.assertTrue(found, profile)
            names = [name.lower() for name, _, _ in found]
            self.assertEqual(len(names), len(set(names)), f"{profile} names a package twice")
            for name, operator, version in found:
                if (profile, name) in NOT_EXACT:
                    continue
                self.assertEqual(operator, "==", f"{profile}: {name} {operator} {version} is not an exact pin")

    def test_profile_table_is_complete(self):
        from sweepkit.env import PROFILES

        for name, profile in PROFILES.items():
            self.assertEqual(set(profile), {"for", "environment", "default", "licences", "platforms", "imports"}, name)
            self.assertIn(profile["environment"], ("main", "rapids", "automl"), name)
            self.assertTrue(profile["imports"], name)

    def test_components_name_known_profiles_and_installable_imports(self):
        load_catalog()
        from sweepkit.env import PROFILES
        from sweepkit.registry import REGISTRIES

        importable = {m for p in PROFILES.values() for m in p["imports"]}
        for kind, registry in REGISTRIES.items():
            for entry in registry.entries():
                profile = entry.meta.get("profile")
                if profile is None:
                    continue
                self.assertIn(profile, PROFILES, f"{kind} {entry.name}")
                for package in entry.meta.get("packages", ()):
                    self.assertIn(package, importable, f"{kind} {entry.name}: {package} is in no profile")

    def test_required_imports(self):
        from sweepkit.env import required_imports

        self.assertEqual(required_imports(["core", "core"]), required_imports(["core"]))
        self.assertIn("torch", required_imports(["core", "torch"]))
        with self.assertRaises(ValueError):
            required_imports(["no_such_profile"])

    @unittest.skipUnless((REPO / "docs" / "PROFILES.md").is_file(), "docs/PROFILES.md is not written yet")
    def test_profiles_doc_is_current(self):
        load_catalog()
        from sweepkit.env import profiles_doc

        written = (REPO / "docs" / "PROFILES.md").read_text(encoding="utf-8")
        self.assertEqual(written, profiles_doc(), "run: python -m sweepkit.cli profiles-doc --write yes")


if __name__ == "__main__":
    unittest.main()
