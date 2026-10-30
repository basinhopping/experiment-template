"""The documentation stays true to the code, and the repository holds nothing personal.

- docs/COMMANDS.md is what commands-doc writes now, and every command's example parses with the real parser;
- the testing matrix in TESTING.md is what test-matrix writes now;
- every relative link and every `docs/...`, `templates/...` or `sweeps/...` path named in a Markdown file exists;
- no file of the repository holds an email address, an IP address, a user's home path, or one of the names the
  template must never carry (people, places, hosts, accounts and the original project's dataset: held here only as
  SHA-256 digests, so this test does not carry them either).
"""
import hashlib
import re
import shlex
import unittest

from tests.support import REPO, git_ignored, load_catalog

FORBIDDEN_SHA256 = {
    "b0d3518b1abfb59187800deb569decddae434fe85e383a2a0802325b04994ced",
    "3e0711305c65f8c661fe55732bed63e5692a6a20b418b4f3227248246570c870",
    "a6457bbdc50b0b89acbb5e672ef672eeba3a1a47e8ba8f7f31767a562d625e4d",
    "0d8ece0166d253b7b848a5e46266b0a1e71a4c7be6176696119466129738ac9f",
    "ec75b86699f31bd1f6e62959175c251825082868051d071c55defaa49c1cf744",
    "c79fecd76db5852f5823b03f193e01edd96b98ec91ca05f91bd16611c1dd852c",
    "76e32510789a773f6fbb8c27d7603018398df83b08609fbca08f0817ed98dbf6",
    "bae7d16cef7ef3a3e3805282482408a1051a9eb2c2b4f63a7be71d7af4dc90d6",
    "8e898e2af41d76f259f6e218b04b8bf120a1723f53e47c8d574b7f1c835ec81e",
    "19e6ddfdbb7a382007e0d881175a58d8dcdb4cb92437e6b828f80f8c2d4136b2",
    "043e2135cfcdeef28c5ce26de3142fc5449ee16fb40fce5a72eb6f7ffee6d336",
    "67845a6525041c26e2590aae9f49c82c08ba14cca545dc23b09f49be3d61e0a9",
    "35251268bb50dfbb1f5bf711ff0514001069e18c6e18247acdf582b7d89590ce",
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
HOME_PATH = re.compile(r"(?i)(?:[a-z]:\\users\\[a-z0-9_.-]+|/users/[a-z0-9_.-]+/|/home/[a-z0-9_.-]+/)")
# addresses that are fine: documentation examples and version numbers that look like addresses
ALLOWED = {"noreply@anthropic.com"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".sh", ".ps1", ".tmpl", ".csv", ".toml", ".cfg", ".ipynb", ""}
SKIP_DIRS = {".git", "data", "artifacts", "queue", "logs", "reports", "__pycache__", ".venv", "venv"}


def published(path) -> bool:
    """A file git would publish: not under a folder of data, results or environments, and not kept out by
    .gitignore (a person's login file, state notes, data and environments)."""
    relative = path.relative_to(REPO)
    return not (any(part in SKIP_DIRS or part.startswith(".venv") for part in relative.parts)
                or git_ignored(relative))


def repository_files() -> list:
    out = []
    for path in REPO.rglob("*"):
        if not path.is_file() or not published(path):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in (".gitignore", ".gitattributes", "LICENSE"):
            out.append(path)
    return out


class TestCommandsDoc(unittest.TestCase):
    def test_every_command_has_an_example_that_parses(self):
        load_catalog()
        from sweepkit import cli
        from sweepkit.commands_doc import EXAMPLES

        names = [name for name, _, _, _ in cli.COMMANDS]
        self.assertEqual(sorted(EXAMPLES), sorted(names))
        parser = cli.parser()
        for name in names:
            args = parser.parse_args(shlex.split(EXAMPLES[name]))
            self.assertEqual(args.command, name)

    @unittest.skipUnless((REPO / "docs" / "COMMANDS.md").is_file(), "docs/COMMANDS.md is not written yet")
    def test_commands_doc_is_current(self):
        load_catalog()
        from sweepkit.commands_doc import render

        self.assertEqual((REPO / "docs" / "COMMANDS.md").read_text(encoding="utf-8"), render(),
                         "run: python -m sweepkit.cli commands-doc --write yes")


class TestTestingMatrix(unittest.TestCase):
    def test_the_matrix_in_testing_md_is_current(self):
        """The testing matrix (TESTING.md, its last section) is what test-matrix writes from the components, the
        example sweeps and the tests now."""
        load_catalog()
        from sweepkit.test_matrix import current_section, render

        self.assertEqual(current_section(), render(), "run: python -m sweepkit.cli test-matrix --write yes")


def numbers_as_pinned() -> bool:
    """Whether the packages that compute the example round's numbers are installed at their pinned versions."""
    import importlib.metadata

    from sweepkit.env import pins

    pinned = {name.lower(): version for name, op, version in pins("core") if op == "=="}
    for name in ("numpy", "scipy", "pandas", "scikit-learn"):
        try:
            if importlib.metadata.version(name) != pinned[name]:
                return False
        except importlib.metadata.PackageNotFoundError:
            return False
    return True


class TestExampleRound(unittest.TestCase):
    @unittest.skipUnless((REPO / "docs" / "EXAMPLE_ROUND.md").is_file(), "docs/EXAMPLE_ROUND.md is not written yet")
    def test_example_round_is_current(self):
        """Written again by running both rounds (about a minute), and compared byte for byte."""
        load_catalog()
        if not numbers_as_pinned():
            self.skipTest("numpy, SciPy, pandas or scikit-learn differ from the pins, so a number may round differently")
        from sweepkit.example_round import render

        self.assertEqual((REPO / "docs" / "EXAMPLE_ROUND.md").read_text(encoding="utf-8"), render(lambda m: None),
                         "run: python -m sweepkit.cli example-round --write yes")


class TestLinks(unittest.TestCase):
    def test_paths_named_in_markdown_exist(self):
        missing = []
        for doc in sorted(REPO.rglob("*.md")):
            if not published(doc):
                continue
            text = doc.read_text(encoding="utf-8")
            targets = re.findall(r"\]\((?!https?://|#|mailto:)([^)#\s]+)", text)
            targets += re.findall(r"`((?:docs|templates|sweeps/examples|node|env|tests|notebooks)/[A-Za-z0-9_./-]+)`",
                                  text)
            for target in targets:
                if "<" in target or "*" in target:
                    continue
                base = doc.parent if not target.startswith(("docs/", "templates/", "sweeps/", "node/", "env/",
                                                            "tests/", "notebooks/", "src/")) else REPO
                if not (base / target).exists() and not (REPO / target).exists():
                    missing.append(f"{doc.relative_to(REPO)}: {target}")
        self.assertEqual(missing, [])


class TestNothingPersonal(unittest.TestCase):
    def test_no_names_addresses_or_home_paths(self):
        found = []
        for path in repository_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            where = path.relative_to(REPO).as_posix()
            for token in set(re.findall(r"[a-z0-9_.-]+", text.lower())):
                for part in {token, token.strip("._-")}:
                    if hashlib.sha256(part.encode("utf-8")).hexdigest() in FORBIDDEN_SHA256:
                        found.append(f"{where}: a forbidden name")
            for address in EMAIL.findall(text):
                if address not in ALLOWED and not address.endswith(("@example.com", "@example.org")):
                    found.append(f"{where}: an email address")
            for ip in IPV4.findall(text):
                if all(0 <= int(p) <= 255 for p in ip.split(".")) and not ip.startswith(("0.", "127.", "10.0.0.")):
                    found.append(f"{where}: an IP address {ip}")
            for home in HOME_PATH.findall(text):
                found.append(f"{where}: a home path {home}")
        self.assertEqual(sorted(set(found)), [])


if __name__ == "__main__":
    unittest.main()
