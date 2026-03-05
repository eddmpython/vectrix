"""
Sync constants.json values to all hardcoded locations across the project.

Usage:
    uv run python scripts/syncConstants.py          # apply changes
    uv run python scripts/syncConstants.py --check   # CI mode: fail if out of sync
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSTANTS_PATH = ROOT / "constants.json"


def loadConstants():
    with open(CONSTANTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class FilePatcher:
    def __init__(self, path, encoding="utf-8"):
        self.path = Path(path)
        self.encoding = encoding
        self.original = self.path.read_text(encoding=encoding)
        self.content = self.original

    def replace(self, pattern, replacement, count=0):
        self.content = re.sub(pattern, replacement, self.content, count=count, flags=re.MULTILINE)
        return self

    def replaceLiteral(self, old, new, count=0):
        if count == 0:
            self.content = self.content.replace(old, new)
        else:
            for _ in range(count):
                self.content = self.content.replace(old, new, 1)
        return self

    @property
    def changed(self):
        return self.content != self.original

    def write(self):
        if self.changed:
            self.path.write_text(self.content, encoding=self.encoding)
        return self.changed


def syncPyprojectToml(c):
    p = FilePatcher(ROOT / "pyproject.toml")
    p.replace(r'^version = ".*?"', f'version = "{c["version"]}"', count=1)
    return p


def syncCargoToml(c):
    p = FilePatcher(ROOT / "Cargo.toml")
    p.replace(r'^version = ".*?"', f'version = "{c["version"]}"', count=1)
    return p


def syncRustCargoToml(c):
    p = FilePatcher(ROOT / "rust" / "Cargo.toml")
    p.replace(r'^version = ".*?"', f'version = "{c["version"]}"', count=1)
    return p


def syncRustPyprojectToml(c):
    p = FilePatcher(ROOT / "rust" / "pyproject.toml")
    p.replace(r'(?<=^version = ").*?(?=")', c["version"], count=1)
    return p


def syncLlmsTxt(c):
    p = FilePatcher(ROOT / "llms.txt")
    p.replace(r"(?<=- Version: )[\d.]+", c["version"])
    return p


def syncLlmsFullTxt(c):
    p = FilePatcher(ROOT / "llms-full.txt")
    p.replace(r"(?<=- Version: )[\d.]+", c["version"])
    p.replace(
        r"\d+ accelerated functions",
        f'{c["rustFunctions"]} accelerated functions',
    )
    return p


def syncLandingLlmsTxt(c):
    path = ROOT / "landing" / "static" / "llms.txt"
    if not path.exists():
        return None
    p = FilePatcher(path)
    p.replace(
        r"\d+ core hot loops",
        f'{c["rustFunctions"]} core hot loops',
    )
    return p


def syncLandingLlmsFullTxt(c):
    path = ROOT / "landing" / "static" / "llms-full.txt"
    if not path.exists():
        return None
    p = FilePatcher(path)
    p.replace(r"(?<=- Version: )[\d.]+", c["version"])
    p.replace(
        r"\d+ accelerated functions",
        f'{c["rustFunctions"]} accelerated functions',
    )
    p.replace(
        r"\d+ core hot loops",
        f'{c["rustFunctions"]} core hot loops',
    )
    return p


def syncHeroSvelte(c):
    path = ROOT / "landing" / "src" / "lib" / "components" / "sections" / "Hero.svelte"
    if not path.exists():
        return None
    p = FilePatcher(path)
    p.replace(r"v[\d.]+ — Built-in Rust Engine", f'v{c["version"]} — Built-in Rust Engine')
    p.replace(r"value: '\d+', label: 'Tests'", f"value: '{c['testCount']}', label: 'Tests'")
    p.replace(r"value: '\d+x', label: 'Rust Engine'", f"value: '{c['maxSpeedup']}', label: 'Rust Engine'")
    return p


def syncPageSvelte(c):
    path = ROOT / "landing" / "src" / "routes" / "+page.svelte"
    if not path.exists():
        return None
    p = FilePatcher(path)
    p.replace(r'"softwareVersion": "[\d.]+"', f'"softwareVersion": "{c["version"]}"')
    return p


def syncFeaturesSvelte(c):
    path = ROOT / "landing" / "src" / "lib" / "components" / "sections" / "Features.svelte"
    if not path.exists():
        return None
    p = FilePatcher(path)
    p.replace(
        r"\d+ accelerated hot loops",
        f'{c["rustFunctions"]} accelerated hot loops',
    )
    return p


def syncPerformanceSvelte(c):
    path = ROOT / "landing" / "src" / "lib" / "components" / "sections" / "Performance.svelte"
    if not path.exists():
        return None
    p = FilePatcher(path)
    p.replace(
        r"\d+ Rust-accelerated core loops",
        f'{c["rustFunctions"]} Rust-accelerated core loops',
    )
    return p


def syncHomeHtml(c):
    p = FilePatcher(ROOT / "docs" / "overrides" / "home.html")
    p.replace(r"v[\d.]+ — Built-in Rust Engine", f'v{c["version"]} — Built-in Rust Engine')
    p.replace(
        r'<span class="vx-hero-stat-value">\d+</span>\s*<span class="vx-hero-stat-label">Tests',
        f'<span class="vx-hero-stat-value">{c["testCount"]}</span>\n        <span class="vx-hero-stat-label">Tests',
    )
    p.replace(r"\d+ accelerated hot loops", f'{c["rustFunctions"]} accelerated hot loops')
    p.replace(r"\d+ core hot loops", f'{c["rustFunctions"]} core hot loops')
    p.replace(
        r"<strong>\d+ Tests</strong>",
        f'<strong>{c["testCount"]} Tests</strong>',
    )
    return p


def syncReadme(c):
    p = FilePatcher(ROOT / "README.md")
    p.replace(
        r"Tests-\d+%20passed",
        f'Tests-{c["testCount"]}%20passed',
    )
    p.replace(r"\d+ core hot loops", f'{c["rustFunctions"]} core hot loops')
    p.replace(r"\d+ accelerated functions", f'{c["rustFunctions"]} accelerated functions')
    return p


def syncInitPy(c):
    p = FilePatcher(ROOT / "src" / "vectrix" / "__init__.py")
    p.replace(r'^__version__ = ".*?"', f'__version__ = "{c["version"]}"', count=1)
    return p


def syncVectrixPy(c):
    p = FilePatcher(ROOT / "src" / "vectrix" / "vectrix.py")
    p.replace(r'^    VERSION = ".*?"', f'    VERSION = "{c["version"]}"', count=1)
    return p


ALL_SYNCS = [
    ("pyproject.toml", syncPyprojectToml),
    ("Cargo.toml", syncCargoToml),
    ("rust/Cargo.toml", syncRustCargoToml),
    ("rust/pyproject.toml", syncRustPyprojectToml),
    ("llms.txt", syncLlmsTxt),
    ("llms-full.txt", syncLlmsFullTxt),
    ("landing/static/llms.txt", syncLandingLlmsTxt),
    ("landing/static/llms-full.txt", syncLandingLlmsFullTxt),
    ("landing Hero.svelte", syncHeroSvelte),
    ("landing +page.svelte", syncPageSvelte),
    ("landing Features.svelte", syncFeaturesSvelte),
    ("landing Performance.svelte", syncPerformanceSvelte),
    ("docs home.html", syncHomeHtml),
    ("README.md", syncReadme),
    ("src/vectrix/__init__.py", syncInitPy),
    ("src/vectrix/vectrix.py", syncVectrixPy),
]


def main():
    checkMode = "--check" in sys.argv
    c = loadConstants()

    outOfSync = []
    updated = []

    for label, fn in ALL_SYNCS:
        try:
            patcher = fn(c)
        except FileNotFoundError:
            continue

        if patcher is None:
            continue

        if patcher.changed:
            if checkMode:
                outOfSync.append(label)
            else:
                patcher.write()
                updated.append(label)

    if checkMode:
        if outOfSync:
            print(f"ERROR: {len(outOfSync)} file(s) out of sync with constants.json:")
            for f in outOfSync:
                print(f"  - {f}")
            print("\nRun: uv run python scripts/syncConstants.py")
            sys.exit(1)
        else:
            print("OK: All files in sync with constants.json")
            sys.exit(0)
    else:
        if updated:
            print(f"Updated {len(updated)} file(s):")
            for f in updated:
                print(f"  - {f}")
        else:
            print("All files already in sync.")


if __name__ == "__main__":
    main()
