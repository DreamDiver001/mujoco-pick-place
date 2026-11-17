"""Download the Franka Panda model from MuJoCo Menagerie (Apache-2.0) into third_party/."""
import re
import sys
import urllib.request
from pathlib import Path

COMMIT = "370c43965323c0944b433b202aac5058b5f907a9"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/franka_emika_panda"
DEST = Path(__file__).resolve().parent.parent / "third_party" / "franka_emika_panda"


def get(rel: str) -> bytes:
    with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as r:
        return r.read()


def main():
    (DEST / "assets").mkdir(parents=True, exist_ok=True)
    xml = get("panda.xml")
    (DEST / "panda.xml").write_bytes(xml)
    (DEST / "LICENSE").write_bytes(get("LICENSE"))
    files = sorted(set(re.findall(r'file="([^"]+)"', xml.decode())))
    for i, name in enumerate(files, 1):
        target = DEST / "assets" / name
        if not target.exists():
            target.write_bytes(get(f"assets/{name}"))
        print(f"\r{i}/{len(files)} meshes", end="", flush=True)
    print(f"\nPanda model ready in {DEST}")


if __name__ == "__main__":
    sys.exit(main())
