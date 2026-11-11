"""Print the newest stored prediction: ``python -m zeroshot.tools.show_latest`` (quickstart §7)."""

from __future__ import annotations

import json
import sys

from zeroshot.config import load_settings
from zeroshot.errors import ZeroShotError
from zeroshot.services.storage import StorageService


def main() -> int:
    try:
        storage = StorageService(load_settings())
        keys = storage.latest_keys(limit=1)
        if not keys:
            print(f"No predictions stored yet in bucket '{storage.bucket}'.")
            return 0
        record = storage.get(keys[0])
    except ZeroShotError as exc:
        print(f"error: {exc.user_message}", file=sys.stderr)
        return 1
    print(f"bucket: {storage.bucket}\nkey:    {keys[0]}\n")
    print(json.dumps(record.to_json_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
