"""Entry point for hosts that look for app.py at the repository root.

The app itself lives in app/app.py. It imports both `components` (a sibling
package inside app/) and `src` (at the root), so both directories have to be
importable before it runs.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

for path in (os.path.join(ROOT, "app"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

runpy.run_path(os.path.join(ROOT, "app", "app.py"), run_name="__main__")
