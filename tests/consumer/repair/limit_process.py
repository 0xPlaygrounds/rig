"""Trusted pre-exec limits; argv is passed directly, never interpreted as shell."""

import os
import resource
import sys

mode = sys.argv[1]
if mode not in ("compile", "test") or len(sys.argv) < 3:
    raise SystemExit("invalid validation launcher invocation")

# The launcher remains the tracked PID after exec. A session leader cannot
# join an unrelated process group; the host also retains a direct PID kill.
os.setsid()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_CPU, (30 if mode == "compile" else 10,) * 2)
if sys.platform.startswith("linux"):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024,) * 2)
# Darwin rejects finite RLIMIT_AS/RLIMIT_DATA here. Do not claim a hard
# memory quota there; wall time, CPU, descriptors and per-file size are bounded.
os.execv(sys.argv[2], sys.argv[2:])
