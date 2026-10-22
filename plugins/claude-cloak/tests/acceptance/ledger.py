"""One private, locked budget shared by successive authorized live runs."""
import fcntl
import json
import os
from pathlib import Path
import time

from audit import reject_symlinks

AGGREGATE_REQUEST_LIMITS = (48, 96, 200)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        reject_symlinks(self.path)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def update(self, operation):
        reject_symlinks(self.path)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            data = file.read()
            state = json.loads(data) if data else {"schema": 1, "attempts": 0, "live_seconds": 0, "active": None, "runs": {}}
            if state.get("schema") != 1 or not isinstance(state.get("runs"), dict):
                raise ValueError("invalid live ledger")
            if state.get("aggregate_requests", 48) not in AGGREGATE_REQUEST_LIMITS:
                raise ValueError("invalid aggregate request limit in live ledger")
            result = operation(state)
            file.seek(0)
            json.dump(state, file, sort_keys=True)
            file.truncate()
            file.flush()
            os.fsync(file.fileno())
            return result

    def begin(self, run, seconds, aggregate_requests=48):
        if aggregate_requests not in AGGREGATE_REQUEST_LIMITS:
            raise ValueError("aggregate request limit must be 48 or separately authorized 96 or 200")
        def operation(state):
            if state["active"] is not None:
                raise ValueError("live ledger has an unfinished run; inspect it before authorizing more calls")
            if run in state["runs"] or state["attempts"] >= aggregate_requests or state["live_seconds"] >= 1800:
                raise ValueError("aggregate live budget exhausted or run already recorded")
            remaining = min(seconds, 1800 - state["live_seconds"])
            started = time.time()
            state["active"] = run
            state["aggregate_requests"] = max(state.get("aggregate_requests", 48), aggregate_requests)
            state["runs"][run] = {"attempts": 0, "started": started, "deadline": started + remaining,
                                  "aggregate_requests": aggregate_requests}
            return started + remaining
        return self.update(operation)

    def attempt(self, run, run_cap):
        def operation(state):
            item = state["runs"].get(run)
            if state["active"] != run or item is None:
                raise ValueError("run does not own the live ledger")
            aggregate_requests = item.get("aggregate_requests", 48)
            if aggregate_requests not in AGGREGATE_REQUEST_LIMITS:
                raise ValueError("invalid run aggregate request limit")
            # Rejected attempts count too. The driver stops when a guard is observed.
            state["attempts"] += 1
            item["attempts"] += 1
            reason = ("aggregate_cap" if state["attempts"] >= aggregate_requests else
                      "request_cap" if item["attempts"] >= min(run_cap, 24) else
                      "deadline" if time.time() >= item["deadline"] else None)
            return item["attempts"], state["attempts"], reason
        return self.update(operation)

    def finish(self, run):
        def operation(state):
            if state["active"] != run:
                return
            item = state["runs"][run]
            item["elapsed"] = max(0, time.time() - item["started"])
            state["live_seconds"] += item["elapsed"]
            state["active"] = None
        self.update(operation)
