"""Bounded process scheduler; idle cores race remaining searches."""
import collections
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: run tasks.txt")
    tasks = [line.split() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    if any(len(t) != 3 for t in tasks):
        raise ValueError("invalid task")
    binary = str(Path(__file__).resolve().parent / "md5fastcoll")
    cores = min(32, len(os.sched_getaffinity(0)))
    pending = collections.deque(range(len(tasks)))
    unfinished = set(pending)
    active = {}
    copies = collections.Counter()
    attempt = 0
    with tempfile.TemporaryDirectory(prefix="miniclash-race-", dir=".") as temp:
        try:
            while unfinished:
                while len(active) < cores:
                    if pending:
                        index = pending.popleft()
                    else:
                        index = min(unfinished, key=lambda i: (copies[i], i))
                    first = str(Path(temp) / (str(attempt) + "a"))
                    second = str(Path(temp) / (str(attempt) + "b"))
                    attempt += 1
                    child = subprocess.Popen([binary, "-q", "--seed1", str(secrets.randbits(32)),
                        "--seed2", str(secrets.randbits(32) or 1), "-p", tasks[index][0],
                        "-o", first, second], stdout=subprocess.DEVNULL)
                    active[child] = (index, first, second)
                    copies[index] += 1
                progressed = False
                for child in list(active):
                    if child not in active:
                        continue
                    status = child.poll()
                    if status is None:
                        continue
                    index, first, second = active.pop(child)
                    copies[index] -= 1
                    if status != 0:
                        raise RuntimeError("collision child exited with status " + str(status))
                    os.replace(first, tasks[index][1])
                    os.replace(second, tasks[index][2])
                    unfinished.remove(index)
                    for sibling, (other, a, b) in list(active.items()):
                        if other == index:
                            if sibling.poll() is None:
                                sibling.kill()
                            sibling.wait()
                            del active[sibling]
                            copies[index] -= 1
                    progressed = True
                if not progressed:
                    time.sleep(0.005)
        finally:
            for child in active:
                if child.poll() is None:
                    child.kill()
            for child in active:
                child.wait()
    print("generated", len(tasks), "collisions; attempts", attempt, "cores", cores)


if __name__ == "__main__":
    main()
