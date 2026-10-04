from __future__ import annotations

from rune_scheduler.api.v1 import DispatchAction, DispatchDecision, Placement


class Policy:
    """Resource-aware list scheduling using only the public scenario model."""

    def __init__(self, model):
        self.model = model
        self.types = {t.id: t for t in model.task_types}
        self.workers = {w.id: w for w in model.workers}
        self.cost = {}
        self.best = {}
        self.tail = {}
        self.stable = {}
        for t in model.task_types:
            costs = {}
            for w in model.workers:
                if set(t.required_capabilities).issubset(w.capabilities) and t.memory_reservation <= w.memory_capacity:
                    ratios = {m.task_type: m.ratio.numerator / m.ratio.denominator for m in w.duration_multipliers}
                    costs[w.id] = ((t.runtime_work.min + t.runtime_work.max) / 2 + t.setup_work) * ratios.get(t.id, 1)
            self.cost[t.id] = costs
            self.best[t.id] = min(costs.values())
            self.stable[t.id] = min((c for wid,c in costs.items() if self.workers[wid].outage_prior is None), default=self.best[t.id])
        # Critical-path estimate from public workflow and derivation edges.
        successors = {t.id: [] for t in model.task_types}
        for d in model.derivations:
            successors[d.parent_type].append(d.successor_type)
        for wf in model.workflow_templates:
            nodes = {n.id: n for n in wf.nodes}
            for n in wf.nodes:
                for p in n.depends_on:
                    successors[nodes[p].task_type].append(n.task_type)
        def tail(t, visiting):
            if t in visiting:
                return 0
            if t not in self.tail:
                self.tail[t] = self.best[t] + max((tail(s, visiting | {t}) for s in successors[t]), default=0)
            return self.tail[t]
        for t in self.types:
            tail(t, set())

    def choose_placements(self, observation):
        cpu = {w.worker_id: w.cpu_demand for w in observation.workers}
        mem = {w.worker_id: w.memory_reserved for w in observation.workers}
        available = {w.worker_id for w in observation.workers if w.available}
        groups = {}
        for task in observation.ready_tasks:
            groups.setdefault(task.task_type, []).append(task)
        placements = []
        batches = {}
        for _ in range(self.model.safety_limits.max_batch_size):
            best_choice = None
            best_score = float('inf')
            for tid, tasks in groups.items():
                if not tasks:
                    continue
                t = self.types[tid]
                for wid, duration in self.cost[tid].items():
                    w = self.workers[wid]
                    if duration > self.stable[tid] * 4.0:
                        continue
                    if wid not in available or mem[wid] + t.memory_reservation > w.memory_capacity:
                        continue
                    runtime = (t.runtime_work.min + t.runtime_work.max) / 2
                    n = batches.get((tid, wid), 0)
                    # Only setup is shared; discount the marginal CPU estimate
                    # while still reserving every task's full memory footprint.
                    increment = t.cpu_demand * (runtime / (runtime + t.setup_work) if n else 1)
                    if cpu[wid] and cpu[wid] + increment > w.cpu_capacity * 1.1:
                        continue
                    score = duration / self.best[tid] * (1 + cpu[wid] / w.cpu_capacity) ** 0.5
                    score /= (self.tail[tid] / self.best[tid]) ** 0.5
                    score *= (runtime + t.setup_work / (n + 1)) / (runtime + t.setup_work)
                    if score < best_score:
                        best_score = score
                        best_choice = (tid, wid, increment)
            if best_choice is None:
                break
            tid, wid, increment = best_choice
            t = self.types[tid]
            batches[tid, wid] = batches.get((tid, wid), 0) + 1
            task = groups[tid].pop(0)
            placements.append(Placement(task.task_id, wid))
            cpu[wid] += increment
            mem[wid] += t.memory_reservation
        return DispatchDecision(DispatchAction.DISPATCH, tuple(placements)) if placements else DispatchDecision(DispatchAction.DEFER)
