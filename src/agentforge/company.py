"""The company layer: a roster of role-specialised agents executing a dependency graph of tasks.

Each worker is the same Agent engine wearing a different role, so the company runs fully offline
with the local brain and gets smarter if a third-party brain is plugged in.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .agent import Agent
from .tools import slugify
from .workspace import Workspace


@dataclass(frozen=True)
class Role:
    key: str
    title: str
    document: str  # human phrase understood by the brains / template alias
    tools: frozenset[str] = frozenset({"write_doc", "recall", "summarize_text", "read_file"})  # least privilege


@dataclass(frozen=True)
class Task:
    key: str
    role: str
    document: str
    needs: tuple[str, ...] = ()


ROSTER = {
    "analyst": Role("analyst", "Research Analyst", "research brief",
                    frozenset({"write_doc", "recall", "summarize_text", "read_file", "fetch_url"})),
    "strategist": Role("strategist", "Chief Strategist", "business plan"),
    "pm": Role("pm", "Product Manager", "product spec"),
    "marketer": Role("marketer", "Head of Marketing", "go-to-market plan"),
    "cfo": Role("cfo", "Chief Financial Officer", "financial outline",
                frozenset({"write_doc", "recall", "summarize_text", "read_file", "calculate"})),
    "ceo": Role("ceo", "Chief Executive Officer", "executive summary"),
}

DEFAULT_TASKS = (
    Task("research", "analyst", "research brief"),
    Task("strategy", "strategist", "business plan", ("research",)),
    Task("product", "pm", "product spec", ("strategy",)),
    Task("marketing", "marketer", "go-to-market plan", ("strategy", "product")),
    Task("finance", "cfo", "financial outline", ("strategy",)),
    Task("summary", "ceo", "executive summary", ("research", "strategy", "product", "marketing", "finance")),
)


@dataclass
class TaskOutcome:
    task: str
    role: str
    title: str
    file: str
    output: str
    brain: str
    degraded: bool


@dataclass
class CompanyRun:
    name: str
    mission: str
    folder: str
    outcomes: list[TaskOutcome] = field(default_factory=list)
    report: str = ""


class Company:
    def __init__(self, agent: Agent, workspace: Workspace, memory=None,
                 roster: dict[str, Role] | None = None, tasks: tuple[Task, ...] = DEFAULT_TASKS,
                 max_parallel: int = 3) -> None:
        self.max_parallel = max(1, min(max_parallel, 8))
        self.agent, self.workspace, self.memory = agent, workspace, memory
        self.roster = roster or ROSTER
        self.tasks = tasks
        self._check_graph()

    def _check_graph(self) -> None:
        done: set[str] = set()
        for t in self.tasks:  # tasks must be listed in a valid execution order
            if t.role not in self.roster:
                raise ValueError(f"task '{t.key}' uses unknown role '{t.role}'")
            if not set(t.needs) <= done:
                raise ValueError(f"task '{t.key}' depends on a task that is not defined before it")
            done.add(t.key)

    def run(self, mission: str, name: str = "") -> CompanyRun:
        mission = " ".join(mission.split())
        if not 3 <= len(mission) <= 300:
            raise ValueError("mission must be 3-300 characters")
        slug = slugify(name or mission)
        run = CompanyRun(name=name.strip() or slug, mission=mission, folder=f"company-{slug}")
        results: dict[str, str] = {}

        def execute(task: Task) -> TaskOutcome:
            role = self.roster[task.role]
            context = "\n\n".join(results[k] for k in task.needs)
            res = self.agent.with_role(role.title, role.tools).run(f"Write a {task.document} for {mission}", context)
            rel = self.workspace.write(f"{run.folder}/{task.key}.md", res.answer)
            return TaskOutcome(task.key, role.key, role.title, rel, res.answer, res.brain, res.degraded)

        pending = list(self.tasks)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while pending:  # run every task whose dependencies are done, concurrently
                ready = [t for t in pending if all(k in results for k in t.needs)]
                for task, outcome in zip(ready, pool.map(execute, ready)):
                    results[task.key] = outcome.output
                    run.outcomes.append(outcome)
                pending = [t for t in pending if t.key not in results]
        order = {t.key: i for i, t in enumerate(self.tasks)}
        run.outcomes.sort(key=lambda o: order[o.task])
        run.report = self._report(run)
        self.workspace.write(f"{run.folder}/REPORT.md", run.report)
        if self.memory is not None:
            self.memory.add(f"Company '{run.name}' was formed for: {mission}. Documents are in {run.folder}/")
        return run

    @staticmethod
    def _report(run: CompanyRun) -> str:
        lines = [f"# {run.name}: company report", "", f"**Mission:** {run.mission}", "",
                 "| Task | Owner | Output | Brain |", "|---|---|---|---|"]
        for o in run.outcomes:
            note = " (fallback)" if o.degraded else ""
            lines.append(f"| {o.task} | {o.title} | [{o.file.split('/')[-1]}]({o.file.split('/')[-1]}) | {o.brain}{note} |")
        return "\n".join(lines) + "\n"
