"""Document templates: structured scaffolds filled from the subject, memory and earlier work."""
from __future__ import annotations

import re

from .nlp import keywords, summarize_text

# alias phrase -> template name (longest alias wins when routing)
ALIASES = {
    "research brief": "research_brief", "research": "research_brief", "brief": "research_brief",
    "business plan": "business_plan", "strategy": "business_plan", "plan": "business_plan",
    "product spec": "product_spec", "spec": "product_spec", "prd": "product_spec",
    "go-to-market plan": "go_to_market", "go to market": "go_to_market", "go-to-market": "go_to_market",
    "gtm": "go_to_market", "marketing plan": "go_to_market", "launch plan": "go_to_market",
    "financial outline": "financial_outline", "financial plan": "financial_outline",
    "budget": "financial_outline", "finance": "financial_outline",
    "executive summary": "exec_summary", "summary": "exec_summary", "memo": "exec_summary", "report": "exec_summary",
}

_B = "- "
SECTIONS: dict[str, tuple[str, list[tuple[str, list[str]]]]] = {
    "research_brief": ("Research Brief", [
        ("Research question", ["How should we approach: {S}?"]),
        ("What to find out", [_B + "Who has this problem, and how do they solve it today?", _B + "Size and growth of the market",
                              _B + "Existing alternatives and their weaknesses", _B + "Regulatory or technical constraints"]),
        ("Sources to check", [_B + "Customer interviews (aim for 10)", _B + "Competitor sites, pricing pages and reviews",
                              _B + "Industry reports and public filings", _B + "Search/forum trends"]),
        ("Open questions", [_B + "What would make us stop pursuing {S}?", _B + "What is the cheapest experiment that could prove demand?"]),
    ]),
    "business_plan": ("Business Plan", [
        ("Vision", ["A one-sentence outcome for {S} that a customer would pay for."]),
        ("Problem", [_B + "Who feels the pain most?", _B + "What does it cost them today (time, money, risk)?"]),
        ("Target customers", [_B + "Primary segment (be specific)", _B + "Secondary segment", _B + "Who is explicitly NOT a customer"]),
        ("Solution", [_B + "Core promise of {S}", _B + "Three differentiators vs. alternatives"]),
        ("Revenue model", [_B + "Pricing approach (subscription / usage / one-off)", _B + "Expected first-year revenue streams"]),
        ("Milestones", [_B + "Day 30: validated problem + 10 interviews", _B + "Day 60: working prototype with 3 pilot users",
                        _B + "Day 90: first paying customers"]),
        ("Risks and mitigations", [_B + "Market risk: no demand -> run a fake-door test first", _B + "Execution risk: scope creep -> weekly scope review",
                                   _B + "Financial risk: runway -> track burn monthly"]),
    ]),
    "product_spec": ("Product Specification", [
        ("Goal", ["Deliver {S} so the primary user can complete their key job in minutes."]),
        ("Users and use cases", [_B + "Primary user story", _B + "Edge-case user story"]),
        ("Requirements", [_B + "MUST: core workflow end to end", _B + "SHOULD: onboarding and basic analytics", _B + "COULD: integrations"]),
        ("Non-goals", [_B + "Anything not needed to validate the core promise of {S}"]),
        ("Success metrics", [_B + "Activation rate", _B + "Week-4 retention", _B + "Time to first value"]),
        ("Rollout", [_B + "Internal alpha -> pilot cohort -> general availability, each gated by metrics"]),
    ]),
    "go_to_market": ("Go-To-Market Plan", [
        ("Positioning", ["For [segment] who need [job], {S} is [category] that [key benefit], unlike [alternative]."]),
        ("Channels", [_B + "Owned: website, email list, docs", _B + "Earned: communities, press, partnerships", _B + "Paid: only after organic conversion is proven"]),
        ("Messaging", [_B + "Headline benefit", _B + "Three proof points", _B + "Primary objection and answer"]),
        ("Launch timeline", [_B + "T-4 weeks: waitlist + content", _B + "T-0: launch announcement", _B + "T+4 weeks: review funnel and iterate"]),
        ("KPIs", [_B + "Visitors -> signups -> activated -> paying conversion at each step"]),
    ]),
    "financial_outline": ("Financial Outline", [
        ("Assumptions", [_B + "Price per customer: [fill in]", _B + "Monthly churn: [fill in]", _B + "Customer acquisition cost: [fill in]"]),
        ("Revenue drivers", [_B + "Customers x price x retention", _B + "Expansion revenue from {S} add-ons"]),
        ("Cost structure", [_B + "People", _B + "Infrastructure and tooling", _B + "Sales and marketing", _B + "Support and compliance"]),
        ("Unit economics", ["| Metric | Formula | Target |", "|---|---|---|", "| LTV | ARPA / churn | > 3x CAC |",
                            "| CAC payback | CAC / (ARPA x margin) | < 12 months |"]),
        ("Break-even", ["Break-even customers = fixed monthly costs / (price - variable cost per customer)."]),
    ]),
    "exec_summary": ("Executive Summary", [
        ("Objective", ["{S}"]),
        ("Decisions needed", [_B + "Approve scope and budget", _B + "Pick the primary customer segment", _B + "Assign an owner per workstream"]),
        ("Next steps", [_B + "This week: customer interviews", _B + "This month: prototype and pilot", _B + "This quarter: first revenue"]),
    ]),
}


def clean_subject(subject: str) -> str:
    return re.sub(r"\s+", " ", subject).strip()[:200] or "the project"


_SKIP_SECTIONS = {"Key themes", "Inputs from earlier work", "Relevant memory"}


def _digest(context: str) -> list[str]:
    """One bullet per upstream markdown document (title + section names); plain text is summarized."""
    bullets: list[str] = []
    title, sections = None, []

    def flush() -> None:
        if title:
            covers = f" (covers: {', '.join(sections)})" if sections else ""
            bullets.append(f"- {title}{covers}")

    for line in context.splitlines():
        if line.startswith("# "):
            flush()
            title, sections = line[2:].strip(), []
        elif line.startswith("## ") and title and line[3:].strip() not in _SKIP_SECTIONS:
            sections.append(line[3:].strip())
    flush()
    return list(dict.fromkeys(bullets)) or [summarize_text(context, 4)]



def render(template: str, subject: str, context: str = "", memories: tuple[str, ...] = ()) -> str:
    if template not in SECTIONS:
        raise ValueError(f"unknown template '{template}'")
    subject = clean_subject(subject)
    title, sections = SECTIONS[template]
    out = [f"# {title}: {subject}", ""]
    if context.strip():
        out += ["## Inputs from earlier work"] + _digest(context) + [""]
    if memories:
        out += ["## Relevant memory"] + [f"- {m}" for m in memories] + [""]
    kws = keywords(subject + " " + context, 6)
    if kws:
        out += ["## Key themes", ", ".join(kws), ""]
    for heading, lines in sections:
        out += [f"## {heading}"] + [ln.replace("{S}", subject) for ln in lines] + [""]
    return "\n".join(out).rstrip() + "\n"
