"""Minimal example: capture a Jev run into a DREAM experience store."""

from jev_ultrafast import Agent


def verify(page):
    return {"passed": "Python" in page.get("title", "") or "Python" in page.get("text", "")[:1000]}


with Agent(
    "https://en.wikipedia.org/wiki/Main_Page",
    "Find and open the Wikipedia article about Python programming language.",
    verifier=verify,
    dream_store=".jev/experience.jsonl",
) as agent:
    for state in agent.run():
        print(state["status"], state["elapsed_ms"])
