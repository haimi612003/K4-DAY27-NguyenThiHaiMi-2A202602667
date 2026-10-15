"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    >= 73%       auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.replay import fetch_threads
from common.db import db_path
from common.github import parse_pr_url
from common.schemas import PRAnalysis
from exercises.exercise_4_audit import build_graph


load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")

RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}
DEMO_PR_BASE = "https://github.com/VinUni-AI20k/PR-Demo/pull/"


def normalize_pr_url(raw: str) -> str:
    """Accept a full PR URL, or a demo-repo shorthand like "1", "#1", "PR #1".

    Raises ValueError with a readable message for anything else.
    """
    raw = raw.strip()
    m = re.fullmatch(r"(?:PR)?\s*#?\s*(\d+)", raw, flags=re.IGNORECASE)
    if m:
        return DEMO_PR_BASE + m.group(1)
    parse_pr_url(raw)
    return raw


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "stalled" not in st.session_state:
    st.session_state.stalled = False
if "error" not in st.session_state:
    st.session_state.error = None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None, *, continue_run: bool = False):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}.

    - new run:          resume_value=None, continue_run=False
    - answer interrupt: resume_value=<reviewer answer>
    - crash recovery:   continue_run=True (re-enter from the last checkpoint)
    """
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if continue_run:
            return await app.ainvoke(None, cfg)
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str):
    """Read a thread's latest checkpoint: (state values, pending interrupt payload, has_next)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    interrupts = [i for task in snap.tasks for i in task.interrupts]
    return snap.values, (interrupts[0].value if interrupts else None), bool(snap.next)


def apply_result(result: dict) -> None:
    """Store a graph result in session state: either a pending interrupt or the final state."""
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result
    st.session_state.stalled = False


def drive(**kwargs) -> None:
    """Run/resume the current thread; on failure keep the message for display instead of crashing.

    The checkpoint keeps whatever finished before the error, so the session is marked
    stalled and "Resume run" retries from there (e.g. after topping up LLM credits).
    """
    try:
        apply_result(asyncio.run(run_graph(
            st.session_state.pr_url, st.session_state.thread_id, **kwargs,
        )))
        st.session_state.error = None
    except Exception as e:
        st.session_state.error = f"{type(e).__name__}: {e}"
        st.session_state.stalled = True


def open_thread(thread_id: str) -> None:
    """Point the UI at an existing thread (sidebar click or ?thread=... after a reload)."""
    values, payload, has_next = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = values.get("pr_url", "")
    st.session_state.interrupt_payload = payload
    st.session_state.final = values if payload is None and not has_next and values else None
    # Checkpoint exists but the run stopped mid-graph (e.g. process killed) — offer to resume.
    st.session_state.stalled = payload is None and has_next
    st.session_state.error = None
    st.query_params["thread"] = thread_id


# Reloading the page drops st.session_state; the thread id in the URL brings the session back.
if st.session_state.thread_id is None and st.query_params.get("thread"):
    open_thread(st.query_params["thread"])


st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    threads = asyncio.run(fetch_threads(limit=20))
    if not threads:
        st.caption("No sessions yet — run a review.")
    else:
        rows = [
            {
                "risk": f"{RISK_ICON.get(t['worst_risk'], '')} {t['worst_risk']}",
                "PR": "#" + t["pr_url"].rstrip("/").rsplit("/", 1)[-1],
                "thread": t["thread_id"][:8],
                "last event": str(t["last_event"])[:19].replace("T", " "),
            }
            for t in threads
        ]
        selection = st.dataframe(
            rows, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row", key="threads_table",
        )
        picked = selection.selection.rows
        if picked:
            chosen = threads[picked[0]]["thread_id"]
            if chosen != st.session_state.thread_id:
                open_thread(chosen)
                st.rerun()
    if st.session_state.thread_id:
        st.caption(f"Current thread: `{st.session_state.thread_id}`")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        help="Full GitHub PR URL, or a demo-repo shorthand: `1`, `#2`, `PR #1`.",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", width="stretch"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risks**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(q, key=f"escalation_q{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if all(a.strip() for a in answers.values()):
                return {q: a.strip() for q, a in answers.items()}
            st.warning("Please answer every question before submitting.")
    return None


def render_review(analysis: PRAnalysis) -> None:
    st.markdown(analysis.summary)
    for c in analysis.comments:
        st.markdown(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    try:
        url = normalize_pr_url(pr_url)
    except ValueError:
        st.error(f"`{pr_url}` is not a PR. Enter a full URL like "
                 f"`{DEMO_PR_BASE}1`, or just `1` / `#2` for the demo repo.")
        st.stop()
    st.session_state.pr_url = url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.query_params["thread"] = st.session_state.thread_id

    with st.spinner("Fetching PR + asking the LLM..."):
        drive()
    st.rerun()

if st.session_state.error:
    st.error(f"Run failed — {st.session_state.error}")

if st.session_state.stalled:
    st.warning("This session stopped before finishing (no pending question). "
               "Resume it from the last checkpoint?")
    if st.button("Resume run"):
        with st.spinner("Resuming from checkpoint..."):
            drive(continue_run=True)
        st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            drive(resume_value=answer)
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if isinstance(analysis, dict):
        analysis = PRAnalysis(**analysis)
    pr_link = st.session_state.pr_url

    if action.startswith("auto"):
        st.success(f"Auto-approved — confidence {analysis.confidence:.0%}. "
                   "The agent posted the review itself; no reviewer action needed.")
        render_review(analysis)
        st.link_button("View comment on GitHub", pr_link)
    elif action.startswith("committed"):
        if final.get("escalation_answers"):
            st.success(f"Refined review posted after escalation — confidence now {analysis.confidence:.0%}")
            render_review(analysis)
            with st.expander("Reviewer answers"):
                for q, a in final["escalation_answers"].items():
                    st.markdown(f"**{q}**\n\n> {a}")
        else:
            st.success(f"Approved — review posted (confidence {analysis.confidence:.0%})")
        st.link_button("View comment on GitHub", pr_link)
    elif action == "rejected":
        if final.get("human_choice") == "edit":
            st.info("Edit requested — no comment posted. Reviewer feedback recorded in the audit trail.")
        else:
            st.warning("Rejected — no comment posted")
        if final.get("human_feedback"):
            st.caption(f"Reviewer feedback: {final['human_feedback']}")
    else:
        st.error(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
