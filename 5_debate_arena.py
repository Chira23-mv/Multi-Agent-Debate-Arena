"""
Multi-Agent Debate Arena (CrewAI + Streamlit)
---------------------------------------------
Two agents argue opposite sides of a topic. A third agent judges every round
and picks a winner at the end.

Run:      streamlit run 5_debate_arena.py
Install:  pip install crewai streamlit

Model string examples (CrewAI uses LiteLLM-style names):
  gpt-4o-mini
  gemini/gemini-2.0-flash
  groq/llama-3.3-70b-versatile
Paste the matching API key in the sidebar.
"""
import json
import re

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task

st.set_page_config(page_title="Debate Arena", page_icon="⚔️", layout="centered")
st.title("⚔️ Multi-Agent Debate Arena")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", "gpt-4o-mini")
    api_key = st.text_input("API key", type="password")
    rounds = st.slider("Rounds", 1, 4, 2)
    st.caption("Each round = 1 argument from each side + judge scoring.")

topic = st.text_input("Debate topic", "Should college students be allowed to use AI tools in exams?")


def make_agents(llm):
    pro = Agent(
        role="Pro Debater",
        goal="Argue convincingly IN FAVOUR of the topic using logic, examples and evidence.",
        backstory="A sharp university debate champion who always finds the strongest case for the proposition.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    con = Agent(
        role="Con Debater",
        goal="Argue convincingly AGAINST the topic using logic, examples and evidence.",
        backstory="A fearless debater who dismantles weak arguments and exposes risks others miss.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    judge = Agent(
        role="Impartial Judge",
        goal="Score each argument fairly on logic, evidence and rebuttal quality.",
        backstory="A veteran debate adjudicator who rewards substance over style and never takes sides.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    return pro, con, judge


def ask(agent: Agent, prompt: str) -> str:
    """Run one agent on one prompt through a tiny Crew so we can show results live."""
    task = Task(description=prompt, expected_output="A clear response following the instructions.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff()).strip()


def parse_scores(text: str) -> dict:
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        return {"pro": float(data["pro"]), "con": float(data["con"]), "comment": str(data.get("comment", ""))}
    except Exception:
        return {"pro": 5.0, "con": 5.0, "comment": text[:300]}


if st.button("Start debate ⚔️", type="primary"):
    if not api_key:
        st.error("Add your API key in the sidebar.")
        st.stop()

    llm = LLM(model=model, api_key=api_key, temperature=0.7)
    pro, con, judge = make_agents(llm)

    transcript = ""
    totals = {"pro": 0.0, "con": 0.0}
    last_pro = last_con = ""

    try:
        for r in range(1, rounds + 1):
            st.markdown(f"### Round {r}")

            if r == 1:
                p_prompt = f"Topic: {topic}\nGive your opening argument FOR it in under 120 words."
            else:
                p_prompt = (f"Topic: {topic}\nYour opponent just said:\n{last_con}\n"
                            "Rebut their points and strengthen your case FOR the topic in under 120 words.")
            with st.spinner("Pro is thinking..."):
                last_pro = ask(pro, p_prompt)
            with st.chat_message("assistant", avatar="🟢"):
                st.markdown(f"**Pro:** {last_pro}")

            c_prompt = (f"Topic: {topic}\nYour opponent just said:\n{last_pro}\n"
                        "Rebut them and argue AGAINST the topic in under 120 words.")
            with st.spinner("Con is thinking..."):
                last_con = ask(con, c_prompt)
            with st.chat_message("assistant", avatar="🔴"):
                st.markdown(f"**Con:** {last_con}")

            transcript += f"\n[Round {r}]\nPRO: {last_pro}\nCON: {last_con}\n"

            with st.spinner("Judge is scoring..."):
                raw = ask(judge, (
                    f"Topic: {topic}\nRound {r} arguments:\nPRO: {last_pro}\nCON: {last_con}\n\n"
                    'Score each side from 1 to 10. Reply with ONLY JSON like '
                    '{"pro": 7, "con": 6, "comment": "one sentence why"}'))
            s = parse_scores(raw)
            totals["pro"] += s["pro"]
            totals["con"] += s["con"]
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(f"**Judge:** Pro {s['pro']:.0f} – Con {s['con']:.0f}. {s['comment']}")

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("🟢 Pro total", f"{totals['pro']:.0f}")
        c2.metric("🔴 Con total", f"{totals['con']:.0f}")

        with st.spinner("Judge is writing the final verdict..."):
            verdict = ask(judge, (
                f"Topic: {topic}\nFull debate:\n{transcript}\n\n"
                f"Running scores: Pro {totals['pro']:.0f}, Con {totals['con']:.0f}.\n"
                "Declare a winner and explain the decision in under 100 words."))
        st.success(f"🏆 **Verdict:** {verdict}")
        st.balloons()

    except Exception as e:
        st.error(f"Something went wrong: {e}")
