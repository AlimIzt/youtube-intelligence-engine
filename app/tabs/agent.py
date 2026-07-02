"""Ask questions answered by RAG + LLM agents (needs Ollama)."""
from __future__ import annotations

import streamlit as st

from config import settings
from app.data import Scope
from app.theme import explain


def render(scope: Scope) -> None:
    st.subheader("Ask about the comments")
    explain("Type any question — the system retrieves the most relevant comments "
            "(RAG) and a local LLM answers using only them. Needs Ollama running.")

    c1, c2 = st.columns(2)
    mode = c1.radio(
        "Agent",
        ["Tool-calling (single)", "Multi-agent (router)",
         "Supervisor (full report)", "Swarm (reflection)",
         "DSPy (Chain-of-Thought)", "HyDE retrieval"],
        help="Different agent architectures: a single tool-calling agent, a "
             "router that picks a specialist, a supervisor that writes a full "
             "report, a swarm that critiques its own answer, DSPy "
             "chain-of-thought, and HyDE (retrieves via an LLM-invented "
             "hypothetical comment).")
    retrieval = c2.radio("Retrieval", ["semantic", "mmr"], horizontal=True,
                         help="semantic = most similar comments; mmr = similar "
                              "but diverse, avoids near-duplicates.")
    rerank = st.checkbox("Cross-encoder rerank", value=False,
                         help="A second, slower model re-orders the retrieved "
                              "comments for better relevance.")

    q = st.text_input("Your question", "What do people think about the graphics?")
    if st.button("Ask"):
        with st.spinner("Thinking ..."):
            try:
                if mode.startswith("Multi"):
                    from src.agents.multi_agent import ask_multi
                    cat, resp = ask_multi(q)
                    st.info(f"Routed to: **{cat}**")
                    st.write(resp)
                elif mode.startswith("Supervisor"):
                    from src.agents.supervisor import generate_report
                    st.write(generate_report(q))
                elif mode.startswith("Swarm"):
                    from src.agents.swarm import ask_swarm
                    st.write(ask_swarm(q))
                elif mode.startswith("HyDE"):
                    from src.rag.hyde import hyde_answer, hypothetical_answers
                    hypos = hypothetical_answers(q)
                    st.write(hyde_answer(q, hypotheticals=hypos))
                    with st.expander("Hypothetical comment used for retrieval"):
                        st.write(hypos[0])
                elif mode.startswith("DSPy"):
                    from src.rag.dspy_qa import answer_question_dspy
                    r = answer_question_dspy(q)
                    st.write(r.answer)
                    st.caption(f"Self-rated confidence: {r.confidence}")
                    with st.expander("Reasoning (Chain-of-Thought)"):
                        st.write(getattr(r, "reasoning", ""))
                else:
                    from src.rag.generation import answer_question
                    from src.rag.retrieval import mmr_retriever, semantic_retriever

                    retr = mmr_retriever() if retrieval == "mmr" else semantic_retriever()
                    if rerank:
                        from src.rag.postretrieval import rerank as rr
                        docs = rr(q, retr.invoke(q), top_k=settings.top_k)
                        from src.rag.generation import QA_PROMPT, get_llm
                        from src.rag.retrieval import format_docs
                        from langchain_core.output_parsers import StrOutputParser

                        chain = QA_PROMPT | get_llm() | StrOutputParser()
                        st.write(chain.invoke({"question": q,
                                               "context": format_docs(docs)}))
                    else:
                        st.write(answer_question(q, retriever=retr))
            except Exception as e:
                st.error(f"Agent error: {e}")
