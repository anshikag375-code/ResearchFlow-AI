
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from graph import app as research_graph


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def get_secret(name: str) -> str:
    """Read a secret from environment variables or Streamlit secrets."""
    value = os.getenv(name, "").strip()

    if value:
        return value

    try:
        value = str(st.secrets.get(name, "")).strip()
        return value
    except Exception:
        return ""


# --------------------------------------------------
# 2. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ResearchFlow AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ResearchFlow AI")

st.markdown(
    """
    ### Multi-Agent Research Assistant

    Enter a research topic and let a team of AI agents
    research, write, and critique a report.

    **Technology:** Streamlit, LangGraph, LangChain, Groq,
    and DuckDuckGo Search.

    **Workflow:** Supervisor → Researcher → Writer → Critiquer
    """
)

st.divider()


# --------------------------------------------------
# 3. Check Groq API key
# --------------------------------------------------
groq_key = get_secret("GROQ_API_KEY")

if not groq_key:
    st.error("GROQ_API_KEY is missing.")

    st.markdown(
        """
        Create a `.env` file in the same folder as `app.py`
        and add your Groq API key:

        ```env
        GROQ_API_KEY=your_groq_api_key
        ```

        Save the file and restart the application.
        """
    )

    st.stop()


# --------------------------------------------------
# 4. Sidebar configuration
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    max_iterations = st.slider(
        "Maximum workflow steps",
        min_value=5,
        max_value=25,
        value=15,
        help="Limits the LangGraph workflow execution.",
    )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        1. **Supervisor:** routes tasks.
        2. **Researcher:** gathers web information.
        3. **Writer:** creates the report.
        4. **Critiquer:** evaluates the draft.
        5. **Revision:** improves the report when needed.
        """
    )

    st.divider()

    st.caption(
        "Built with Streamlit, LangGraph, LangChain, "
        "Groq and DuckDuckGo."
    )


# --------------------------------------------------
# 5. Research input
# --------------------------------------------------
st.header("🔍 Start your research")

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Impact of generative AI on software engineering",
    max_chars=500,
)

start_research = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# 6. Run the agent workflow
# --------------------------------------------------
if start_research:
    if not topic.strip():
        st.warning("Please enter a research topic.")

    else:
        initial_state = {
            "main_task": topic.strip(),
            "research_findings": [],
            "draft": "",
            "critique_notes": "",
            "revision_number": 0,
            "next_step": "",
            "current_sub_task": "",
        }

        config = {
            "recursion_limit": 25,
        }

        accumulated_state = initial_state.copy()
        step_count = 0
        error_occurred = False

        progress_bar = st.progress(0)
        status = st.empty()
        activity = st.container()

        try:
            status.info(
                "🤖 AI agents are starting the research..."
            )

            with activity:
                st.subheader("📋 Agent Activity Log")

                for update in research_graph.stream(
                    initial_state,
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in update.items():
                        if not isinstance(node_output, dict):
                            continue

                        step_count += 1
                        accumulated_state.update(node_output)

                        progress_bar.progress(
                            min(step_count / max_iterations, 1.0)
                        )

                        st.markdown(
                            f"### 🤖 {node_name.replace('_', ' ').title()}"
                        )
                        st.caption(f"Workflow step {step_count}")

                        # Supervisor output
                        if node_name == "supervisor":
                            st.write(
                                "**Next step:**",
                                node_output.get(
                                    "next_step", "Not specified"
                                ),
                            )

                            st.write(
                                "**Current task:**",
                                node_output.get(
                                    "current_sub_task", "Not specified"
                                ),
                            )

                        # Researcher output
                        elif node_name == "researcher":
                            findings = accumulated_state.get(
                                "research_findings", []
                            )

                            if findings:
                                latest = findings[-1]

                                if isinstance(latest, str):
                                    st.success(
                                        "Research findings received"
                                    )

                                    with st.expander(
                                        "View latest research findings"
                                    ):
                                        st.markdown(latest)

                        # Writer output
                        elif node_name == "writer":
                            draft = accumulated_state.get("draft", "")

                            if draft:
                                st.success(
                                    f"Draft updated: {len(draft)} characters"
                                )

                                with st.expander(
                                    "Preview current draft"
                                ):
                                    st.markdown(draft[:5000])

                        # Critiquer output
                        elif node_name == "critiquer":
                            critique = accumulated_state.get(
                                "critique_notes", ""
                            )

                            if isinstance(critique, str) and critique:
                                if "APPROVED" in critique.upper():
                                    st.success(
                                        "Draft approved by the critiquer"
                                    )
                                else:
                                    st.warning("Critique received")

                                with st.expander("View critique"):
                                    st.markdown(critique)

                        st.divider()
                        time.sleep(0.15)

            progress_bar.progress(1.0)
            status.success("✅ Workflow execution finished.")

        except Exception as exc:
            error_occurred = True

            status.error(
                "❌ The research workflow encountered an error."
            )

            st.error(f"{type(exc).__name__}: {exc}")
            st.exception(exc)

        # ------------------------------------------
        # 7. Display final report
        # ------------------------------------------
        final_draft = accumulated_state.get("draft", "")

        if (
            isinstance(final_draft, str)
            and len(final_draft.strip()) > 50
        ):
            st.divider()
            st.header("📄 Research Report")
            st.markdown(final_draft)

            findings = accumulated_state.get(
                "research_findings", []
            )

            if not isinstance(findings, list):
                findings = []

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Word count",
                    len(final_draft.split()),
                )

            with col2:
                st.metric(
                    "Research findings",
                    len(findings),
                )

            with col3:
                st.metric(
                    "Revision number",
                    accumulated_state.get("revision_number", 0),
                )

            with st.expander("View all research findings"):
                if findings:
                    for index, finding in enumerate(
                        findings, start=1
                    ):
                        st.markdown(f"**Finding {index}**")
                        st.write(finding)
                        st.divider()
                else:
                    st.info(
                        "No research findings were returned."
                    )

            safe_topic = re.sub(
                r"[^a-zA-Z0-9_-]+",
                "_",
                topic.strip(),
            ).strip("_")[:60]

            st.download_button(
                label="📥 Download research report",
                data=final_draft,
                file_name=(
                    f"research_report_{safe_topic or 'report'}.md"
                ),
                mime="text/markdown",
                use_container_width=True,
            )

        elif not error_occurred:
            st.warning(
                "The workflow finished, but no complete report "
                "was found. Check graph.py and the agent outputs."
            )


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------
st.divider()
st.caption(
    "ResearchFlow AI · Multi-agent research and report generation"
)
