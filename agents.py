
# agents.py

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ddgs import DDGS

from prompts import (
    supervisor_prompt_template,
    researcher_prompt_template,
    writer_prompt_template,
    critique_prompt_template,
)

# --------------------------------------------------
# 1. Load Environment Variables
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# --------------------------------------------------
# 2. Initialize LLM
# --------------------------------------------------

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is missing. Add it to your project's .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=2048,
)

# --------------------------------------------------
# 3. Common LLM Helper
# --------------------------------------------------

def _call_llm(llm_obj, *args, **kwargs):
    """Invoke an LLM or another compatible callable."""

    if hasattr(llm_obj, "invoke"):
        return llm_obj.invoke(*args, **kwargs)

    if hasattr(llm_obj, "run"):
        return llm_obj.run(*args, **kwargs)

    if callable(llm_obj):
        return llm_obj(*args, **kwargs)

    raise AttributeError("Object has no supported invocation method.")


def _response_text(response):
    """Extract text from an LLM response."""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content)


# --------------------------------------------------
# 4. DuckDuckGo Search
# --------------------------------------------------

def search_web(query, max_results=5):
    """Search the web without a paid search API key."""

    with DDGS() as ddgs:
        return list(
            ddgs.text(query, max_results=max_results)
        )


# --------------------------------------------------
# 5. Supervisor
# --------------------------------------------------

def create_supervisor_chain():
    """Decide which agent should execute next."""

    def supervisor_invoke(state):
        research = state.get("research_findings", [])
        research_text = "\n---\n".join(research) if research else "No research yet."

        revision = state.get("revision_number", 0)
        has_research = bool(research)
        has_draft = bool(state.get("draft", "").strip())
        critique = state.get("critique_notes", "")

        if "APPROVED" in critique.upper() and has_draft:
            return {
                "next_step": "END",
                "task_description": "Report approved and complete",
            }

        if not has_research:
            return {
                "next_step": "researcher",
                "task_description": f"Research the topic: {state.get('main_task', '')}",
            }

        if has_research and not has_draft:
            return {
                "next_step": "writer",
                "task_description": "Write the first draft based on research findings",
            }

        if has_draft and not critique:
            return {
                "next_step": "writer",
                "task_description": "Prepare draft for critique",
            }

        if critique and "APPROVED" not in critique.upper() and revision < 3:
            return {
                "next_step": "writer",
                "task_description": "Revise the draft based on critique feedback",
            }

        if revision >= 3:
            return {
                "next_step": "END",
                "task_description": "Maximum revisions reached",
            }

        prompt = supervisor_prompt_template.format(
            main_task=state.get("main_task", ""),
            research_findings=research_text,
            draft=state.get("draft", "No draft yet."),
            critique_notes=critique or "No critique yet.",
            revision_number=revision,
        )

        try:
            response = _call_llm(llm, prompt)
            content = _response_text(response).strip()

            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            decision = json.loads(content)

            if decision.get("next_step") in {
                "researcher", "writer", "END"
            }:
                return decision

        except Exception as exc:
            print(f"Supervisor parsing error: {exc}")

        return {
            "next_step": "writer",
            "task_description": "Continue with draft creation",
        }

    return supervisor_invoke


# --------------------------------------------------
# 6. Researcher
# --------------------------------------------------

def create_researcher_agent():
    """Search with DuckDuckGo and summarize with Groq."""

    def researcher_invoke(input_dict):
        query = input_dict.get("input", "").strip()

        if not query or query in {"Continue work", "Complete"}:
            query = "General research information"

        print(f"Researching: {query}")

        try:
            results = search_web(query, max_results=5)

            if not results:
                return {
                    "output": f"No web results found for: {query}",
                    "input": query,
                }

            formatted_results = []

            for result in results:
                title = result.get("title", "Untitled")
                url = result.get("href", "N/A")
                body = result.get("body", "")

                formatted_results.append(
                    f"Title: {title}\n"
                    f"Source: {url}\n"
                    f"Content: {body}\n"
                )

            raw_output = "\n---\n".join(formatted_results)

            summary_prompt = f"""
You are a careful research assistant.

Research topic: {query}

Summarize the following web search results in 5-7
useful bullet points.

Rules:
- Use only information supported by the results.
- Do not invent facts, statistics, or citations.
- Include relevant source URLs.
- Mention when the available information is insufficient.

Search results:
{raw_output}
"""

            response = _call_llm(llm, summary_prompt)
            summary = _response_text(response)

            return {
                "output": summary or raw_output,
                "input": query,
            }

        except Exception as exc:
            print(f"Research error: {exc}")
            raise RuntimeError(
                f"Research failed for '{query}': {exc}"
            ) from exc

    return researcher_invoke


# --------------------------------------------------
# 7. Writer
# --------------------------------------------------

# --------------------------------------------------
# 7. Writer
# --------------------------------------------------

def create_writer_chain():
    """Create or revise the research report with bounded prompt size."""

    MAX_RESEARCH_CHARS = 4500
    MAX_DRAFT_CHARS = 4500
    MAX_CRITIQUE_CHARS = 1200

    def writer_invoke(state):
        research = state.get("research_findings", [])

        if isinstance(research, list):
            research_text = "\n\n".join(str(item) for item in research)
        else:
            research_text = str(research)

        draft = str(state.get("draft", "") or "")
        critique = str(state.get("critique_notes", "") or "")
        main_task = str(state.get("main_task", "") or "")

        # Keep the prompt small enough for the model's token limit.
        research_text = research_text[:MAX_RESEARCH_CHARS]
        draft = draft[:MAX_DRAFT_CHARS]
        critique = critique[:MAX_CRITIQUE_CHARS]

        prompt = writer_prompt_template.format(
            main_task=main_task,
            research_findings=research_text,
            draft=draft,
            critique_notes=critique,
        )

        try:
            # Reduce the maximum response size for this call.
            writer_llm = llm.bind(max_tokens=2048)
            response = _call_llm(writer_llm, prompt)
            content = _response_text(response).strip()

            return content or "Draft in progress..."

        except Exception as exc:
            print(f"Writer error: {exc}")
            raise RuntimeError(
                f"Could not generate the report: {exc}"
            ) from exc

    return writer_invoke


# --------------------------------------------------
# 8. Critiquer
# --------------------------------------------------

def create_critique_chain():
    """Review the draft and provide revision feedback."""

    def critique_invoke(state):
        draft = state.get("draft", "")
        revision_num = state.get("revision_number", 0)

        if len(draft.strip()) < 100:
            return "APPROVED - Draft is minimal but acceptable."

        if revision_num >= 3:
            return "APPROVED - Maximum revisions reached."

        prompt = critique_prompt_template.format(
            main_task=state.get("main_task", ""),
            draft=draft,
        )

        try:
            response = _call_llm(llm, prompt)
            content = _response_text(response)

            return content or "APPROVED"

        except Exception as exc:
            print(f"Critique error: {exc}")
            raise RuntimeError(
                f"Could not review the report: {exc}"
            ) from exc

    return critique_invoke