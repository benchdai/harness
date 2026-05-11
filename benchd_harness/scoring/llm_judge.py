"""
LLM-based answerer and judge for Bench'd harness.

The evaluation pipeline:
1. Memory system retrieves evidence via recall()
2. Answerer LLM generates a natural language answer from retrieved evidence + query
3. Judge LLM evaluates the answer against expected answer → CORRECT/INCORRECT

Both models are locked per benchmark version. Temperature is always 0.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

ANSWERER_PROMPT = """You are answering a question using ONLY the retrieved memories below.

Retrieved memories:
{retrieved_memories}

Question: {question}

Instructions:
- Answer the question using ONLY information from the retrieved memories above.
- If the retrieved memories do not contain enough information to answer, say "Insufficient information in memory."
- Be concise and direct. Give the specific answer, not an explanation.
- Do not make up information not present in the retrieved memories."""

JUDGE_PROMPT = """You are a judge evaluating whether an answer is correct.

Question: {question}
Expected Answer: {expected_answer}
Given Answer: {given_answer}

Instructions:
- Determine if the Given Answer is semantically equivalent to the Expected Answer.
- The Given Answer does not need to match word-for-word. It needs to contain the correct information.
- If the Given Answer contains the correct fact even with extra context, judge it as CORRECT.
- If the Given Answer says "insufficient information" or equivalent, judge it as INCORRECT.
- If the Given Answer is partially correct but missing key information, judge it as INCORRECT.

Respond with EXACTLY one line in this format:
JUDGMENT: CORRECT
or
JUDGMENT: INCORRECT

Then on the next line, briefly explain your reasoning in one sentence."""


@dataclass
class LLMJudgeConfig:
    """Configuration for the answerer + judge pipeline."""
    answerer_model: str = "openai/gpt-4o-mini"
    judge_model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_answer_tokens: int = 256
    max_judge_tokens: int = 128
    api_base: str = "https://openrouter.ai/api/v1"

    @property
    def api_key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLM judge requires an API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY."
            )
        return key


@dataclass
class AnswererResult:
    """Result from the answerer LLM step."""
    answer: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class JudgeResult:
    """Result from the judge LLM step."""
    correct: bool
    reasoning: str
    model: str
    input_tokens: int
    output_tokens: int


def _call_llm(
    config: LLMJudgeConfig,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Call an OpenAI-compatible API. Returns the response dict."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required for LLM judge. pip install openai")

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.api_base,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    usage = response.usage

    return {
        "content": choice.message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


def generate_answer(
    query: str,
    retrieved_memories: str,
    config: Optional[LLMJudgeConfig] = None,
) -> AnswererResult:
    """
    Step 2 of the pipeline: generate a natural language answer
    from retrieved memories using a locked answerer LLM.
    """
    if config is None:
        config = LLMJudgeConfig()

    prompt = ANSWERER_PROMPT.format(
        retrieved_memories=retrieved_memories,
        question=query,
    )

    resp = _call_llm(config, config.answerer_model, prompt, config.max_answer_tokens)

    return AnswererResult(
        answer=resp["content"].strip(),
        model=config.answerer_model,
        input_tokens=resp["input_tokens"],
        output_tokens=resp["output_tokens"],
    )


def judge_answer(
    query: str,
    expected_answer: str,
    given_answer: str,
    config: Optional[LLMJudgeConfig] = None,
) -> JudgeResult:
    """
    Step 3 of the pipeline: judge whether the generated answer
    is correct using a locked judge LLM.
    """
    if config is None:
        config = LLMJudgeConfig()

    prompt = JUDGE_PROMPT.format(
        question=query,
        expected_answer=expected_answer,
        given_answer=given_answer,
    )

    resp = _call_llm(config, config.judge_model, prompt, config.max_judge_tokens)

    content = resp["content"].strip()

    # Parse judgment
    correct = False
    reasoning = content
    for line in content.split("\n"):
        line_upper = line.strip().upper()
        if "JUDGMENT:" in line_upper:
            correct = "CORRECT" in line_upper and "INCORRECT" not in line_upper
            reasoning = content.replace(line, "").strip()
            break

    return JudgeResult(
        correct=correct,
        reasoning=reasoning or "No reasoning provided",
        model=config.judge_model,
        input_tokens=resp["input_tokens"],
        output_tokens=resp["output_tokens"],
    )
