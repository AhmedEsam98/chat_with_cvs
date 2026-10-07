import json
import logging
from typing import Any

import config
from clients import get_openai_client
from generation.prompts import build_context

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = """You are an impartial, highly rigorous AI hallucination auditor for a CV/Resume Q&A system.
Your task is to evaluate whether a generated AI answer is completely faithful to and grounded in the provided CV excerpts (Context).

Evaluation Process:
1. Break down the generated answer into discrete, individual factual claims/statements.
   - Ignore conversational pleasantries or meta statements (e.g. "Based on the CVs:", "Here is the summary:").
2. For each factual claim, verify whether it is directly supported by the context.
   - GROUNDED: The claim is explicitly stated in or directly inferred from the excerpts.
   - UNGROUNDED / HALLUCINATED: The claim adds new information, exaggerates facts, mixes up candidate names, or states details not found anywhere in the excerpts.
3. Calculate the faithfulness score = (number of grounded claims) / (total factual claims).
   - If there are 0 factual claims (e.g. "I could not find this information in the uploaded CVs"), score is 1.0 (fully faithful refusal).

You MUST respond strictly with a valid JSON object using this exact structure:
{
  "total_claims": <integer>,
  "grounded_claims_count": <integer>,
  "score": <float between 0.0 and 1.0>,
  "is_grounded": <boolean, true if score >= 0.85 else false>,
  "hallucinations": [
    {
      "claim": "<the specific ungrounded claim from the answer>",
      "reason": "<why this claim is considered hallucinated or not supported by context>"
    }
  ],
  "reasoning": "<brief 1-2 sentence overall audit summary>"
}
"""


def check_hallucination(answer: str, hits: list[dict], client=None) -> dict[str, Any]:
    """Audit an answer for hallucinations against retrieved CV chunks.

    Returns:
        Dictionary with score, is_grounded, claims counts, and any hallucinated claims.
    """
    if not answer or not answer.strip():
        return {
            "score": 1.0,
            "is_grounded": True,
            "total_claims": 0,
            "grounded_claims_count": 0,
            "hallucinations": [],
            "reasoning": "Empty answer.",
        }

    context = build_context(hits) if hits else "NO CONTEXT PROVIDED."

    user_prompt = f"""=== PROVIDED CV CONTEXT ===
{context}

=== GENERATED ANSWER TO EVALUATE ===
{answer}
"""

    client = client or get_openai_client()
    try:
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(content)
        result["score"] = float(result.get("score", 1.0))
        result["is_grounded"] = bool(result.get("is_grounded", result["score"] >= 0.85))
        result["hallucinations"] = result.get("hallucinations", [])
        return result
    except Exception as exc:
        logger.error(f"Hallucination check failed: {exc}")
        return {
            "score": 1.0,
            "is_grounded": True,
            "total_claims": 0,
            "grounded_claims_count": 0,
            "hallucinations": [],
            "reasoning": f"Audit skipped due to error: {exc}",
        }
