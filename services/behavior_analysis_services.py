import ollama
import json
import re
from typing import Optional


def analyze_behavior_text(text: str) -> dict:
    """
    Analyze behavioral markers using an LLM via Ollama.
    Returns a JSON with the markers.
    """

    prompt = f"""

    Answer:
    {text}

    Return ONLY a valid JSON with the fields:
    - tone
    - predominant_emotion
    - confidence_level
    - stress_signals
    - behavioral_summary
    """

    response = ollama.generate(
        model="phi3",
        prompt=prompt
    )

    raw_output = response["response"].strip()

    def extract_json_block(text: str) -> Optional[str]:
        """Attempt to clean code fences and extra instructions, returning only the JSON."""
        cleaned = text

        # Remove ```json ... ``` fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned, count=1)
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]

        cleaned = cleaned.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return None

    json_candidate = extract_json_block(raw_output)

    # Ensure the JSON returned is a dictionary
    if json_candidate:
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    return {
        "error": "Failed to convert response into JSON.",
        "raw": raw_output,
    }