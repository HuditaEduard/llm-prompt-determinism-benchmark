from typing import Callable

# Each tier returns (system_prompt, user_prompt)
PromptBuilder = Callable[[str], tuple[str, str]]


def tier_0_zero_shot(task: str) -> tuple[str, str]:
    """Baseline: Unstructured natural language query with no schema or persona."""
    system = "You are a helpful assistant."
    user = f"{task}\nProvide the final numerical or binary answer."
    return system, user


def tier_1_role_and_schema(task: str) -> tuple[str, str]:
    """Adds domain role anchoring and strict JSON schema constraints."""
    system = (
        "You are an electrical engineering and computer science computation engine. "
        "You must respond ONLY with a valid JSON object matching this exact schema:\n"
        '{"reasoning": "<concise explanation>", "final_answer": "<number or binary digit only>"}\n'
        "Do not include markdown code fences, units, or any text outside the JSON object."
    )
    user = f"Solve the following problem:\n{task}"
    return system, user


def tier_2_zero_shot_cot(task: str) -> tuple[str, str]:
    """Enforces explicit step-by-step Chain-of-Thought decomposition inside JSON."""
    system = (
        "You are an electrical engineering and computer science computation engine. "
        "Break the problem down into explicit mathematical or logical steps inside the 'reasoning' field "
        "before computing 'final_answer'. Respond ONLY with valid JSON matching:\n"
        '{"reasoning": "<step-by-step derivation>", "final_answer": "<number or binary digit only>"}\n'
        "Do not output markdown fences or units in 'final_answer'."
    )
    user = f"Solve step by step:\n{task}"
    return system, user


def tier_3_few_shot_cot(task: str) -> tuple[str, str]:
    """Combines domain role, strict JSON schema, Chain-of-Thought, and cross-domain exemplars."""
    system = (
        "You are an electrical engineering and computer science computation engine. "
        "Follow the exact step-by-step derivation style and strict JSON schema shown in the examples below. "
        "Output ONLY raw JSON with no markdown fences and no units in 'final_answer'."
    )
    user = (
        "Example 1:\n"
        "Problem: A 2-stage logic circuit takes A=1, B=1, C=0. A and B feed into a XOR gate to produce X. "
        "X and C feed into a NOR gate to produce Q. Determine Q.\n"
        'Response: {"reasoning": "Step 1: X = A XOR B = 1 XOR 1 = 0. Step 2: Q = X NOR C = 0 NOR 0 = 1.", "final_answer": "1"}\n\n'
        "Example 2:\n"
        "Problem: Calculate the -3 dB cutoff frequency (in Hz, rounded to 2 decimal places) of an RC filter with R = 1000 Ohms and C = 100 nF.\n"
        'Response: {"reasoning": "Step 1: f_c = 1 / (2 * pi * R * C). Step 2: R * C = 1000 * 100e-9 = 1e-4 s. Step 3: f_c = 1 / (6.283185 * 1e-4) = 1591.55 Hz.", "final_answer": "1591.55"}\n\n'
        f"Target Problem:\n{task}"
    )
    return system, user


PROMPT_TIERS: dict[str, PromptBuilder] = {
    "T0_zero_shot": tier_0_zero_shot,
    "T1_schema_only": tier_1_role_and_schema,
    "T2_zero_shot_cot": tier_2_zero_shot_cot,
    "T3_few_shot_cot": tier_3_few_shot_cot,
}