"""
Source: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/common.py
"""
# Sampling temperature configs for)
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

NEED_REF_CATS = ["math", "reasoning", "coding"]
