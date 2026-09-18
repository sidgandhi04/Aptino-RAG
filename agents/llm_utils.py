import os
import time
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Model constants — pick per-agent in each agent file
MODEL_FAST = "openai/gpt-oss-20b"      # 1000 t/s, cheapest — for Coverage & Validation
MODEL_STRONG = "openai/gpt-oss-120b"   # 500 t/s, strongest — for Decision Agent


def get_llm(model_name=MODEL_FAST, temperature=0, max_tokens=2048):
    """Return a ChatGroq instance for the given model."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(
        temperature=temperature,
        model_name=model_name,
        groq_api_key=api_key,
        max_tokens=max_tokens,
    )


def invoke_with_retry(chain, inputs, task_id="", max_retries=10):
    """Invoke a chain with automatic retry on rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            result = chain.invoke(inputs)
            raw_text = result.content if hasattr(result, 'content') else str(result)
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json_match.group(0)
            return raw_text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Rate limit" in error_str or "rate_limit" in error_str or "TPM" in error_str or "RPM" in error_str:
                match = re.search(r'try again in ([0-9.]+)s', error_str, re.IGNORECASE)
                if match:
                    wait_time = float(match.group(1)) + 2.0
                else:
                    match_min = re.search(r'try again in ([0-9.]+)m([0-9.]+)s', error_str, re.IGNORECASE)
                    if match_min:
                        wait_time = float(match_min.group(1)) * 60.0 + float(match_min.group(2)) + 2.0
                    else:
                        wait_time = (attempt + 1) * 7.0
                msg = f"API Rate Limit hit. Retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})"
                if task_id:
                    try:
                        from api.tasks import log_task_msg
                        log_task_msg(task_id, msg)
                    except ImportError:
                        pass
                print(msg)
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max API retries exceeded.")
