"""LLM factory — the one place that builds a chat model, so provider choice is a config
value (provider id + model name) rather than a hardcoded class. Uses LangChain's own
init_chat_model; credentials come from env vars (e.g. DEEPSEEK_API_KEY) loaded by the caller."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def build_llm(model: str, provider: str = "", **kwargs) -> BaseChatModel:
    # provider empty -> LangChain infers it from the model name. kwargs (temperature,
    # callbacks, ...) pass straight through to the concrete model.
    if (provider or "").lower() == "deepseek":
        # DeepSeek defaults to thinking mode, but langchain-deepseek drops its
        # reasoning_content on tool-call/multi-turn replay -> HTTP 400 (langchain #37174).
        # Disable thinking so multi-step agent runs finish cleanly.
        kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
    return init_chat_model(model, model_provider=provider or None, **kwargs)
