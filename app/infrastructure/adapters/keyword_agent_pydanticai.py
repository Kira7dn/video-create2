from __future__ import annotations

import logging
from typing import List, Optional

from pydantic_ai import Agent  # type: ignore
from pydantic_ai.models.openai import OpenAIModel  # type: ignore
from pydantic import BaseModel  # type: ignore

from app.application.interfaces import IKeywordAgent
from app.core.config import settings

logger = logging.getLogger(__name__)


class PydanticAIKeywordAgent(IKeywordAgent):
    """Keyword agent implemented via pydantic-ai + OpenAI model.

    This adapter encapsulates any dependency on LLM SDKs from the application layer.
    If initialization fails (e.g., missing dependencies or API key), it will
    gracefully fall back in `extract_keywords` to simply returning provided fields
    truncated to the configured max.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._agent = None
        self._init_exception: Optional[Exception] = None
        self._model_name = model_name or settings.ai_pydantic_model
        self._initialize()

    def _initialize(self) -> None:
        if not settings.openai_api_key or not settings.ai_keyword_extraction_enabled:
            logger.debug(
                "PydanticAIKeywordAgent: AI disabled or missing API key; running in fallback mode"
            )
            return
        try:

            class KeywordExtractionResult(BaseModel):
                keywords: List[str] = []

            model = OpenAIModel(model_name=self._model_name)
            self._agent = Agent(
                model=model,
                output_type=KeywordExtractionResult,
                system_prompt=(
                    "You are a smart keyword-extraction assistant for image search.\n"
                    "Extract 4-8 multi-word phrases optimized for Pixabay."
                ),
            )
            logger.info("🤖 PydanticAIKeywordAgent initialized")
        except Exception as e:  # noqa: BLE001
            self._init_exception = e
            self._agent = None
            logger.warning("PydanticAIKeywordAgent init failed: %s", e)

    async def extract_keywords(
        self,
        content: str,
        *,
        fields: Optional[List[str]] = None,
        max_keywords: int = 8,
    ) -> List[str]:
        fields = fields or []

        # If disabled/unavailable, return fields (truncated)
        if not settings.ai_keyword_extraction_enabled or self._agent is None:
            return fields[:max_keywords]

        try:
            user_prompt = (
                f"Extract image search keywords from this content: '{content}' "
                f"with these related fields: [{', '.join(fields)}]"
            )
            result = await self._agent.run(user_prompt=user_prompt)  # type: ignore[attr-defined]
            return list(result.output.keywords)[:max_keywords]
        except Exception as e:  # noqa: BLE001
            logger.warning("PydanticAIKeywordAgent.extract_keywords failed: %s", e)
            return fields[:max_keywords]
