from __future__ import annotations

from typing import List
from pydantic import ValidationError
from app.core.pyd_schemas import InputPayload

from app.application.pipeline.base import PipelineContext, BaseStep


class ValidateInputStep(BaseStep):
    """Matches service stage: request_validation

    Input:  json_data
    Output: validated_data
    """

    name = "validate_input"

    async def run(self, context: PipelineContext) -> None:  # type: ignore[override]
        # Extract input
        data = context.input or {}
        if not isinstance(data, dict):
            raise ValueError("context.data must be a dict")

        # Normalized envelope used across the app
        json_data = {
            "json_data": data.get("json_data", data),
            "transitions": data.get("transitions", []),
            "background_music": data.get("background_music"),
        }
        current = json_data["json_data"]

        # ---- Validate and normalize ----
        try:
            payload = InputPayload.model_validate(current)
        except ValidationError as e:
            # Aggregate human-friendly messages
            lines: List[str] = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "invalid input")
                lines.append(f"{loc}: {msg}")
            raise ValueError("\n".join(lines)) from e

        context.set("validated_data", payload.model_dump())
        context.ensure_run_id()
