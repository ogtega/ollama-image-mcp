import json
import logging
from typing import Any, List, Optional

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, ValidationError

# --- Strict Data Models ---


class ProgressEvent(BaseModel):
    step: int
    total: int


class ImageData(BaseModel):
    b64_json: str


class DoneEvent(BaseModel):
    created: int
    data: List[ImageData]


# --- Server Setup ---

mcp = FastMCP("Ollama Image Generator")


@mcp.tool()
async def generate_image(
    ctx: Context, prompt: str, size: str = "1024x1024", model: str = "x/z-image-turbo"
) -> dict[str, Any] | str:
    """
    Generates an image using the local Ollama API with strict Pydantic validation.
    """
    url = "http://localhost:11434/v1/images/generations"

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
        "stream": True,
    }

    final_image: Optional[str] = None

    # Connect timeout only; infinite read timeout for generation
    timeout = httpx.Timeout(connect=60.0, read=None)

    # SSE State
    current_event_type: Optional[str] = None
    current_data_lines: list[str] = []

    async def process_sse_event(
        event_type: Optional[str], data_text: str
    ) -> Optional[str]:
        """
        Parses JSON strictly against Pydantic models.
        """
        if data_text.strip() == "[DONE]":
            return None

        try:
            raw = json.loads(data_text)
        except json.JSONDecodeError:
            logging.warning(f"Skipping invalid JSON: {data_text[:50]}...")
            return None

        # 1. Check for Progress
        # We check structure presence because 'event' field is optional in some SSE implementations
        if event_type == "progress" or ("step" in raw and "total" in raw):
            try:
                event = ProgressEvent.model_validate(raw)
                await ctx.info(f"Generating... {event.step}/{event.total}")
                return None
            except ValidationError as e:
                logging.error(f"Validation failed for ProgressEvent: {e}")
                return None

        # 2. Check for Done/Success
        if event_type == "done" or "data" in raw:
            try:
                event = DoneEvent.model_validate(raw)
                if event.data:
                    return event.data[0].b64_json
            except ValidationError as e:
                logging.error(f"Validation failed for DoneEvent: {e}")
                return None

        # If we reach here, it's an unknown event type.
        # In strict mode, we ignore it rather than guessing.
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    if raw_line is None:
                        continue
                    line = raw_line.rstrip("\r\n")

                    if line == "":
                        if current_data_lines:
                            # Process buffered event
                            img = await process_sse_event(
                                current_event_type, "\n".join(current_data_lines)
                            )
                            if img:
                                final_image = img
                                break
                            current_event_type = None
                            current_data_lines = []
                        continue

                    if line.startswith("event:"):
                        current_event_type = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        current_data_lines.append(line.split(":", 1)[1].lstrip())
                    else:
                        # Handle non-SSE JSON lines if server sends them (Strict JSON check)
                        try:
                            json.loads(line)
                            img = await process_sse_event(None, line)
                            if img:
                                final_image = img
                                break
                        except json.JSONDecodeError:
                            pass

    except Exception as e:
        return f"Error: {str(e)}"

    if not final_image:
        return "Stream ended without returning valid image data."

    return {
        "content": [
            {"type": "text", "text": f"Generated: {prompt}"},
            {"type": "image", "data": final_image, "mimeType": "image/png"},
        ]
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
