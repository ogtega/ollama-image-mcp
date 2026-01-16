import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal, NamedTuple, Optional

import httpx
from mcp import ServerSession
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import BaseModel, SkipValidation, ValidationError

# --- Strict Data Models ---


class ProgressEvent(BaseModel):
    step: int
    total: int


class ImageData(BaseModel):
    b64_json: SkipValidation[str]


class DoneEvent(BaseModel):
    created: int
    data: list[ImageData]


# --- Server Setup ---


class AppContext(NamedTuple):
    """Application context with typed dependencies."""

    http_client: httpx.AsyncClient


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[AppContext]:
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=60.0, read=None, write=None, pool=None)
    )

    try:
        yield AppContext(http_client)
    finally:
        await http_client.aclose()


mcp = FastMCP("Ollama Image Generator", lifespan=lifespan)


async def process_sse_event(
    ctx: Context[ServerSession, AppContext], event_type: Optional[str], data_text: str
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
                # Strip newlines/CR to ensure a clean base64 string
                return next(iter(event.data)).b64_json
        except ValidationError as e:
            logging.error(f"Validation failed for DoneEvent: {e}")
            return None

    # If we reach here, it's an unknown event type.
    # In strict mode, we ignore it rather than guessing.
    return None


@mcp.tool()
async def generate_image(
    ctx: Context[ServerSession, AppContext],
    prompt: str,
    size: str = "1024x1024",
    model: Literal["x/z-image-turbo"] = "x/z-image-turbo",
) -> list[ImageContent | TextContent] | str:
    """Generates an image using the local Ollama API with strict Pydantic validation."""
    url = "http://localhost:11434/v1/images/generations"

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
        "stream": True,
    }

    (http_client,) = ctx.request_context.lifespan_context

    final_image: Optional[str] = None

    # SSE State
    current_event_type: Optional[str] = None
    current_data_lines: list[str] = []

    try:
        async with http_client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for raw_line in response.aiter_lines():
                line = raw_line.rstrip("\r\n")

                if line == "":  # End of an SSE event
                    if current_data_lines:
                        # Process buffered event
                        img = await process_sse_event(
                            ctx, current_event_type, "\n".join(current_data_lines)
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
                        img = await process_sse_event(ctx, None, line)
                        if img:
                            final_image = img
                            break
                    except json.JSONDecodeError:
                        pass

    except Exception as e:
        return f"Error: {str(e)}"

    if not final_image:
        return "Stream ended without returning valid image data."

    return [
        TextContent(type="text", text=f"Generated: {prompt}"),
        ImageContent(type="image", data=final_image, mimeType="image/png"),
    ]


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
