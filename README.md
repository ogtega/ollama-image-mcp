# Ollama Image MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that integrates with a local [Ollama](https://ollama.com/) instance to provide image generation capabilities.

This server exposes a tool that allows connected LLMs to generate images from text prompts, leveraging image models running locally via Ollama.

## Features

- **Image Generation**: Exposes a `generate_image` tool to create images from text descriptions.
- **Streaming Feedback**: Provides real-time progress logging back to the client during generation.
- **Strict Validation**: Utilizes Pydantic for strict parsing and validation of Ollama's Server-Sent Events (SSE) responses.
- **FastMCP**: Built using the high-performance `FastMCP` framework.

## Prerequisites

- **Python**: Version 3.10 or higher.
- **uv**: Project dependency manager (recommended).
- **Ollama**: A local instance running at `http://localhost:11434`.
- **Image Model**: An image generation model available in Ollama. The default is `x/z-image-turbo`.
  - Ensure you have pulled the model: `ollama pull x/z-image-turbo`

## Installation

This project uses `uv` for dependency management.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ollama-image-mcp
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

## Configuration

To use this server with an MCP client (like Claude Desktop), configure it to run the server script.

### Claude Desktop Configuration

Edit your `claude_desktop_config.json` (typically located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS) to include:

```json
{
  "mcpServers": {
    "ollama-image": {
      "command": "uvx",
      "args": [
        "git+https://github.com/ogtega/ollama-image-mcp"
      ]
    }
  }
}
```

*Note: This will automatically download and run the server using `uvx`.*

## Tools Available

### `generate_image`

Generates an image using the specified prompt.

- **prompt** (string, required): The text description of the image you want to generate.
- **size** (string, optional): Image size (default: "1024x1024").
- **model** (string, optional): The Ollama model tag to use (default: "x/z-image-turbo").

## Development

The project includes a `Makefile` for common development tasks.

```bash
# Format code (Black + isort)
make format

# Lint code (check only)
make lint
```

## License

MIT
