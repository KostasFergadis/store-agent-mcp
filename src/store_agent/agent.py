"""A small agent: the loop that Claude Code normally runs, written out by hand.

It starts the store's MCP server, hands its tools to a local model through
Ollama, and repeats three steps until the model gives a final answer:

    1. ask the model what to do next
    2. run the tool it asked for, through MCP
    3. give it the result

The agent gets the same tools as any other MCP client, so it can search and
propose price changes but never change a price, and every call it makes lands
in the audit log under its own name (AGENT_ID, default "local-agent").

    uv run store-agent-chat "Find what I need for a greek salad and propose 10% off the feta"
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_STEPS = 8

SYSTEM_PROMPT = """\
You are an assistant for a grocery store's catalog staff.
- Use the tools for every product, price or stock fact. Never guess a SKU or a price:
  find the product with search_products first and use the SKU and price it returns.
- For a dish or a list, search for each ingredient or item separately.
- If a tool returns an error, read it, fix the call and try again.
- You cannot change prices. request_price_change only creates a request that a
  person approves or rejects; say so when you use it.
- Keep the final answer short and say which tools you used.
"""
# The SKU, error and per-ingredient rules were added after the first real runs: the
# model guessed sku "feta" and a wrong price without searching, and searched a salad
# only once, missing feta and olives. See the README.

# The loop only needs two things from the outside world, so tests can swap both:
Chat = Callable[[list[dict], list[dict]], Awaitable[dict]]  # messages, tools -> model reply
CallTool = Callable[[str, dict], Awaitable[str]]  # tool name, arguments -> result text
OnEvent = Callable[[str, Any], None]


async def run_agent(
    question: str,
    chat: Chat,
    call_tool: CallTool,
    tools: list[dict],
    max_steps: int = MAX_STEPS,
    on_event: OnEvent = lambda kind, data: None,
) -> str:
    """Run the agent loop and return the model's final answer."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        reply = await chat(messages, tools)
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return reply.get("content", "").strip()  # no tool requested: this is the answer
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            on_event("call", (name, args))
            result = await call_tool(name, args)
            on_event("result", (name, result))
            messages.append({"role": "tool", "tool_name": name, "content": result})
    # A model can keep calling tools forever; the step limit is what stops it.
    return f"Stopped after {max_steps} steps without a final answer."


# --- MCP side -------------------------------------------------------------------


def to_model_tool(tool) -> dict:
    """An MCP tool description, in the function-calling format the model expects."""
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema},
    }


def result_text(result) -> str:
    """Turn an MCP tool result into text for the model. Errors are passed on, not raised,
    so the model can read the message and correct its next call."""
    if result.structured_content is not None:
        data = result.structured_content
        if isinstance(data, dict) and set(data) == {"result"}:  # the SDK wraps non-object returns
            data = data["result"]
        text = json.dumps(data, ensure_ascii=False, default=str)
    else:
        text = "\n".join(getattr(c, "text", "") for c in result.content)
    return f"ERROR: {text}" if result.is_error else text


def mcp_server_params(agent_id: str, extra_env: dict[str, str] | None = None) -> StdioServerParameters:
    # Start the server with the same Python as this process, so it uses this project's code.
    env = {**os.environ, **(extra_env or {}), "AGENT_ID": agent_id}
    return StdioServerParameters(command=sys.executable, args=["-m", "store_agent.mcp_server"], env=env)


async def run_with_store(
    question: str,
    chat: Chat,
    agent_id: str,
    max_steps: int = MAX_STEPS,
    on_event: OnEvent = lambda kind, data: None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Start the MCP server, give its tools to the model, and run the loop."""
    async with (
        stdio_client(mcp_server_params(agent_id, extra_env)) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = [to_model_tool(t) for t in (await session.list_tools()).tools]

        async def call_tool(name: str, args: dict) -> str:
            return result_text(await session.call_tool(name, args))

        return await run_agent(question, chat, call_tool, tools, max_steps, on_event)


# --- model side ------------------------------------------------------------------


def ollama_chat(client: httpx.AsyncClient, model: str) -> Chat:
    async def chat(messages: list[dict], tools: list[dict]) -> dict:
        resp = await client.post(
            "/api/chat",
            # think=False: qwen3 otherwise writes long reasoning before every tool call.
            json={"model": model, "messages": messages, "tools": tools, "stream": False, "think": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]

    return chat


async def chat_with_store(question: str, model: str, ollama_url: str, agent_id: str, max_steps: int) -> str:
    async with httpx.AsyncClient(base_url=ollama_url, timeout=180) as client:
        return await run_with_store(question, ollama_chat(client, model), agent_id, max_steps, print_event)


def print_event(kind: str, data: Any) -> None:
    name, payload = data
    if kind == "call":
        print(f"  → {name} {json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)
    else:
        short = payload if len(payload) <= 160 else payload[:157] + "..."
        print(f"  ← {short}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the store a question through a local agent.")
    parser.add_argument("question")
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--agent-id", default=os.environ.get("AGENT_ID", "local-agent"))
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    answer = asyncio.run(chat_with_store(args.question, args.model, args.ollama_url, args.agent_id, args.max_steps))
    print(answer)


if __name__ == "__main__":
    main()
