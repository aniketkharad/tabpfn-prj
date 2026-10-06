"""Scripted Demo for RelPilot: runs the full end-to-end story non-interactively and writes docs/demo_transcript.md."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.genai._mcp_utils import mcp_to_gemini_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from relpilot.agent import RelPilotStdioAgent

DEMO_QUESTION = (
    "I have an e-commerce database with sellers, orders, items, and reviews. "
    "First, discover the workspace tables and read the skill on acting on predictions. "
    "Next, run the bad_review_risk prediction task. "
    "Explain the backtest trust report in plain words, comparing against baselines. "
    "Finally, propose QA outreach actions for entities meeting at least 75% precision."
)


async def run_scripted_demo() -> None:
    print("==================================================================")
    print("RELPILOT SCRIPTED END-TO-END DEMO")
    print("==================================================================")
    print(f"\nUser Question:\n\"{DEMO_QUESTION}\"\n")

    transcript_lines: list[str] = [
        "# RelPilot Scripted Demo Transcript",
        "",
        f"**Executed**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## User Prompt",
        "",
        f"> {DEMO_QUESTION}",
        "",
        "## Execution Log & Tool Calls",
        "",
    ]

    def on_log(msg: str) -> None:
        if msg.startswith("-> "):
            transcript_lines.append(f"- `{msg}`")
        elif msg.startswith("[Gate]") or msg.startswith("[RelPilot"):
            transcript_lines.append(f"> *{msg}*")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "relpilot.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            gemini_tools = mcp_to_gemini_tools(mcp_tools.tools)

            agent = RelPilotStdioAgent(
                auto_confirm=True,
                on_log=on_log,
            )

            print("Processing agent workflow (discovering, reading skills, running task, gating)...")
            final_response = await agent.run_turn(
                user_message=DEMO_QUESTION,
                session=session,
                gemini_tools=gemini_tools,
            )

            print("\n==================================================================")
            print("RELPILOT FINAL RESPONSE")
            print("==================================================================")
            print(final_response)

            transcript_lines.extend([
                "",
                "## Agent Response & Trust Synthesis",
                "",
                final_response,
                "",
                "---",
                "*Transcript generated automatically by `python -m relpilot.demo`.*",
            ])

    # Save to docs/demo_transcript.md
    out_path = Path("docs/demo_transcript.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    print(f"\nSaved demo transcript to: {out_path.resolve()}")


def main() -> None:
    asyncio.run(run_scripted_demo())


if __name__ == "__main__":
    main()
