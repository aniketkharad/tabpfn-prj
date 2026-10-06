"""RelPilot Gemini CLI Agent: connects to RelPilot MCP server over stdio."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import errors, types
from google.genai._mcp_utils import mcp_to_gemini_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MAX_GEMINI_CALLS_PER_SESSION = 30

SYSTEM_PROMPT = """You are RelPilot, an autonomous agent that operates over relational databases.
Mission: "Ask your database about the future. The agent acts only where the model has proven it deserves trust."

Core Protocol:
1. Discover the database: Call `describe_workspace` to inspect tables, schema, FKs, and available tasks/skills.
2. Skill consultation: Read relevant skills with `read_skill` (e.g. 'task-design', 'acting-on-predictions') before running tasks or acting.
3. Temporal prediction & backtest: Call `run_task` to fit TabPFN-Rel on historical data and backtest on the latest labelled period.
4. Explain the trust report: Explain the metrics in plain words (TabPFN-Rel AUROC vs constant baselines, calibration ECE, precision@k).
5. Never hallucinate numbers: Never state a number that is not explicitly in a tool output.
6. Gated action proposal: Call `propose_actions` only after reading the trust report.
7. Safe fallback: If the gate yields no threshold (due to low precision or sample support), state clearly that no action is justified and explain why entities require human review.
"""


def load_env_keys() -> tuple[str | None, str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not api_key:
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("GEMINI_MODEL=") and not os.environ.get("GEMINI_MODEL"):
                val = line.split("=", 1)[1].strip()
                if val:
                    model_name = val
    return api_key, model_name


class RelPilotStdioAgent:
    """Gemini agent that interacts with the RelPilot MCP server over stdio."""

    def __init__(
        self,
        model: str | None = None,
        auto_confirm: bool = False,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        api_key, default_model = load_env_keys()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to .env.")

        self.client = genai.Client(api_key=api_key)
        self.model = model or default_model
        self.fallback_model = "gemini-3.5-flash"
        self.auto_confirm = auto_confirm
        self.on_log = on_log or (lambda msg: None)
        self.call_count = 0
        self.contents: list[types.Content] = []

    def _log(self, msg: str) -> None:
        print(msg)
        self.on_log(msg)

    async def run_turn(
        self,
        user_message: str,
        session: ClientSession,
        gemini_tools: list[types.Tool],
    ) -> str:
        """Process one conversational turn, handling tool calls via the MCP session."""
        self.contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        while True:
            if self.call_count >= MAX_GEMINI_CALLS_PER_SESSION:
                msg = (
                    f"\n[Quota Alert] Reached session limit of {MAX_GEMINI_CALLS_PER_SESSION} Gemini calls. Halting."
                )
                self._log(msg)
                return msg

            # Generate content with retry once on 429/503
            response = None
            models_to_try = [self.model]
            if self.model != self.fallback_model:
                models_to_try.append(self.fallback_model)

            for target_model in models_to_try:
                for attempt in range(2):
                    try:
                        self.call_count += 1
                        response = self.client.models.generate_content(
                            model=target_model,
                            contents=self.contents,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                tools=gemini_tools,
                            ),
                        )
                        break
                    except errors.APIError as e:
                        if attempt == 0 and ("503" in str(e) or "429" in str(e)):
                            time.sleep(2.0)
                            continue
                        break
                    except Exception:
                        break
                if response is not None:
                    break

            if response is None:
                err_msg = "[RelPilot Error] Gemini API request failed after retry."
                self._log(err_msg)
                return err_msg

            # Append model candidate response
            model_content = response.candidates[0].content
            self.contents.append(model_content)

            # Check if model wants to call functions
            function_calls = [part.function_call for part in model_content.parts if part.function_call]
            if not function_calls:
                # Text response produced
                final_text = "".join(part.text for part in model_content.parts if part.text)
                return final_text.strip()

            # Execute function calls over MCP session
            response_parts: list[types.Part] = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                args_str = ", ".join(f"{k}={v!r}" for k, v in fn_args.items())
                self._log(f"-> {fn_name}({args_str})")

                # Human-in-the-loop gate for propose_actions
                if fn_name == "propose_actions":
                    if not self.auto_confirm:
                        print(f"\n[HUMAN IN THE LOOP] Pending action proposal: {fn_args}")
                        user_choice = input("Approve action proposal? (y/n) > ").strip().lower()
                        if user_choice not in ("y", "yes"):
                            self._log("[Gate] Action proposal rejected by user.")
                            declined_data = {
                                "status": "rejected_by_user",
                                "message": "User declined to execute propose_actions.",
                            }
                            response_parts.append(
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=fn_name,
                                        response={"result": json.dumps(declined_data)},
                                    )
                                )
                            )
                            continue

                # Call tool via stdio MCP session
                try:
                    mcp_res = await session.call_tool(fn_name, fn_args)
                    tool_output_text = mcp_res.content[0].text if mcp_res.content else "{}"
                except Exception as e:
                    tool_output_text = json.dumps({"error": True, "message": str(e)})

                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn_name,
                            response={"result": tool_output_text},
                        )
                    )
                )

            # Append tool execution results back to conversation
            self.contents.append(types.Content(role="user", parts=response_parts))


async def run_repl(initial_query: str | None = None, auto_confirm: bool = False) -> None:
    """Start the interactive Gemini CLI REPL connected to the MCP server."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "relpilot.server"],
    )

    print("Connecting to RelPilot MCP Server over stdio...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            gemini_tools = mcp_to_gemini_tools(mcp_tools.tools)

            agent = RelPilotStdioAgent(auto_confirm=auto_confirm)
            print("Connected to RelPilot MCP server. Ready.\n")

            if initial_query:
                print(f"User > {initial_query}\n")
                ans = await agent.run_turn(initial_query, session, gemini_tools)
                print(f"\nRelPilot >\n{ans}\n")
                return

            print("Type your instructions below (type 'exit' or 'quit' to stop).\n")
            while True:
                try:
                    user_input = input("User > ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("exit", "quit", "q"):
                        print("Goodbye!")
                        break

                    answer = await agent.run_turn(user_input, session, gemini_tools)
                    print(f"\nRelPilot >\n{answer}\n")
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting RelPilot.")
                    break


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(run_repl(initial_query=query, auto_confirm=False))


if __name__ == "__main__":
    main()
