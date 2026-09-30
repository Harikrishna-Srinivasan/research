"""Tool Registry — unified dispatch + dynamic tool creation."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from deepagent.config import DeepAgentConfig
from deepagent.utils.cleanup import get_tracker
from deepagent.utils.logger import get_logger

log = get_logger(__name__)


class ToolRegistry:
    """Central registry of all available tools (built-in + dynamic + MCP)."""

    def __init__(self, config: DeepAgentConfig) -> None:
        self.config = config
        self._tools: dict[str, dict] = {}  # name -> {description, parameters, function}

        # Register built-in tools
        self._register_builtins()
        # Register meta-tools (task_complete, task_failed, create_tool)
        self._register_meta_tools()
        # Load custom tools from disk
        self._load_custom_tools()

    # ----------------------------------------------------------------
    # Registration
    # ----------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        function: Callable,
        source: str = "builtin",
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function,
            "source": source,
        }

    def _register_builtins(self) -> None:
        from deepagent.tools.file_ops import FILE_TOOLS
        from deepagent.tools.shell import SHELL_TOOLS
        from deepagent.tools.web import WEB_TOOLS, SEARCH_PROVIDER_TOOLS
        from deepagent.tools.code_exec import CODE_EXEC_TOOLS
        from deepagent.tools.research import RESEARCH_TOOLS

        for tool_list in [FILE_TOOLS, SHELL_TOOLS, WEB_TOOLS, CODE_EXEC_TOOLS, RESEARCH_TOOLS]:
            for t in tool_list:
                self.register(
                    name=t["name"],
                    description=t["description"],
                    parameters=t.get("parameters", {}),
                    function=t["function"],
                )

        # Register only the configured web search provider
        provider = getattr(self.config.tools, "web_search_provider", "duckduckgo")
        search_tool = SEARCH_PROVIDER_TOOLS.get(provider)
        if search_tool:
            self.register(
                name=search_tool["name"],
                description=search_tool["description"],
                parameters=search_tool.get("parameters", {}),
                function=search_tool["function"],
            )
        else:
            log.warning("Unknown web_search_provider '%s'; no search tool registered.", provider)

    def _register_meta_tools(self) -> None:
        """Register special control-flow tools."""

        def task_complete(result: str) -> str:
            return f"TASK_COMPLETE: {result}"

        def task_failed(reason: str) -> str:
            return f"TASK_FAILED: {reason}"

        def create_tool(name: str, code: str) -> str:
            """Dynamically create a new tool from Python code."""
            return self.create_tool_from_code(name, code)

        self.register(
            "task_complete",
            "Signal that the current sub-task is complete.",
            {"result": {"type": "string", "description": "Summary of what was accomplished"}},
            task_complete,
            source="meta",
        )
        self.register(
            "task_failed",
            "Signal that the current sub-task has failed.",
            {"reason": {"type": "string", "description": "Why the task failed"}},
            task_failed,
            source="meta",
        )

        if self.config.agent.enable_tool_creation:
            self.register(
                "create_tool",
                "Create a new reusable tool from Python code. The code must define a function "
                "with the same name as the tool. Include TOOL_META comments for metadata.",
                {
                    "name": {"type": "string", "description": "Tool name (snake_case)"},
                    "code": {"type": "string", "description": "Python source code defining the tool function"},
                },
                create_tool,
                source="meta",
            )

    # ----------------------------------------------------------------
    # Dynamic tool creation
    # ----------------------------------------------------------------

    def create_tool_from_code(self, name: str, code: str) -> str:
        """Parse, validate, save, and register a dynamically created tool."""
        # Validate name
        if not re.match(r"^[a-z_][a-z0-9_]*$", name):
            return f"ERROR: Tool name must be snake_case. Got: '{name}'"

        if name in self._tools:
            return f"ERROR: Tool '{name}' already exists. Choose a different name."

        # Parse TOOL_META comments
        description = "Custom tool"
        parameters = {}
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("# description:"):
                description = line.split(":", 1)[1].strip()
            elif line.startswith("# parameters:"):
                try:
                    parameters = json.loads(line.split(":", 1)[1].strip())
                except json.JSONDecodeError:
                    pass

        # Save to disk
        tools_dir = Path(self.config.tools.custom_tools_dir).resolve()
        tools_dir.mkdir(parents=True, exist_ok=True)
        tool_path = tools_dir / f"{name}.py"
        tool_path.write_text(code, encoding="utf-8")
        get_tracker().track_file(str(tool_path))

        # Load and register
        try:
            fn = self._load_function_from_file(str(tool_path), name)
            self.register(name, description, parameters, fn, source="dynamic")
            log.info("Created new tool: %s", name)
            return f"SUCCESS: Tool '{name}' created and registered. It is now available for use."
        except Exception as e:
            return f"ERROR creating tool '{name}': {e}"

    def _load_function_from_file(self, path: str, func_name: str) -> Callable:
        """Dynamically load a function from a Python file."""
        spec = importlib.util.spec_from_file_location(f"custom_tool_{func_name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, func_name, None)
        if fn is None:
            raise AttributeError(f"Function '{func_name}' not found in {path}")
        return fn

    def _load_custom_tools(self) -> None:
        """Load previously created custom tools from disk."""
        tools_dir = Path(self.config.tools.custom_tools_dir).resolve()
        if not tools_dir.exists():
            return

        for py_file in tools_dir.glob("*.py"):
            name = py_file.stem
            if name.startswith("_"):
                continue
            try:
                # Read metadata from file
                content = py_file.read_text(encoding="utf-8")
                description = "Custom tool"
                parameters = {}
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("# description:"):
                        description = line.split(":", 1)[1].strip()
                    elif line.startswith("# parameters:"):
                        try:
                            parameters = json.loads(line.split(":", 1)[1].strip())
                        except json.JSONDecodeError:
                            pass

                fn = self._load_function_from_file(str(py_file), name)
                self.register(name, description, parameters, fn, source="dynamic")
                log.info("Loaded custom tool: %s", name)
            except Exception as e:
                log.warning("Failed to load custom tool %s: %s", name, e)

    # ----------------------------------------------------------------
    # Invocation
    # ----------------------------------------------------------------

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool by name."""
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            return f"ERROR: Unknown tool '{name}'. Available tools: {available}"

        tool = self._tools[name]
        fn = tool["function"]

        # Inject config-based safety params where applicable
        if name == "run_command":
            arguments.setdefault("allowed_commands", self.config.tools.allowed_commands)
            arguments.setdefault("blocked_patterns", self.config.tools.blocked_patterns)
        elif name in ("write_file", "append_file"):
            arguments.setdefault("workspace_dirs", self.config.tools.workspace_dirs)

        try:
            result = fn(**arguments)
            return str(result)
        except Exception as e:
            return f"ERROR in tool '{name}': {type(e).__name__}: {e}"

    # ----------------------------------------------------------------
    # Schema export (for LLM prompts)
    # ----------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas formatted for the LLM."""
        schemas = []
        for name, t in self._tools.items():
            schemas.append({
                "name": name,
                "description": t["description"],
                "parameters": t["parameters"],
            })
        return schemas

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
