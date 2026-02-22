#!/usr/bin/env python3
"""
h4ck_cli.py v0.0.11 — AI-Powered Pentest REPL + Web Dashboard
Supports Ollama, OpenAI, Grok, Anthropic, Google Gemini, OpenRouter.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import select
import shlex
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

import requests
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------------------------------------------------------- #
#                               CONFIG & PATHS                                #
# --------------------------------------------------------------------------- #

__version__ = "0.0.11"
DEFAULT_SERVER = "http://127.0.0.1:5000"
CONFIG_DIR = Path.home() / ".mcpcli"
REPORT_DIR = CONFIG_DIR / "reports"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# --------------------------------------------------------------------------- #
#                                   BANNER                                    #
# --------------------------------------------------------------------------- #

BANNER = r"""
,--.       ,---.      ,--.                  ,---.,--.               ,--.   
|  ,---.  /    | ,---.|  |,-. ,---. ,--.--./  .-'`--',--.--. ,---.,-'  '-. 
|  .-.  |/  '  || .--'|     /| .-. :|  .--'|  `-,,--.|  .--'(  .-''-.  .-' 
|  | |  |'--|  |\ `--.|  \  \\   --.|  |   |  .-'|  ||  |   .-'  `) |  |   
`--' `--'   `--' `---'`--'`--'`----'`--'   `--'  `--'`--'   `----'  `--'   
                                   BY NIKHIL DEEPAK VISHWAKARMA
"""
print(BANNER)
time.sleep(0.1)

# --------------------------------------------------------------------------- #
#                               DEPENDENCIES                                  #
# --------------------------------------------------------------------------- #

try:
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as exc:
    print("pip install langchain langgraph langchain-ollama langchain-openai langchain-anthropic langchain-google-genai google-generativeai")
    raise SystemExit(1) from exc

try:
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# --------------------------------------------------------------------------- #
#                                SAFETY LIST                                  #
# --------------------------------------------------------------------------- #

DANGEROUS_WORDS = {
    "rm", "dd", "mkfs", "fdisk", "reboot", "shutdown", "halt", "poweroff",
    "wipefs", "sfdisk", "parted", "format", "del", "erase", ":"
}

# --------------------------------------------------------------------------- #
#                                 HELPERS                                     #
# --------------------------------------------------------------------------- #


@contextmanager
def safe_op(error_msg: str = "Operation failed"):
    try:
        yield
    except Exception as exc:
        print(f"[ERROR] {error_msg}: {exc}")
        logging.exception(exc)


def load_config() -> Dict[str, str]:
    cfg_file = CONFIG_DIR / "config.ini"
    if not cfg_file.exists():
        return {}
    import configparser
    parser = configparser.ConfigParser()
    with safe_op("Failed to read config"):
        parser.read(cfg_file)
    return dict(parser["default"]) if parser.has_section("default") else {}


def save_config(server: str, model: str, llm_type: str) -> None:
    import configparser
    cfg = configparser.ConfigParser()
    cfg["default"] = {"server": server, "model": model, "llm_type": llm_type}
    with safe_op("Failed to write config"):
        with open(CONFIG_DIR / "config.ini", "w") as f:
            cfg.write(f)


def export_txt_report(history: List[Dict[str, str]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"report_{ts}.txt"
    lines = [f"MCP+LLM Report – {datetime.now():%Y-%m-%d %H:%M:%S}\n", "=" * 60 + "\n"]
    for i, h in enumerate(history, 1):
        lines.append(f"[{i}] INPUT: {h.get('input')}\n")
        lines.append(f"     RESULT: {h.get('summary')}\n")
        lines.append("-" * 40 + "\n")
    with safe_op("Failed to write TXT report"):
        out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_html_report(history: List[Dict[str, str]]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"report_{ts}.html"
    entries = ""
    for h in history:
        entries += f"<div class='entry'><b>Input:</b> {h.get('input', '')}<br><b>Result:</b> {h.get('summary', '—')}</div>"
    html = f"""<!DOCTYPE html>
<html><head><title>MCP Report {ts}</title>
<style>
  body {{font-family:monospace;background:#1e1e1e;color:#d4d4d4;margin:20px}}
  .entry {{border:1px solid #444;margin:1rem;padding:1rem;background:#252526;border-left:4px solid #007acc}}
</style></head>
<body><h1>MCP+LLM Report – {datetime.now():%Y-%m-%d %H:%M:%S}</h1>{entries}</body></html>"""
    with safe_op("Failed to write HTML report"):
        out.write_text(html, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
#                               LLM CONFIG                                    #
# --------------------------------------------------------------------------- #


@dataclass
class LLMConfig:
    model: str = "gemini-2.0-flash"
    llm_type: str = "google"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1

    OPENROUTER_TOOL_MODELS = {
        "anthropic/claude-3-5-sonnet-20241022",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash",
        "x-ai/grok-4",
    }

    GEMINI_MODELS = {"gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro"}

    def validate(self) -> bool:
        if self.llm_type == "ollama":
            url = self.base_url or "http://localhost:11434"
            with safe_op("Ollama unreachable"):
                r = requests.get(f"{url}/api/tags", timeout=5)
                r.raise_for_status()
                models = {m["name"] for m in r.json().get("models", [])}
                if self.model not in models:
                    print(f"Ollama model '{self.model}' not found. Try: {', '.join(sorted(models)[:5])}")
                    return False
                self.base_url = url
                return True

        if self.llm_type in {"openai", "grok", "anthropic", "google"}:
            if not self.api_key:
                print(f"{self.llm_type.title()} requires --api-key or --api-key-file")
                return False

            if self.llm_type == "google" and not self.api_key.startswith("AIza"):
                print("[WARN] Google API key should start with 'AIza...'")

            if self.base_url and "openrouter.ai" in self.base_url:
                if self.model not in self.OPENROUTER_TOOL_MODELS:
                    print(f"OpenRouter model '{self.model}' does NOT support tool calling.")
                    return False

            if self.llm_type == "google" and self.model not in self.GEMINI_MODELS:
                print(f"Invalid Gemini model '{self.model}'. Try: {', '.join(self.GEMINI_MODELS)}")
                return False

            self.base_url = self.base_url or {
                "openai": "https://api.openai.com/v1",
                "grok": "https://api.x.ai/v1",
                "anthropic": "https://api.anthropic.com/v1",
                "google": None,
            }[self.llm_type]
            return True
        return False


# --------------------------------------------------------------------------- #
#                               MCP SERVER                                    #
# --------------------------------------------------------------------------- #


class MCPServer:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.tools_meta: List[Dict[str, Any]] = []
        self.tool_names: List[str] = []
        self._exec_pattern: Optional[str] = None

    def discover_capabilities(self) -> bool:
        for endpoint in ("/mcp/capabilities", "/tools"):
            url = f"{self.base_url}{endpoint}"
            with safe_op("Discovery failed"):
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                tools = self._extract_tools(data)
                if tools:
                    self.tools_meta = tools
                    self.tool_names = [t["name"] for t in tools]
                    print(f"Discovered {len(tools)} tool(s) @ {url}")
                    return True
        return False

    @staticmethod
    def _extract_tools(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict) and "tools" in data:
            return data["tools"]
        if isinstance(data, list):
            return [{"name": t} if isinstance(t, str) else t for t in data]
        return []

    def _discover_exec_pattern(self, sample_tool: str) -> bool:
        candidates = ["/mcp/tools/kali_tools/{tool}", "/tool/{tool}", "/mcp/tool/{tool}"]
        for pat in candidates:
            url = f"{self.base_url}{pat.format(tool=sample_tool)}"
            with safe_op():
                r = requests.post(url, json={"args": ""}, timeout=5)
                if r.status_code in (200, 201, 400):
                    self._exec_pattern = f"{self.base_url}{pat}"
                    return True
        return False

    def call_tool(self, tool: str, args: str = "", timeout: int = 180) -> Dict[str, Any]:
        if not self._exec_pattern:
            if not self.tool_names:
                raise RuntimeError("No tools discovered")
            self._discover_exec_pattern(self.tool_names[0])
        if self._exec_pattern is None:
            raise RuntimeError("Could not determine tool execution endpoint")

        quoted = " ".join(shlex.quote(p) for p in shlex.split(args)) if args else ""
        url = self._exec_pattern.format(tool=tool)
        payload = {"args": quoted}
        with safe_op("Tool execution failed"):
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()


# --------------------------------------------------------------------------- #
#                               LANGCHAIN TOOL                                #
# --------------------------------------------------------------------------- #


@tool
def mcp_tool(tool_input: str) -> str:
    """Execute a Kali tool on the remote MCP server. Input: `<tool_name> [args...]`"""
    global mcp_server
    if not mcp_server:
        return "[ERROR] MCP server not initialized"

    parts = shlex.split(tool_input)
    if not parts:
        return "[ERROR] Empty input"

    tool_name, *arg_parts = parts
    args = " ".join(arg_parts)

    lower = tool_input.lower()
    if any(word in lower for word in DANGEROUS_WORDS):
        return f"[BLOCKED] Dangerous command: {tool_name}"

    if tool_name not in mcp_server.tool_names:
        return f"[ERROR] Unknown tool: {tool_name}"

    result = mcp_server.call_tool(tool_name, args)
    rc = result.get("return_code", -1)
    out = (result.get("stdout") or "").strip()
    err = (result.get("stderr") or "").strip()

    lines = []
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")
    lines.append(f"RC: {rc}")
    return "\n".join(lines) if lines else f"RC: {rc}"


# --------------------------------------------------------------------------- #
#                                 AGENT                                       #
# --------------------------------------------------------------------------- #


class LangChainAgent:
    def __init__(self, cfg: LLMConfig, mcp: MCPServer, memory: MemorySaver):
        self.cfg = cfg
        self.mcp = mcp
        self.memory = memory
        self.llm = self._build_llm()

        # SYSTEM PROMPT: FORCE TOOL USE
        system_prompt = SystemMessage(content=(
            "You are a penetration testing assistant connected to a remote MCP server with 4786 Kali Linux tools. "
            "Your ONLY job is to help with security testing and system commands. "
            "NEVER answer general knowledge questions. "
            "ALWAYS use the `mcp_tool` function to execute commands. "
            "If asked about geography, history, or trivia, respond: "
            "'I am a pentest assistant. I cannot answer general knowledge. Try: mcp_tool nmap -sV 192.168.1.1'"
        ))

        self.app = create_react_agent(
            self.llm,
            tools=[mcp_tool],
            checkpointer=self.memory,
            messages_modifier=system_prompt
        )
        print(f"[DEBUG] Using LLM: {cfg.llm_type} / {cfg.model}")

    def _build_llm(self):
        if self.cfg.llm_type == "ollama":
            return ChatOllama(model=self.cfg.model, temperature=self.cfg.temperature, base_url=self.cfg.base_url)
        if self.cfg.llm_type == "openai":
            return ChatOpenAI(model=self.cfg.model, temperature=self.cfg.temperature, api_key=self.cfg.api_key, base_url=self.cfg.base_url)
        if self.cfg.llm_type == "grok":
            return ChatOpenAI(model=self.cfg.model, temperature=self.cfg.temperature, api_key=self.cfg.api_key, base_url="https://api.x.ai/v1")
        if self.cfg.llm_type == "anthropic":
            return ChatAnthropic(model=self.cfg.model, temperature=self.cfg.temperature, api_key=self.cfg.api_key)
        if self.cfg.llm_type == "google":
            return ChatGoogleGenerativeAI(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                google_api_key=self.cfg.api_key,
                convert_system_message_to_human=False
            )
        raise ValueError(f"Unsupported LLM: {self.cfg.llm_type}")

    def query(self, user_input: str, thread_id: str = "default") -> Dict[str, str]:
        try:
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 5}
            full_response = ""
            print("Thinking… ", end="", flush=True)

            streamed = False
            for chunk in self.app.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                if "messages" in chunk and chunk["messages"]:
                    msg = chunk["messages"][-1]
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content.strip()
                        if content:
                            print(content, end=" ", flush=True)
                            full_response += content + " "
                            streamed = True

                if "agent" in chunk and "tool_calls" in chunk.get("agent", {}):
                    for tc in chunk["agent"]["tool_calls"]:
                        print(f"\n[TOOL CALL] {tc['name']}({tc.get('args', '')})", flush=True)

            if not streamed:
                print("(no stream, invoking…)", end="", flush=True)
                result = self.app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
                final_msg = result["messages"][-1]
                if hasattr(final_msg, 'content') and final_msg.content:
                    full_response = final_msg.content
                    print(f"\n{full_response}")
                else:
                    print("\n[No response]")
                    full_response = ""

            print()
            return {"response": full_response.strip()}

        except Exception as exc:
            print(f"\n[LLM ERROR] {exc}")
            logging.exception(exc)
            return {"error": str(exc), "response": ""}


# --------------------------------------------------------------------------- #
#                               WEB UI                                        #
# --------------------------------------------------------------------------- #


def start_web_ui(history_ref: List[Dict[str, str]], input_queue: Queue, stop_event: threading.Event) -> None:
    if not FLASK_AVAILABLE:
        return

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            cmd = request.form.get("cmd", "").strip()
            if cmd:
                input_queue.put(cmd)
        return render_template_string("""
            <!DOCTYPE html><html><head><title>MCP+LLM Dashboard</title>
            <style>
                body{font-family:monospace;background:#1e1e1e;color:#d4d4d4;margin:20px}
                .log{height:70vh;overflow-y:auto;background:#252526;padding:10px;border:1px solid #444}
                .entry{margin:8px 0;padding:8px;background:#2d2d30;border-left:4px solid #007acc}
                input{width:100%;padding:10px;font:16px monospace;background:#3c3c3c;color:white;border:1px solid #555}
                button{padding:10px 20px;font:16px monospace;background:#007acc;color:white;border:none}
            </style></head>
            <body>
                <h1>MCP+LLM Dashboard (localhost only)</h1>
                <form method=POST><input name=cmd placeholder="Enter command..." autofocus><button>Send</button></form>
                <div class=log id=log></div>
                <script>
                    const log = document.getElementById('log'); let last = 0;
                    setInterval(() => {
                        fetch('/api/history').then(r => r.json()).then(data => {
                            if (data.length > last) {
                                const html = data.slice(last).map(e =>
                                    `<div class=entry><b>Input:</b> ${e.input}<br><small>${e.summary}</small></div>`
                                ).join('');
                                log.insertAdjacentHTML('beforeend', html);
                                log.scrollTop = log.scrollHeight;
                                last = data.length;
                            }
                        });
                    }, 1500);
                </script>
            </body></html>
        """)

    @app.route("/api/history")
    def api_history():
        return jsonify(history_ref)

    def run_server():
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(host="127.0.0.1", port=8080, use_reloader=False, threaded=True, debug=False)

    threading.Thread(target=run_server, daemon=True).start()
    print("Web UI → http://127.0.0.1:8080")


# --------------------------------------------------------------------------- #
#                                   REPL                                      #
# --------------------------------------------------------------------------- #


def repl(server_url: str, llm_cfg: LLMConfig, *, web_ui: bool = False, auto_yes: bool = False) -> None:
    global mcp_server
    mcp_server = MCPServer(server_url)
    if not mcp_server.discover_capabilities():
        print("Could not discover tools. Is the MCP server running?")
        return

    memory = MemorySaver()
    agent = LangChainAgent(llm_cfg, mcp_server, memory)

    input_queue: Queue[str] = Queue()
    history: List[Dict[str, str]] = []
    stop_event = threading.Event()

    if web_ui:
        start_web_ui(history, input_queue, stop_event)

    time.sleep(0.5)

    print(f"MCP+LLM CLI v{__version__} | Model: {llm_cfg.model} ({llm_cfg.llm_type}) | Auto-yes: {auto_yes}")
    print("Natural language → tool. Type 'exit', 'report txt', 'report html', or 'clear'.\n")

    thread_id = f"cli_{random.getrandbits(32):08x}"

    while not stop_event.is_set():
        try:
            print("mcp+llm> ", end="", flush=True)

            while True:
                if not input_queue.empty():
                    raw = input_queue.get_nowait()
                    print(f"\n[Web] {raw}")
                    break
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    raw = sys.stdin.readline().rstrip("\n")
                    if raw:
                        print(raw)
                        break
                time.sleep(0.05)

        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        raw = raw.strip()
        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            break
        if raw == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            print(BANNER)
            continue

        if raw.startswith("report"):
            parts = raw.split(maxsplit=1)
            fmt = parts[1] if len(parts) > 1 else "html"
            path = export_txt_report(history) if fmt == "txt" else export_html_report(history)
            print(f"{'TXT' if fmt == 'txt' else 'HTML'} report → {path}")
            continue

        start_time = time.time()
        resp = agent.query(raw, thread_id=thread_id)
        query_time = time.time() - start_time
        print(f"\n[DEBUG] Query took {query_time:.1f}s")

        summary = "error" if "error" in resp else "ok"
        history.append({"input": raw, "summary": summary})


# --------------------------------------------------------------------------- #
#                                   MAIN                                      #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Pentest CLI + Web Dashboard")
    parser.add_argument("--server", "-s", help="MCP server URL")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash", help="LLM model")
    parser.add_argument("--llm-type", choices=["ollama", "openai", "grok", "anthropic", "google"], default="google")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-key-file", help="File with API key")
    parser.add_argument("--base-url", help="Custom LLM base URL")
    parser.add_argument("--web-ui", action="store_true")
    parser.add_argument("--auto-yes", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    api_key = args.api_key
    if args.api_key_file:
        try:
            api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
            if not api_key:
                print("[ERROR] API key file is empty")
                sys.exit(1)
        except Exception as exc:
            print(f"[ERROR] Cannot read API key file: {exc}")
            sys.exit(1)

    cfg = load_config()
    server = args.server or cfg.get("server") or DEFAULT_SERVER
    model = args.model or cfg.get("model") or "gemini-2.0-flash"
    llm_type = args.llm_type or cfg.get("llm_type") or "google"

    llm_cfg = LLMConfig(model=model, llm_type=llm_type, api_key=api_key, base_url=args.base_url)
    if not llm_cfg.validate():
        sys.exit(1)

    if args.server or args.model or args.llm_type:
        save_config(server, model, llm_type)

    repl(server, llm_cfg, web_ui=args.web_ui, auto_yes=args.auto_yes)


if __name__ == "__main__":
    main()
