#!/usr/bin/env python3

import os
import sys
import json
import time
import queue
import shlex
import logging
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, Response

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("kali-mcp-server")

# ---------------- CONFIG ----------------
HOST = "192.168.56.101"
PORT = 50001
COMMAND_TIMEOUT = 240

ALLOWED_PREFIXES = [
    "/usr/bin", "/usr/sbin", "/usr/local/bin",
    "/bin", "/sbin", "/opt"
]

# ---------------- MCP STATE ---------------
clients = {}

# ---------------- HELPERS ----------------
def resolve_executable(cmd: str):
    try:
        parts = shlex.split(cmd)
        exe = parts[0]
    except Exception:
        return None

    for prefix in ALLOWED_PREFIXES:
        path = Path(prefix) / exe
        if path.is_file() and os.access(path, os.X_OK):
            return parts

    return None

def execute_command(cmd: str):
    args = resolve_executable(cmd)
    if not args:
        return {"error": "command not allowed"}, 403

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            env=os.environ.copy()
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }, 200
    except subprocess.TimeoutExpired:
        return {"error": "command timed out"}, 504
    except Exception as e:
        logger.exception("Execution failed")
        return {"error": str(e)}, 500

def mcp_send(client_id, payload):
    clients[client_id].put(payload)

# ---------------- BASIC ROUTES ----------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "server": "kali-mcp"})

@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(silent=True) or {}
    cmd = data.get("command", "")
    if not cmd:
        return jsonify({"error": "missing command"}), 400
    result, code = execute_command(cmd)
    return jsonify(result), code

# ---------------- MCP CAPABILITIES ----------------
@app.route("/mcp/capabilities")
def capabilities():
    return jsonify({
        "name": "Kali MCP Tool Server",
        "description": "Executes commands on Kali (dangerous)",
        "tools": [
            {
                "name": "run_command",
                "description": "Run a shell command",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string"
                        }
                    },
                    "required": ["command"]
                }
            }
        ]
    })

# ---------------- MCP SSE STREAM ----------------
@app.route("/mcp")
def mcp_stream():
    client_id = str(time.time())
    clients[client_id] = queue.Queue()

    def event_stream():
        # MCP initialize
        yield f"data: {json.dumps({
            'type': 'initialize',
            'serverInfo': {
                'name': 'Kali MCP Server',
                'version': '1.0'
            }
        })}\n\n"

        while True:
            msg = clients[client_id].get()
            yield f"data: {json.dumps(msg)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")

# ---------------- MCP MESSAGE HANDLER ----------------
@app.route("/mcp/message", methods=["POST"])
def mcp_message():
    msg = request.get_json(force=True)
    msg_type = msg.get("type")
    client_id = msg.get("clientId")

    if client_id not in clients:
        return jsonify({"error": "invalid client"}), 400

    # LIST TOOLS
    if msg_type == "list_tools":
        mcp_send(client_id, {
            "type": "list_tools_result",
            "tools": [
                {
                    "name": "run_command",
                    "description": "Execute a command",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }
            ]
        })

    # CALL TOOL
    elif msg_type == "call_tool":
        if msg.get("name") != "run_command":
            mcp_send(client_id, {
                "type": "call_tool_result",
                "error": "unknown tool"
            })
            return "", 204

        args = msg.get("arguments", {})
        cmd = args.get("command", "")
        result, _ = execute_command(cmd)

        mcp_send(client_id, {
            "type": "call_tool_result",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        })

    return "", 204

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info(f"Starting MCP server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)
