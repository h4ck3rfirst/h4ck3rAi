#!/usr/bin/env python3
"""
Improved MCP-Kali-Server (2026 edition)
- Generic command execution for any Kali tool
- Blocks dangerous commands
- Simple /health endpoint
- Optional non-root execution
- Better timeout & logging
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import traceback
from typing import Dict, Any
from flask import Flask, request, jsonify

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
DEFAULT_PORT = 5000
COMMAND_TIMEOUT = 420          # 7 minutes — good for long scans
DEBUG = False

# Dangerous command blacklist (regex, case-insensitive)
BLACKLIST_PATTERNS = [
    r'rm\s+-[rf]+',               # rm -rf, rm -f -r, etc.
    r'dd\s+if=/dev/zero|if=/dev/urandom',  # disk wiping
    r'mkfs|format|fdisk|parted|gdisk',     # formatting / partitioning
    r'shutdown|reboot|halt|poweroff|init\s+[06]',
    r':\(\)\s*{\s*:\|\:|\s*&\s*};\s*:\s*',  # fork bomb
    r'su\s+-c|sudo\s+.*rm|sudo\s+.*dd|sudo\s+.*mkfs',  # sneaky sudo destructive
    r'>\s*/dev/|>>/dev/',                 # redirect to devices
    r'chmod\s+777\s+/|chown\s+.*root.*777',  # dangerous perm changes
    # Add more patterns as needed, e.g.:
    # r'apt\s+(remove|purge|autoremove)',
    # r'crontab\s+-r',
]

COMPILED_BLACKLIST = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]

# Optional: drop privileges (create user 'mcpuser' with limited rights first)
RUN_AS_NON_ROOT = False
NON_ROOT_USER = "mcpuser"   # create with: sudo adduser --system --shell /bin/false mcpuser

app = Flask(__name__)

class SafeCommandExecutor:
    """Handles command execution with timeout, blacklist, and partial output capture"""
    
    def __init__(self, cmd: str, timeout: int = COMMAND_TIMEOUT):
        self.cmd = cmd
        self.timeout = timeout
        self.proc = None
        self.stdout = ""
        self.stderr = ""
        self.returncode = None
        self.timed_out = False

    def _read_stream(self, stream, target: list):
        for line in iter(stream.readline, ''):
            target.append(line)

    def run(self) -> Dict[str, Any]:
        if any(pattern.search(self.cmd) for pattern in COMPILED_BLACKLIST):
            logger.warning(f"Blocked dangerous command: {self.cmd}")
            return {
                "stdout": "",
                "stderr": "Command blocked: matches dangerous pattern (destructive/system-altering).",
                "return_code": -403,
                "success": False,
                "timed_out": False
            }

        full_cmd = self.cmd
        if RUN_AS_NON_ROOT:
            full_cmd = f"sudo -u {NON_ROOT_USER} {self.cmd}"

        logger.info(f"Executing: {full_cmd}")

        try:
            self.proc = subprocess.Popen(
                full_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            stdout_lines = []
            stderr_lines = []

            t1 = threading.Thread(target=self._read_stream, args=(self.proc.stdout, stdout_lines))
            t2 = threading.Thread(target=self._read_stream, args=(self.proc.stderr, stderr_lines))
            t1.daemon = True
            t2.daemon = True
            t1.start()
            t2.start()

            try:
                self.returncode = self.proc.wait(timeout=self.timeout)
                t1.join()
                t2.join()
            except subprocess.TimeoutExpired:
                self.timed_out = True
                logger.warning(f"Timeout after {self.timeout}s — terminating")
                self.proc.terminate()
                try:
                    self.proc.wait(8)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                self.returncode = -1

            self.stdout = "".join(stdout_lines).rstrip()
            self.stderr = "".join(stderr_lines).rstrip()

            success = (self.returncode == 0) or (self.timed_out and (self.stdout or self.stderr))

            return {
                "stdout": self.stdout,
                "stderr": self.stderr,
                "return_code": self.returncode,
                "success": success,
                "timed_out": self.timed_out
            }

        except Exception as e:
            logger.error(f"Execution failed: {e}\n{traceback.format_exc()}")
            return {
                "stdout": self.stdout,
                "stderr": f"Server error: {str(e)}",
                "return_code": -1,
                "success": False,
                "timed_out": False
            }


def execute_safe_command(command: str) -> Dict[str, Any]:
    executor = SafeCommandExecutor(command)
    return executor.run()


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "MCP Kali Server running"}), 200


@app.route("/api/command", methods=["POST"])
def run_command():
    try:
        data = request.get_json()
        cmd = data.get("command", "").strip()

        if not cmd:
            return jsonify({"error": "Missing 'command' parameter"}), 400

        result = execute_safe_command(cmd)
        return jsonify(result)

    except Exception as e:
        logger.error(f"/api/command error: {e}")
        return jsonify({"error": str(e)}), 500


# Optional: root redirect to health
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "MCP Kali Server active", "endpoints": ["/health", "/api/command"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Kali Server (improved)")
    parser.add_argument("--ip", default="127.0.0.1", help="Bind IP (0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    DEBUG = args.debug
    app.run(host=args.ip, port=args.port, debug=DEBUG, use_reloader=DEBUG)