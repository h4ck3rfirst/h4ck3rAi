<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Security%20Tool-000000?style=for-the-badge&logoColor=white" alt="Security">
  <img src="https://img.shields.io/github/license/h4ck3rfirst/h4ck3rAi?style=for-the-badge&color=green" alt="License">
</p>

<h1 align="center">h4ck3rAi</h1>

<p align="center">
  <strong>AI-assisted offensive security / C2 / red team infrastructure</strong><br>
  Modern Python-based server-side toolkit (work in progress)
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#license">License</a>
</p>

---

## About

**h4ck3rAi** is a Python-powered project focused on security tooling, including command & control (C2), server management, multi-client handling, and AI-enhanced red teaming components.

Current main files:

- `kali_server.py` — core server logic exposing Kali Linux tools via a REST API
- `mcp_server.py` — MCP (Model Context Protocol) server bridging Claude AI to Kali tools
- `v.03/` — versioned components / modules (v0.3 development branch)

Project is under active early-stage development.

---

## Features

- Lightweight C2 server implementation
- Multi-client / multi-implant management
- AI-assisted command generation and evasion suggestions
- Kali-compatible deployment and tooling integration
- Secure communication channels (encrypted)
- Modular payload / agent support
- Logging, session handling, task queuing
- Recon and enumeration: Nmap, Gobuster, Dirb, Nikto, Enum4linux, WPScan
- Exploitation via Metasploit module execution
- Credential attacks: Hydra brute-force, John the Ripper hash cracking
- Web vulnerability scanning: SQLmap, Nikto

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/h4ck3rfirst/h4ck3rAi.git
cd h4ck3rAi

# 2. Recommended: create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> Core dependencies include `fastapi`, `uvicorn`, `anthropic`, `mcp`, and `requests`.

---

## Usage

### Start the Kali Server

```bash
python kali_server.py
```

Runs on `http://localhost:8000` by default. This exposes your Kali Linux tools over a local HTTP API.

### Start the MCP Server

```bash
python mcp_server.py
```

Connects Claude AI to the Kali API via the Model Context Protocol.

### Connect Your MCP Client

Point any MCP-compatible client (e.g., Claude Desktop) to the MCP server. Claude will then have direct access to all registered Kali tools and can reason about targets, select tools, and interpret results automatically.

---

## Available Tools

| Tool | Description |
|------|-------------|
| `nmap_scan` | Port scanning and service version detection |
| `gobuster_scan` | Directory, DNS, and vhost brute-forcing |
| `dirb_scan` | Web content scanning |
| `nikto_scan` | Web server vulnerability scanning |
| `sqlmap_scan` | SQL injection detection and exploitation |
| `metasploit_run` | Execute Metasploit modules |
| `hydra_attack` | Password brute-force over many protocols |
| `john_crack` | Hash cracking with John the Ripper |
| `wpscan_analyze` | WordPress vulnerability scanning |
| `enum4linux_scan` | Windows/Samba enumeration |
| `execute_command` | Run arbitrary shell commands on Kali |
| `server_health` | Check Kali API server status |

---

## Project Structure

```
h4ck3rAi/
├── kali_server.py      # REST API server exposing Kali Linux tools
├── mcp_server.py       # MCP server — bridges Claude AI to Kali tools
├── v.03/               # Version 0.3 assets / development files
└── README.md
```

---

## Roadmap

- [x] Core Kali server with tool exposure via REST API
- [x] MCP server integration with Claude AI
- [ ] Encrypted C2 communication channels
- [ ] Web dashboard for session/client management
- [ ] Modular agent/payload support
- [ ] Persistent logging and task queuing
- [ ] Docker deployment support
- [ ] Automated reporting

---

## Legal Disclaimer

> **This tool is intended strictly for authorized penetration testing, CTF challenges, and security research on systems you own or have explicit written permission to test.**
>
> Unauthorized use against systems you do not own is illegal and unethical. The author assumes no liability for misuse of this software.

---

## Versioning

| Version | Notes |
|---------|-------|
| v0.3 | Current release — MCP integration, multi-tool support |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Made by <a href="https://github.com/h4ck3rfirst">h4ck3rfirst</a></p>
