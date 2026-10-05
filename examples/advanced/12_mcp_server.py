import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Run the MCP-style JSON-RPC server (stdin/stdout).

Run:  python examples/advanced/12_mcp_server.py
Then send a JSON-RPC line like:
  {"id":1,"method":"olmoforge_train","params":{}}
"""
from integrations.mcp_server import main

main()
