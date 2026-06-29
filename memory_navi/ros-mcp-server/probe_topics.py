"""Minimal, host-agnostic probe to verify ROS2 topics are visible THROUGH the MCP layer.

No LLM involved. This is only a plumbing check: it talks to the same FastMCP server
instance the agent will use, points it at rosbridge, and lists topics.

Prereqs (see MCP_SETUP.md):
  - rosbridge running on 127.0.0.1:9090
  - a ROS2 graph alive (e.g. the Webots robomaster sim publishing /agent0/... /agent1/...)

Run (from this directory):
  uv run python probe_topics.py
  # or point at another rosbridge (e.g. the real car):
  uv run python probe_topics.py --ip 192.168.101.138 --port 9090
"""

import argparse
import asyncio

from fastmcp import Client

# In-memory transport: import the exact FastMCP instance the server exposes.
from ros_mcp.main import mcp


def _extract(result):
    """Pull a plain value out of a fastmcp CallToolResult across versions."""
    for attr in ("data", "structured_content"):
        v = getattr(result, attr, None)
        if v is not None:
            return v
    content = getattr(result, "content", None)
    if content:
        return "\n".join(getattr(c, "text", str(c)) for c in content)
    return result


async def main(ip: str, port: int):
    print(f"[probe] connecting MCP server -> rosbridge {ip}:{port}")
    async with Client(mcp) as client:
        # 1) point the server at the rosbridge endpoint (not hardcoded; switch here for real robot)
        conn = _extract(await client.call_tool("connect_to_robot", {"ip": ip, "port": port}))
        print("[probe] connect_to_robot ->", conn)

        # 2) list all topics visible through MCP
        topics_res = _extract(await client.call_tool("get_topics", {}))
        print("[probe] get_topics ->", topics_res)

        topics = topics_res.get("topics", []) if isinstance(topics_res, dict) else []
        print(f"\n[probe] {len(topics)} topics visible through MCP:")
        for t in sorted(topics):
            print("   ", t)

        # 3) quick sanity on what memory_navi cares about
        wanted = ["/agent0/camera/image_color", "/agent0/cmd_vel"]
        print("\n[probe] key topics check:")
        for w in wanted:
            print(f"   {'OK ' if w in topics else 'MISSING'}  {w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verify ROS2 topics are visible through ros-mcp-server")
    ap.add_argument("--ip", default="127.0.0.1", help="rosbridge IP (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=9090, help="rosbridge port (default 9090)")
    args = ap.parse_args()
    asyncio.run(main(args.ip, args.port))
