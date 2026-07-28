"""Entry point for `python -m src.mcp_server`."""

import sys

from src.mcp_server.server import main

if __name__ == "__main__":
    sys.exit(main())
