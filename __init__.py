"""
ComfyUI-UniRig

UniRig integration for ComfyUI - State-of-the-art automatic rigging and skeleton extraction.

Based on: One Model to Rig Them All (SIGGRAPH 2025)
Repository: https://github.com/VAST-AI-Research/UniRig
"""

import os
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Set web directory for JavaScript extensions (FBX viewer widget)
# This tells ComfyUI where to find our JavaScript files and HTML viewer
# Files will be served at /extensions/ComfyUI-UniRig/*
WEB_DIRECTORY = "./web"

# Add static route for Three.js and other libraries
# This serves files from static/ directory at /extensions/ComfyUI-UniRig/static/
# We keep these outside of WEB_DIRECTORY to prevent ComfyUI from auto-loading .js files
try:
    from server import PromptServer
    from aiohttp import web

    static_path = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_path):
        PromptServer.instance.app.add_routes([
            web.static('/extensions/ComfyUI-UniRig/static', static_path)
        ])
except Exception as e:
    print(f"[ComfyUI-UniRig] Warning: Could not add static route: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "1.0.0"
