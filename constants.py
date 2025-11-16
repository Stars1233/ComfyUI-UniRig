"""
Constants for UniRig ComfyUI nodes.
"""

# Timeouts (in seconds)
BLENDER_TIMEOUT = 120  # 2 minutes for Blender operations
INFERENCE_TIMEOUT = 600  # 10 minutes for ML inference
PARSE_TIMEOUT = 60  # 1 minute for skeleton parsing
MESH_INFO_TIMEOUT = 30  # 30 seconds for mesh info extraction

# Mesh processing
TARGET_FACE_COUNT = 50000  # Default target for mesh decimation

# Normalization
NORMALIZATION_MIN = -1
NORMALIZATION_MAX = 1

# Export settings
DEFAULT_EXTRUDE_SIZE = 0.03  # For bone visualization in FBX export

# File patterns
SUPPORTED_3D_FORMATS = ['.glb', '.gltf', '.obj', '.fbx', '.ply']
SUPPORTED_EXPORT_FORMATS = ['fbx', 'glb', 'obj', 'ply']
