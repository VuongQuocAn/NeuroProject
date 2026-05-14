import sys
import os

# Thêm thư mục hiện tại (backend) vào sys.path
sys.path.append(os.getcwd())

try:
    from ai_core.utils.wsi_filter import WSITileFilter
    print("Initializing WSITileFilter...")
    tile_filter = WSITileFilter(top_k=5)
    print("WSITileFilter initialized successfully!")
except Exception as e:
    print(f"Error initializing WSITileFilter: {e}")
    import traceback
    traceback.print_exc()
