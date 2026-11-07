# A ComfyUI node for resizing an imput image to the nearest resolution supported by model. 
# --------------------------------------------------------------------------------
# Node Registration
# --------------------------------------------------------------------------------

from .resize_to_model_resolution import ImageRes2ModelRes

NODE_CLASS_MAPPINGS = {
    "Resize to Model Resolution": ImageRes2ModelRes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Resize to Model Resolution": "📏 Resize to Model Resolution",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]