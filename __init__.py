# A ComfyUI node for resizing an imput image to the nearest resolution supported by a selected model. 
# --------------------------------------------------------------------------------
# Node Registration
# --------------------------------------------------------------------------------

from .image_resize_to_model_resolution import ImageRes2ModelRes

NODE_CLASS_MAPPINGS = {
    "Image Resize to Nearest Model Resolution": ImageRes2ModelRes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Resize to Nearest Model Resolution": "📏Image Resize to Nearest Model Resolution",
}
