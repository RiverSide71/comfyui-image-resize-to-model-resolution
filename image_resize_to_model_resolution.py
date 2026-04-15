import math

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

# ---------------------------------------------------------------------------
# Resolution tables — all entries are (width, height)
# ---------------------------------------------------------------------------

QWEN_IMAGE_RESOLUTIONS = (
    # Vertical
    (928,  1664),  # 9:16
    (1056, 1584),  # 2:3
    (1140, 1472),  # 3:4
    # Square
    (1328, 1328),  # 1:1
    # Horizontal
    (1664,  928),  # 16:9
    (1584, 1056),  # 3:2
    (1472, 1140),  # 4:3
)

Z_IMAGE_RESOLUTIONS = (
    # Vertical
    (720,  1280),  # 9:16
    (900,  1600),  # 9:16
    (832,  1248),  # 2:3
    (1024, 1536),  # 2:3
    (864,  1152),  # 3:4
    (960,  1280),  # 3:4
    # Square
    (1024, 1024),  # 1:1
    (1280, 1280),  # 1:1
    (1536, 1536),  # 1:1
    # Horizontal
    (1280,  720),  # 16:9
    (1600,  900),  # 16:9
    (1248,  832),  # 3:2
    (1536, 1024),  # 3:2
    (1152,  864),  # 4:3
    (1280,  960),  # 4:3
)

SDXL_RESOLUTIONS = (
    # Vertical
    (704,  1408),  # 1:2
    (704,  1344),  # 11:21
    (768,  1344),  # 4:7
    (768,  1280),  # 3:5
    (832,  1216),  # 13:19
    (832,  1152),  # 13:18
    (896,  1152),  # 7:9
    (896,  1088),  # 14:17
    (960,  1088),  # 15:17
    (960,  1024),  # 15:16
    # Square
    (1024, 1024),  # 1:1
    # Horizontal
    (1024,  960),  # 16:15
    (1088,  960),  # 17:15
    (1088,  896),  # 17:14
    (1152,  896),  # 18:13
    (1152,  832),  # 18:13
    (1216,  832),  # 19:13
    (1280,  768),  # 5:3
    (1280,  704),  # 7:4
    (1344,  768),  # 21:11
    (1344,  704),  # 21:11
    (1408,  704),  # 2:1
    (1408,  640),  # 11:5
    (1472,  704),  # 2:1
    (1536,  640),  # 12:5
    (1600,  640),  # 5:2
    (1664,  576),  # 26:9
    (1728,  576),  # 3:1
)

FLUX_RESOLUTIONS = (
    # Vertical
    (768,  1024),  # 3:4
    (960,  1280),  # 3:4
    (960,  1440),  # 2:3
    (1024, 1536),  # 2:3
    # Square
    (512,   512),  # 1:1
    (768,   768),  # 1:1
    (1024, 1024),  # 1:1
    (1536, 1536),  # 1:1
    # Horizontal
    (1024,  768),  # 4:3
    (1280,  960),  # 4:3
    (1440,  960),  # 3:2
    (1536, 1024),  # 3:2
)

FLUX2_RESOLUTIONS = (
    # Vertical
    (1408, 2816),  # 1:2
    (1408, 2688),  # 11:21
    (1536, 2688),  # 4:7
    (1536, 2560),  # 3:5
    (1664, 2432),  # 13:19
    (1664, 2304),  # 13:18
    (1792, 2304),  # 7:9
    (1792, 2176),  # 14:17
    (1920, 2176),  # 15:17
    (1920, 2048),  # 15:16
    # Square
    (2048, 2048),  # 1:1
    # Horizontal
    (2048, 1920),  # 16:15
    (2176, 1920),  # 17:15
    (2176, 1792),  # 17:14
    (2304, 1792),  # 18:13
    (2304, 1664),  # 18:13
    (2432, 1664),  # 19:13
    (2560, 1536),  # 5:3
    (2560, 1408),  # 7:4
    (2688, 1536),  # 21:11
    (2688, 1408),  # 21:11
    (2816, 1408),  # 2:1
    (2816, 1280),  # 11:5
    (2944, 1408),  # 2:1
    (3072, 1280),  # 12:5
    (3200, 1280),  # 5:2
    (3328, 1152),  # 26:9
    (3456, 1152),  # 3:1
)

WAN_2_2_RESOLUTIONS = (
    # Vertical (Portrait)
    (368,   624),  # 9:16
    (480,   848),  # 9:16
    (576,  1008),  # 9:16
    (608,  1072),  # 9:16
    (672,  1184),  # 9:16
    (720,  1264),  # 9:16
    (384,   576),  # 2:3
    (528,   768),  # 2:3
    (624,   912),  # 2:3
    (656,   960),  # 2:3
    (736,  1072),  # 2:3
    (784,  1136),  # 2:3
    (416,   544),  # 3:4
    (560,   720),  # 3:4
    (672,   864),  # 3:4
    (720,   912),  # 3:4
    (784,  1008),  # 3:4
    (848,  1088),  # 3:4
    # Square
    (480,   480),  # 1:1
    (640,   640),  # 1:1
    (768,   768),  # 1:1
    (800,   800),  # 1:1
    (880,   880),  # 1:1
    (960,   960),  # 1:1
    # Horizontal (Landscape)
    (624,   368),  # 16:9
    (848,   480),  # 16:9
    (1008,  576),  # 16:9
    (1072,  608),  # 16:9
    (1184,  672),  # 16:9
    (1264,  720),  # 16:9
    (576,   384),  # 3:2
    (768,   528),  # 3:2
    (912,   624),  # 3:2
    (960,   656),  # 3:2
    (1072,  736),  # 3:2
    (1136,  784),  # 3:2
    (544,   416),  # 4:3
    (720,   560),  # 4:3
    (864,   672),  # 4:3
    (912,   720),  # 4:3
    (1008,  784),  # 4:3
    (1088,  848),  # 4:3
)

LTXV_RESOLUTIONS = (
    # Vertical (Portrait)
    (1080, 1920),  # 9:16
    ( 720, 1280),  # 9:16
    ( 480,  832),  # 15:26
    ( 384,  512),  # 3:4
    ( 288,  480),  # 3:5
    # Square
    ( 832,  832),  # 1:1
    ( 640,  640),  # 1:1
    ( 512,  512),  # 1:1
    ( 384,  384),  # 1:1
    ( 256,  256),  # 1:1
    # Horizontal (Landscape)
    (1920, 1080),  # 16:9
    (1280,  720),  # 16:9
    (1216,  704),  # 19:11
    (1088,  832),  # 17:13
    ( 768,  512),  # 3:2
    ( 512,  384),  # 4:3
    ( 480,  288),  # 5:3
)

# ---------------------------------------------------------------------------
# Node definition
# ---------------------------------------------------------------------------

class ImageRes2ModelRes:
    CATEGORY = "riversidenodes"
    #ICON = "ruler_icon.svg"  # place the file next to your node's .py
    
    MODEL_RESOLUTIONS: dict[str, tuple[tuple[int, int], ...]] = {
        "Qwen_Image":    QWEN_IMAGE_RESOLUTIONS,
        "Z_Image_Turbo": Z_IMAGE_RESOLUTIONS,
        "SDXL":          SDXL_RESOLUTIONS,
        "Flux":          FLUX_RESOLUTIONS,
        "Flux2":         FLUX2_RESOLUTIONS,
        "Wan_2_2":       WAN_2_2_RESOLUTIONS,
        "LTXV":          LTXV_RESOLUTIONS,
    }

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "model": (
                ["Qwen Image", "Z Image Turbo", "SDXL", "Flux", "Flux2", "Wan 2.2", "LTXV"],
            ),
            "interpolation_mode": (
                ["bicubic", "bilinear", "lanczos", "nearest", "nearest exact"],
            ),
            "resize_longest_side": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 8,
                    "tooltip": (
                        "0 = use the model's native resolution exactly.\n"
                        "Any other value: the best-matching resolution is "
                        "scaled proportionally so its longest side equals "
                        "this number (rounded to the nearest 8 px)."
                    ),
                },
            ),
        }
    }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "execute"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aspect_angle(w: int, h: int) -> float:
        """Return the arctangent of h/w — used as a compact aspect-ratio proxy."""
        return math.atan2(h, w)

    @classmethod
    def _closest_resolution(
        cls,
        img_w: int,
        img_h: int,
        resolutions: tuple[tuple[int, int], ...],
        target_longest_side: int = 0,
    ) -> tuple[int, int]:
        """
        Return the (w, h) entry from *resolutions* that best matches the image.

        When target_longest_side == 0 (default):
            Picks purely by aspect-ratio proximity.

        When target_longest_side > 0:
            Scores each candidate by a weighted combination of aspect-ratio
            difference and longest-side difference, so the returned resolution
            is still an exact entry from the model's list.
        """
        img_angle = cls._aspect_angle(img_w, img_h)

        if target_longest_side <= 0:
            return min(
                resolutions,
                key=lambda res: abs(cls._aspect_angle(res[0], res[1]) - img_angle),
            )

        # Normalise both terms to [0, 1] range for balanced scoring.
        max_angle = math.pi / 2          # maximum possible atan2 difference
        max_pixels = max(max(r) for r in resolutions)

        def score(res: tuple[int, int]) -> float:
            angle_err  = abs(cls._aspect_angle(res[0], res[1]) - img_angle) / max_angle
            size_err   = abs(max(res) - target_longest_side) / max_pixels
            return angle_err + size_err

        return min(resolutions, key=score)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(
        self,
        image: torch.Tensor,
        model: str,
        interpolation_mode: str,
        resize_longest_side: int,
    ):
        # Resolve interpolation enum
        interp_key = interpolation_mode.upper().replace(" ", "_")
        interp_enum = getattr(InterpolationMode, interp_key)

        # image tensor shape: (B, H, W, C)
        _, img_h, img_w, _ = image.shape

        # Pick the best-matching resolution for the chosen model.
        # When resize_longest_side > 0 it influences which exact model
        # resolution is selected, but the output is always an entry from
        # the model's own list — never an arbitrary scaled dimension.
        model_key = model.replace(" ", "_")
        target_w, target_h = self._closest_resolution(
            img_w, img_h,
            self.MODEL_RESOLUTIONS[model_key],
            target_longest_side=resize_longest_side,
        )

        # Resize to the exact model resolution.
        # Aspect ratio may shift slightly — this is intentional; the node
        # guarantees the output dimensions are valid for the chosen model.
        #
        # Lanczos is not supported by torchvision for tensor inputs, so we
        # route it through PIL instead.
        if interp_enum == InterpolationMode.LANCZOS:
            frames = []
            for i in range(image.shape[0]):
                # (H, W, C) float32 → uint8 PIL image
                frame_np = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_img  = PILImage.fromarray(frame_np)
                pil_img  = pil_img.resize((target_w, target_h), PILImage.LANCZOS)
                # Back to float32 (H, W, C)
                frames.append(torch.from_numpy(np.array(pil_img)).float() / 255.0)
            image = torch.stack(frames)
        else:
            image = image.permute(0, 3, 1, 2)
            image = F.resize(
                image,
                [target_h, target_w],
                interpolation=interp_enum,
                antialias=True,
            )
            image = image.permute(0, 2, 3, 1)

        return (image, target_w, target_h)
