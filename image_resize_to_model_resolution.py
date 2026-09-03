import math
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

# ---------------------------------------------------------------------------
# Resolution tables - all entries are (width, height)
# ---------------------------------------------------------------------------

QWEN_IMAGE_RESOLUTIONS = (
    # Vertical
    (928,  1664),  # 9:16
    (1056, 1584),  # 2:3
    (1104, 1472),  # 3:4  
    # Square
    (1328, 1328),  # 1:1
    # Horizontal
    (1664,  928),  # 16:9
    (1584, 1056),  # 3:2
    (1472, 1104),  # 4:3  
)

# MiniMax H3 open-weights base model, run locally in ComfyUI - not the hosted 2K regeneration

MINIMAX_H3_RESOLUTIONS = (
    # Vertical
    # 9:16  (0.2MP-2.0MP, /32, ceil)
    (  352,   608),  # 9:16  0.2MP target, 0.214MP actual
    (  416,   736),  # 9:16  0.3MP target, 0.306MP actual
    (  480,   864),  # 9:16  0.4MP target, 0.415MP actual
    (  544,   960),  # 9:16  0.5MP target, 0.522MP actual
    (  608,  1056),  # 9:16  0.6MP target, 0.642MP actual
    (  640,  1120),  # 9:16  0.7MP target, 0.717MP actual
    (  672,  1216),  # 9:16  0.8MP target, 0.817MP actual
    (  736,  1280),  # 9:16  0.9MP target, 0.942MP actual
    (  768,  1344),  # 9:16  0.98MP target, 1.032MP actual
    (  832,  1472),  # 9:16  1.2MP target, 1.225MP actual
    (  928,  1664),  # 9:16  1.5MP target, 1.544MP actual
    ( 1024,  1792),  # 9:16  1.8MP target, 1.835MP actual
    ( 1088,  1888),  # 9:16  2.0MP target, 2.054MP actual
    # 3:4  (0.2MP-2.0MP, /32, ceil)
    (  416,   544),  # 3:4  0.2MP target, 0.226MP actual
    (  480,   640),  # 3:4  0.3MP target, 0.307MP actual
    (  576,   736),  # 3:4  0.4MP target, 0.424MP actual
    (  640,   832),  # 3:4  0.5MP target, 0.532MP actual
    (  672,   896),  # 3:4  0.6MP target, 0.602MP actual
    (  736,   992),  # 3:4  0.7MP target, 0.730MP actual
    (  800,  1056),  # 3:4  0.8MP target, 0.845MP actual
    (  832,  1120),  # 3:4  0.9MP target, 0.932MP actual
    (  864,  1152),  # 3:4  0.98MP target, 0.995MP actual
    (  896,  1184),  # 3:4  1.0MP target, 1.061MP actual
    (  960,  1280),  # 3:4  1.2MP target, 1.229MP actual
    ( 1088,  1440),  # 3:4  1.5MP target, 1.567MP actual
    ( 1184,  1568),  # 3:4  1.8MP target, 1.857MP actual
    ( 1248,  1664),  # 3:4  2.0MP target, 2.077MP actual
    # 2:3  (0.2MP-2.0MP, /32, ceil)
    (  384,   576),  # 2:3  0.2MP target, 0.221MP actual
    (  448,   672),  # 2:3  0.3MP target, 0.301MP actual
    (  544,   800),  # 2:3  0.4MP target, 0.435MP actual
    (  608,   896),  # 2:3  0.5MP target, 0.545MP actual
    (  640,   960),  # 2:3  0.6MP target, 0.614MP actual
    (  704,  1056),  # 2:3  0.7MP target, 0.743MP actual
    (  736,  1120),  # 2:3  0.8MP target, 0.824MP actual
    (  800,  1184),  # 2:3  0.9MP target, 0.947MP actual
    (  832,  1216),  # 2:3  0.98MP target, 1.012MP actual
    (  832,  1248),  # 2:3  1.0MP target, 1.038MP actual
    (  896,  1344),  # 2:3  1.2MP target, 1.204MP actual
    ( 1024,  1504),  # 2:3  1.5MP target, 1.540MP actual
    ( 1120,  1664),  # 2:3  1.8MP target, 1.864MP actual
    ( 1184,  1760),  # 2:3  2.0MP target, 2.084MP actual
    # Square
    # 1:1  (0.2MP-2.0MP, /32, ceil)
    (  448,   448),  # 1:1  0.2MP target, 0.201MP actual
    (  576,   576),  # 1:1  0.3MP target, 0.332MP actual
    (  640,   640),  # 1:1  0.4MP target, 0.410MP actual
    (  736,   736),  # 1:1  0.5MP target, 0.542MP actual
    (  800,   800),  # 1:1  0.6MP target, 0.640MP actual
    (  864,   864),  # 1:1  0.7MP target, 0.746MP actual
    (  896,   896),  # 1:1  0.8MP target, 0.803MP actual
    (  960,   960),  # 1:1  0.9MP target, 0.922MP actual
    (  992,   992),  # 1:1  0.98MP target, 0.984MP actual
    ( 1024,  1024),  # 1:1  1.0MP target, 1.049MP actual
    ( 1120,  1120),  # 1:1  1.2MP target, 1.254MP actual
    ( 1248,  1248),  # 1:1  1.5MP target, 1.558MP actual
    ( 1344,  1344),  # 1:1  1.8MP target, 1.806MP actual
    ( 1440,  1440),  # 1:1  2.0MP target, 2.074MP actual
    # Horizontal
    # 4:3  (0.2MP-2.0MP, /32, ceil)
    (  544,   416),  # 4:3  0.2MP target, 0.226MP actual
    (  640,   480),  # 4:3  0.3MP target, 0.307MP actual
    (  736,   576),  # 4:3  0.4MP target, 0.424MP actual
    (  832,   640),  # 4:3  0.5MP target, 0.532MP actual
    (  896,   672),  # 4:3  0.6MP target, 0.602MP actual
    (  992,   736),  # 4:3  0.7MP target, 0.730MP actual
    ( 1056,   800),  # 4:3  0.8MP target, 0.845MP actual
    ( 1120,   832),  # 4:3  0.9MP target, 0.932MP actual
    ( 1152,   864),  # 4:3  0.98MP target, 0.995MP actual
    ( 1184,   896),  # 4:3  1.0MP target, 1.061MP actual
    ( 1280,   960),  # 4:3  1.2MP target, 1.229MP actual
    ( 1440,  1088),  # 4:3  1.5MP target, 1.567MP actual
    ( 1568,  1184),  # 4:3  1.8MP target, 1.857MP actual
    ( 1664,  1248),  # 4:3  2.0MP target, 2.077MP actual
    # 3:2  (0.2MP-2.0MP, /32, ceil)
    (  576,   384),  # 3:2  0.2MP target, 0.221MP actual
    (  672,   448),  # 3:2  0.3MP target, 0.301MP actual
    (  800,   544),  # 3:2  0.4MP target, 0.435MP actual
    (  896,   608),  # 3:2  0.5MP target, 0.545MP actual
    (  960,   640),  # 3:2  0.6MP target, 0.614MP actual
    ( 1056,   704),  # 3:2  0.7MP target, 0.743MP actual
    ( 1120,   736),  # 3:2  0.8MP target, 0.824MP actual
    ( 1184,   800),  # 3:2  0.9MP target, 0.947MP actual
    ( 1216,   832),  # 3:2  0.98MP target, 1.012MP actual
    ( 1248,   832),  # 3:2  1.0MP target, 1.038MP actual
    ( 1344,   896),  # 3:2  1.2MP target, 1.204MP actual
    ( 1504,  1024),  # 3:2  1.5MP target, 1.540MP actual
    ( 1664,  1120),  # 3:2  1.8MP target, 1.864MP actual
    ( 1760,  1184),  # 3:2  2.0MP target, 2.084MP actual
    # 16:9  (0.2MP-2.0MP, /32, ceil)
    (  608,   352),  # 16:9  0.2MP target, 0.214MP actual
    (  736,   416),  # 16:9  0.3MP target, 0.306MP actual
    (  864,   480),  # 16:9  0.4MP target, 0.415MP actual
    (  960,   544),  # 16:9  0.5MP target, 0.522MP actual
    ( 1056,   608),  # 16:9  0.6MP target, 0.642MP actual
    ( 1120,   640),  # 16:9  0.7MP target, 0.717MP actual
    ( 1216,   672),  # 16:9  0.8MP target, 0.817MP actual
    ( 1280,   736),  # 16:9  0.9MP target, 0.942MP actual
    ( 1344,   768),  # 16:9  0.98MP target, 1.032MP actual
    ( 1472,   832),  # 16:9  1.2MP target, 1.225MP actual
    ( 1664,   928),  # 16:9  1.5MP target, 1.544MP actual
    ( 1792,  1024),  # 16:9  1.8MP target, 1.835MP actual
    ( 1888,  1088),  # 16:9  2.0MP target, 2.054MP actual
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

ANIMA_RESOLUTIONS = (
    # Vertical (Portrait)
    (512,  1024),  # 1:2
    (512,   768),  # 2:3
    (640,   768),  # 5:6  ~5:8
    (768,  1024),  # 3:4
    (768,  1344),  # 4:7
    (768,  1280),  # 3:5
    (832,  1216),  # ~13:19
    (832,  1152),  # 13:18
    (896,  1152),  # 7:9
    (896,  1088),  # ~14:17
    (960,  1088),  # ~15:17
    (960,  1024),  # 15:16
    # Square
    (512,   512),  # 1:1
    (768,   768),  # 1:1
    (1024, 1024),  # 1:1
    (1280, 1280),  # 1:1
    (1536, 1536),  # 1:1
    # Horizontal (Landscape)
    (1024,  512),  # 2:1
    (768,   512),  # 3:2
    (1024,  768),  # 4:3
    (1280,  768),  # 5:3
    (1344,  768),  # ~7:4
    (1152,  832),  # ~7:5
    (1216,  832),  # ~3:2
    (1152,  896),  # 9:7
    (1088,  896),  # ~17:14
    (1088,  960),  # ~17:15
    (1024,  960),  # 16:15
)

# Krea 2 Edit LoRA 
KREA2_EDIT_RESOLUTIONS = (
    # Vertical - 1K tier
    (864,  1152),  # 3:4
    (832,  1248),  # 2:3
    (720,  1280),  # 9:16
    # Vertical - 2K tier (single-subject only)
    (1080, 1920),  # 9:16
    # Square
    (1024, 1024),  # 1:1 - 1K tier
    (1440, 1440),  # 1:1 - 2K tier (single-subject only)
    # Horizontal - 1K tier
    (1152,  864),  # 4:3
    (1248,  832),  # 3:2
    (1280,  720),  # 16:9
    # Horizontal - 2K tier (single-subject only)
    (1920, 1080),  # 16:9
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
    
    MODEL_RESOLUTIONS: dict[str, tuple[tuple[int, int], ...]] = {
        "Qwen_Image":    QWEN_IMAGE_RESOLUTIONS,
        "Z_Image_Turbo": Z_IMAGE_RESOLUTIONS,
        "MiniMax_H3":    MINIMAX_H3_RESOLUTIONS,
        "SDXL":          SDXL_RESOLUTIONS,
        "Flux":          FLUX_RESOLUTIONS,
        "Flux2":         FLUX2_RESOLUTIONS,
        "Wan_2_2":       WAN_2_2_RESOLUTIONS,
        "LTXV":          LTXV_RESOLUTIONS,
        "Anima":         ANIMA_RESOLUTIONS,
        "Krea2_Edit":    KREA2_EDIT_RESOLUTIONS,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    ["Qwen_Image", "Z_Image_Turbo", "MiniMax_H3", "SDXL", "Flux", "Flux2", "Wan_2_2", "LTXV", "Anima", "Krea2_Edit"],
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
        """Return the arctangent of h/w - used as a compact aspect-ratio proxy."""
        return math.atan2(h, w)

    @classmethod
    def _closest_resolution(
        cls,
        img_w: int,
        img_h: int,
        resolutions: tuple[tuple[int, int], ...],
    ) -> tuple[int, int]:
        """
        Return the (w, h) entry from *resolutions* that best matches the image.

        Scores by a weighted combination of:
          - aspect-ratio proximity  (primary - avoids orientation flips)
          - pixel-area proximity    (secondary - avoids huge over/under sizing)
        """
        img_angle = cls._aspect_angle(img_w, img_h)
        img_area  = img_w * img_h

        # Normalisation constants
        max_angle = math.pi / 2
        max_area  = max(r[0] * r[1] for r in resolutions)

        def score(res: tuple[int, int]) -> float:
            angle_err = abs(cls._aspect_angle(res[0], res[1]) - img_angle) / max_angle
            area_err  = abs(res[0] * res[1] - img_area) / max_area
            # Aspect ratio weighted 2× so orientation is never flipped,
            # but area proximity breaks ties between same-ratio candidates.
            return 2.0 * angle_err + area_err

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

        # When resize_longest_side > 0, restrict candidates to those whose
        # longest side is <= the requested value, then pick the best aspect
        # match among them.  This guarantees the output is always a listed
        # model resolution (never an arbitrarily scaled size).
        # If no resolution fits within the limit, fall back to the smallest
        # available resolution (by longest side) to avoid returning nothing.
        model_key = model.replace(" ", "_")
        resolutions = self.MODEL_RESOLUTIONS[model_key]
        if resize_longest_side > 0:
            candidates = tuple(r for r in resolutions if max(r) <= resize_longest_side)
            if not candidates:
                # All listed resolutions exceed the limit - use the one with
                # the smallest longest side so we at least get closest.
                candidates = (min(resolutions, key=lambda r: max(r)),)
            resolutions = candidates

        target_w, target_h = self._closest_resolution(img_w, img_h, resolutions)

        # Resize to the exact listed model resolution.
        # Aspect ratio may shift slightly - this is intentional; the node
        # guarantees the output dimensions are a valid listed resolution.
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
