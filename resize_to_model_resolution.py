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

# Qwen Image 2.1 - every 32 px step, longest side 1024-2752
# (~ = nearest /32 approximation of the labelled ratio)
QWEN_IMAGE_2_1_RESOLUTIONS = (
    # Vertical
    # 9:16  (longest side 1024-2752, /32)
    ( 576, 1024),  # 9:16
    ( 608, 1056),  # ~9:16
    ( 608, 1088),  # ~9:16
    ( 640, 1120),  # ~9:16
    ( 640, 1152),  # ~9:16
    ( 672, 1184),  # ~9:16
    ( 672, 1216),  # ~9:16
    ( 704, 1248),  # ~9:16
    ( 736, 1280),  # ~9:16
    ( 736, 1312),  # ~9:16
    ( 768, 1344),  # ~9:16
    ( 768, 1376),  # ~9:16
    ( 800, 1408),  # ~9:16
    ( 800, 1440),  # ~9:16
    ( 832, 1472),  # ~9:16
    ( 832, 1504),  # ~9:16
    ( 864, 1536),  # 9:16
    ( 896, 1568),  # ~9:16
    ( 896, 1600),  # ~9:16
    ( 928, 1632),  # ~9:16
    ( 928, 1664),  # ~9:16
    ( 960, 1696),  # ~9:16
    ( 960, 1728),  # ~9:16
    ( 992, 1760),  # ~9:16
    (1024, 1792),  # ~9:16
    (1024, 1824),  # ~9:16
    (1056, 1856),  # ~9:16
    (1056, 1888),  # ~9:16
    (1088, 1920),  # ~9:16
    (1088, 1952),  # ~9:16
    (1120, 1984),  # ~9:16
    (1120, 2016),  # ~9:16
    (1152, 2048),  # 9:16
    (1184, 2080),  # ~9:16
    (1184, 2112),  # ~9:16
    (1216, 2144),  # ~9:16
    (1216, 2176),  # ~9:16
    (1248, 2208),  # ~9:16
    (1248, 2240),  # ~9:16
    (1280, 2272),  # ~9:16
    (1312, 2304),  # ~9:16
    (1312, 2336),  # ~9:16
    (1344, 2368),  # ~9:16
    (1344, 2400),  # ~9:16
    (1376, 2432),  # ~9:16
    (1376, 2464),  # ~9:16
    (1408, 2496),  # ~9:16
    (1408, 2528),  # ~9:16
    (1440, 2560),  # 9:16
    (1472, 2592),  # ~9:16
    (1472, 2624),  # ~9:16
    (1504, 2656),  # ~9:16
    (1504, 2688),  # ~9:16
    (1536, 2720),  # ~9:16
    (1536, 2752),  # ~9:16
    # 3:4  (longest side 1024-2752, /32)
    ( 768, 1024),  # 3:4
    ( 800, 1056),  # ~3:4
    ( 832, 1088),  # ~3:4
    ( 832, 1120),  # ~3:4
    ( 864, 1152),  # 3:4
    ( 896, 1184),  # ~3:4
    ( 928, 1216),  # ~3:4
    ( 928, 1248),  # ~3:4
    ( 960, 1280),  # 3:4
    ( 992, 1312),  # ~3:4
    (1024, 1344),  # ~3:4
    (1024, 1376),  # ~3:4
    (1056, 1408),  # 3:4
    (1088, 1440),  # ~3:4
    (1120, 1472),  # ~3:4
    (1120, 1504),  # ~3:4
    (1152, 1536),  # 3:4
    (1184, 1568),  # ~3:4
    (1216, 1600),  # ~3:4
    (1216, 1632),  # ~3:4
    (1248, 1664),  # 3:4
    (1280, 1696),  # ~3:4
    (1312, 1728),  # ~3:4
    (1312, 1760),  # ~3:4
    (1344, 1792),  # 3:4
    (1376, 1824),  # ~3:4
    (1408, 1856),  # ~3:4
    (1408, 1888),  # ~3:4
    (1440, 1920),  # 3:4
    (1472, 1952),  # ~3:4
    (1504, 1984),  # ~3:4
    (1504, 2016),  # ~3:4
    (1536, 2048),  # 3:4
    (1568, 2080),  # ~3:4
    (1600, 2112),  # ~3:4
    (1600, 2144),  # ~3:4
    (1632, 2176),  # 3:4
    (1664, 2208),  # ~3:4
    (1696, 2240),  # ~3:4
    (1696, 2272),  # ~3:4
    (1728, 2304),  # 3:4
    (1760, 2336),  # ~3:4
    (1792, 2368),  # ~3:4
    (1792, 2400),  # ~3:4
    (1824, 2432),  # 3:4
    (1856, 2464),  # ~3:4
    (1888, 2496),  # ~3:4
    (1888, 2528),  # ~3:4
    (1920, 2560),  # 3:4
    (1952, 2592),  # ~3:4
    (1984, 2624),  # ~3:4
    (1984, 2656),  # ~3:4
    (2016, 2688),  # 3:4
    (2048, 2720),  # ~3:4
    (2080, 2752),  # ~3:4
    # 2:3  (longest side 1024-2752, /32)
    ( 672, 1024),  # ~2:3
    ( 704, 1056),  # 2:3
    ( 736, 1088),  # ~2:3
    ( 736, 1120),  # ~2:3
    ( 768, 1152),  # 2:3
    ( 800, 1184),  # ~2:3
    ( 800, 1216),  # ~2:3
    ( 832, 1248),  # 2:3
    ( 864, 1280),  # ~2:3
    ( 864, 1312),  # ~2:3
    ( 896, 1344),  # 2:3
    ( 928, 1376),  # ~2:3
    ( 928, 1408),  # ~2:3
    ( 960, 1440),  # 2:3
    ( 992, 1472),  # ~2:3
    ( 992, 1504),  # ~2:3
    (1024, 1536),  # 2:3
    (1056, 1568),  # ~2:3
    (1056, 1600),  # ~2:3
    (1088, 1632),  # 2:3
    (1120, 1664),  # ~2:3
    (1120, 1696),  # ~2:3
    (1152, 1728),  # 2:3
    (1184, 1760),  # ~2:3
    (1184, 1792),  # ~2:3
    (1216, 1824),  # 2:3
    (1248, 1856),  # ~2:3
    (1248, 1888),  # ~2:3
    (1280, 1920),  # 2:3
    (1312, 1952),  # ~2:3
    (1312, 1984),  # ~2:3
    (1344, 2016),  # 2:3
    (1376, 2048),  # ~2:3
    (1376, 2080),  # ~2:3
    (1408, 2112),  # 2:3
    (1440, 2144),  # ~2:3
    (1440, 2176),  # ~2:3
    (1472, 2208),  # 2:3
    (1504, 2240),  # ~2:3
    (1504, 2272),  # ~2:3
    (1536, 2304),  # 2:3
    (1568, 2336),  # ~2:3
    (1568, 2368),  # ~2:3
    (1600, 2400),  # 2:3
    (1632, 2432),  # ~2:3
    (1632, 2464),  # ~2:3
    (1664, 2496),  # 2:3
    (1696, 2528),  # ~2:3
    (1696, 2560),  # ~2:3
    (1728, 2592),  # 2:3
    (1760, 2624),  # ~2:3
    (1760, 2656),  # ~2:3
    (1792, 2688),  # 2:3
    (1824, 2720),  # ~2:3
    (1824, 2752),  # ~2:3
    # Square
    # 1:1  (side 1024-2752, /32)
    (1024, 1024),  # 1:1
    (1056, 1056),  # 1:1
    (1088, 1088),  # 1:1
    (1120, 1120),  # 1:1
    (1152, 1152),  # 1:1
    (1184, 1184),  # 1:1
    (1216, 1216),  # 1:1
    (1248, 1248),  # 1:1
    (1280, 1280),  # 1:1
    (1312, 1312),  # 1:1
    (1344, 1344),  # 1:1
    (1376, 1376),  # 1:1
    (1408, 1408),  # 1:1
    (1440, 1440),  # 1:1
    (1472, 1472),  # 1:1
    (1504, 1504),  # 1:1
    (1536, 1536),  # 1:1
    (1568, 1568),  # 1:1
    (1600, 1600),  # 1:1
    (1632, 1632),  # 1:1
    (1664, 1664),  # 1:1
    (1696, 1696),  # 1:1
    (1728, 1728),  # 1:1
    (1760, 1760),  # 1:1
    (1792, 1792),  # 1:1
    (1824, 1824),  # 1:1
    (1856, 1856),  # 1:1
    (1888, 1888),  # 1:1
    (1920, 1920),  # 1:1
    (1952, 1952),  # 1:1
    (1984, 1984),  # 1:1
    (2016, 2016),  # 1:1
    (2048, 2048),  # 1:1
    (2080, 2080),  # 1:1
    (2112, 2112),  # 1:1
    (2144, 2144),  # 1:1
    (2176, 2176),  # 1:1
    (2208, 2208),  # 1:1
    (2240, 2240),  # 1:1
    (2272, 2272),  # 1:1
    (2304, 2304),  # 1:1
    (2336, 2336),  # 1:1
    (2368, 2368),  # 1:1
    (2400, 2400),  # 1:1
    (2432, 2432),  # 1:1
    (2464, 2464),  # 1:1
    (2496, 2496),  # 1:1
    (2528, 2528),  # 1:1
    (2560, 2560),  # 1:1
    (2592, 2592),  # 1:1
    (2624, 2624),  # 1:1
    (2656, 2656),  # 1:1
    (2688, 2688),  # 1:1
    (2720, 2720),  # 1:1
    (2752, 2752),  # 1:1
    # Horizontal
    # 16:9  (longest side 1024-2752, /32)
    (1024,  576),  # 16:9
    (1056,  608),  # ~16:9
    (1088,  608),  # ~16:9
    (1120,  640),  # ~16:9
    (1152,  640),  # ~16:9
    (1184,  672),  # ~16:9
    (1216,  672),  # ~16:9
    (1248,  704),  # ~16:9
    (1280,  736),  # ~16:9
    (1312,  736),  # ~16:9
    (1344,  768),  # ~16:9
    (1376,  768),  # ~16:9
    (1408,  800),  # ~16:9
    (1440,  800),  # ~16:9
    (1472,  832),  # ~16:9
    (1504,  832),  # ~16:9
    (1536,  864),  # 16:9
    (1568,  896),  # ~16:9
    (1600,  896),  # ~16:9
    (1632,  928),  # ~16:9
    (1664,  928),  # ~16:9
    (1696,  960),  # ~16:9
    (1728,  960),  # ~16:9
    (1760,  992),  # ~16:9
    (1792, 1024),  # ~16:9
    (1824, 1024),  # ~16:9
    (1856, 1056),  # ~16:9
    (1888, 1056),  # ~16:9
    (1920, 1088),  # ~16:9
    (1952, 1088),  # ~16:9
    (1984, 1120),  # ~16:9
    (2016, 1120),  # ~16:9
    (2048, 1152),  # 16:9
    (2080, 1184),  # ~16:9
    (2112, 1184),  # ~16:9
    (2144, 1216),  # ~16:9
    (2176, 1216),  # ~16:9
    (2208, 1248),  # ~16:9
    (2240, 1248),  # ~16:9
    (2272, 1280),  # ~16:9
    (2304, 1312),  # ~16:9
    (2336, 1312),  # ~16:9
    (2368, 1344),  # ~16:9
    (2400, 1344),  # ~16:9
    (2432, 1376),  # ~16:9
    (2464, 1376),  # ~16:9
    (2496, 1408),  # ~16:9
    (2528, 1408),  # ~16:9
    (2560, 1440),  # 16:9
    (2592, 1472),  # ~16:9
    (2624, 1472),  # ~16:9
    (2656, 1504),  # ~16:9
    (2688, 1504),  # ~16:9
    (2720, 1536),  # ~16:9
    (2752, 1536),  # ~16:9
    # 4:3  (longest side 1024-2752, /32)
    (1024,  768),  # 4:3
    (1056,  800),  # ~4:3
    (1088,  832),  # ~4:3
    (1120,  832),  # ~4:3
    (1152,  864),  # 4:3
    (1184,  896),  # ~4:3
    (1216,  928),  # ~4:3
    (1248,  928),  # ~4:3
    (1280,  960),  # 4:3
    (1312,  992),  # ~4:3
    (1344, 1024),  # ~4:3
    (1376, 1024),  # ~4:3
    (1408, 1056),  # 4:3
    (1440, 1088),  # ~4:3
    (1472, 1120),  # ~4:3
    (1504, 1120),  # ~4:3
    (1536, 1152),  # 4:3
    (1568, 1184),  # ~4:3
    (1600, 1216),  # ~4:3
    (1632, 1216),  # ~4:3
    (1664, 1248),  # 4:3
    (1696, 1280),  # ~4:3
    (1728, 1312),  # ~4:3
    (1760, 1312),  # ~4:3
    (1792, 1344),  # 4:3
    (1824, 1376),  # ~4:3
    (1856, 1408),  # ~4:3
    (1888, 1408),  # ~4:3
    (1920, 1440),  # 4:3
    (1952, 1472),  # ~4:3
    (1984, 1504),  # ~4:3
    (2016, 1504),  # ~4:3
    (2048, 1536),  # 4:3
    (2080, 1568),  # ~4:3
    (2112, 1600),  # ~4:3
    (2144, 1600),  # ~4:3
    (2176, 1632),  # 4:3
    (2208, 1664),  # ~4:3
    (2240, 1696),  # ~4:3
    (2272, 1696),  # ~4:3
    (2304, 1728),  # 4:3
    (2336, 1760),  # ~4:3
    (2368, 1792),  # ~4:3
    (2400, 1792),  # ~4:3
    (2432, 1824),  # 4:3
    (2464, 1856),  # ~4:3
    (2496, 1888),  # ~4:3
    (2528, 1888),  # ~4:3
    (2560, 1920),  # 4:3
    (2592, 1952),  # ~4:3
    (2624, 1984),  # ~4:3
    (2656, 1984),  # ~4:3
    (2688, 2016),  # 4:3
    (2720, 2048),  # ~4:3
    (2752, 2080),  # ~4:3
    # 3:2  (longest side 1024-2752, /32)
    (1024,  672),  # ~3:2
    (1056,  704),  # 3:2
    (1088,  736),  # ~3:2
    (1120,  736),  # ~3:2
    (1152,  768),  # 3:2
    (1184,  800),  # ~3:2
    (1216,  800),  # ~3:2
    (1248,  832),  # 3:2
    (1280,  864),  # ~3:2
    (1312,  864),  # ~3:2
    (1344,  896),  # 3:2
    (1376,  928),  # ~3:2
    (1408,  928),  # ~3:2
    (1440,  960),  # 3:2
    (1472,  992),  # ~3:2
    (1504,  992),  # ~3:2
    (1536, 1024),  # 3:2
    (1568, 1056),  # ~3:2
    (1600, 1056),  # ~3:2
    (1632, 1088),  # 3:2
    (1664, 1120),  # ~3:2
    (1696, 1120),  # ~3:2
    (1728, 1152),  # 3:2
    (1760, 1184),  # ~3:2
    (1792, 1184),  # ~3:2
    (1824, 1216),  # 3:2
    (1856, 1248),  # ~3:2
    (1888, 1248),  # ~3:2
    (1920, 1280),  # 3:2
    (1952, 1312),  # ~3:2
    (1984, 1312),  # ~3:2
    (2016, 1344),  # 3:2
    (2048, 1376),  # ~3:2
    (2080, 1376),  # ~3:2
    (2112, 1408),  # 3:2
    (2144, 1440),  # ~3:2
    (2176, 1440),  # ~3:2
    (2208, 1472),  # 3:2
    (2240, 1504),  # ~3:2
    (2272, 1504),  # ~3:2
    (2304, 1536),  # 3:2
    (2336, 1568),  # ~3:2
    (2368, 1568),  # ~3:2
    (2400, 1600),  # 3:2
    (2432, 1632),  # ~3:2
    (2464, 1632),  # ~3:2
    (2496, 1664),  # 3:2
    (2528, 1696),  # ~3:2
    (2560, 1696),  # ~3:2
    (2592, 1728),  # 3:2
    (2624, 1760),  # ~3:2
    (2656, 1760),  # ~3:2
    (2688, 1792),  # 3:2
    (2720, 1824),  # ~3:2
    (2752, 1824),  # ~3:2
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
        "Qwen_Image_2_1": QWEN_IMAGE_2_1_RESOLUTIONS,
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

    # Dimensions are snapped to the nearest multiple of this value whenever
    # the node has to scale (megapixels > 0 or resize_longest_side > 0).
    # This plays the role of "resolution_steps" in ComfyUI's Scale Image to
    # Total Pixels node. Listed resolutions that already match the requested
    # longest side pass through untouched.
    MODEL_MULTIPLE: dict[str, int] = {
        "Qwen_Image":     16,
        "Qwen_Image_2_1": 32,
        "Z_Image_Turbo":  16,
        "MiniMax_H3":     32,
        "SDXL":           64,
        "Flux":           16,
        "Flux2":          16,
        "Wan_2_2":        16,
        "LTXV":           32,
        "Anima":          16,
        "Krea2_Edit":     16,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    ["Qwen_Image", "Qwen_Image_2_1", "Z_Image_Turbo", "MiniMax_H3", "SDXL", "Flux", "Flux2", "Wan_2_2", "LTXV", "Anima", "Krea2_Edit"],
                ),
                "interpolation_mode": (
                    ["bicubic", "bilinear", "lanczos", "nearest", "nearest exact"],
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": (
                            "0 = off.\n"
                            "Any other value: the listed resolution with the "
                            "closest aspect ratio is scaled to this many "
                            "megapixels (1 MP = 1024 x 1024 px, same as "
                            "ComfyUI's Scale Image to Total Pixels), with both "
                            "sides rounded to the nearest multiple the selected "
                            "model requires. Takes priority over "
                            "resize_longest_side."
                        ),
                    },
                ),
                "resize_longest_side": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 16,
                        "tooltip": (
                            "0 = use the model's native resolution exactly "
                            "(when megapixels is also 0).\n"
                            "Any other value: the listed resolution with the "
                            "closest aspect ratio is scaled so its longest side "
                            "equals this number, with both sides rounded to the "
                            "nearest multiple the selected model requires "
                            "(e.g., 16 for Wan 2.2). Ignored when megapixels "
                            "is above 0."
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

    @staticmethod
    def _snap(value: float, multiple: int) -> int:
        """Round *value* to the nearest multiple (half rounds up), min 1 multiple."""
        return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)

    @classmethod
    def _fit_longest_side(
        cls,
        img_w: int,
        img_h: int,
        resolutions: tuple[tuple[int, int], ...],
        longest: int,
        multiple: int,
    ) -> tuple[int, int]:
        """
        Scale a listed resolution so its longest side is *longest*, snapped to
        *multiple*.

        Every listed resolution is scaled and snapped first; the winner is the
        one whose *final* aspect ratio is closest to the image's. Ties (e.g.
        several 9:16 entries snapping to the same size) go to the entry whose
        native longest side is nearest the request, so an exact listed
        resolution is returned whenever one fits.
        """
        img_log_ar = math.log(img_w / img_h)

        def fitted(res: tuple[int, int]) -> tuple[int, int]:
            w, h = res
            long_side = max(w, h)
            if long_side == longest:
                return w, h  # already exactly the requested size
            scale = longest / long_side
            if w >= h:
                return cls._snap(longest, multiple), cls._snap(h * scale, multiple)
            return cls._snap(w * scale, multiple), cls._snap(longest, multiple)

        def score(res: tuple[int, int]) -> tuple[float, int]:
            fw, fh = fitted(res)
            return (abs(math.log(fw / fh) - img_log_ar), abs(max(res) - longest))

        return fitted(min(resolutions, key=score))

    @classmethod
    def _fit_megapixels(
        cls,
        img_w: int,
        img_h: int,
        resolutions: tuple[tuple[int, int], ...],
        megapixels: float,
        multiple: int,
    ) -> tuple[int, int]:
        """
        Scale a listed resolution to *megapixels*, snapped to *multiple*.

        The maths mirrors ComfyUI's "Scale Image to Total Pixels" node:
        1 MP = 1024 * 1024 px, scale = sqrt(total / area), and each side is
        rounded to the step independently. The difference is that the scaled
        shape is a listed model resolution rather than the raw image, so the
        result always keeps one of the model's supported aspect ratios.

        Every listed resolution is scaled and snapped first; the winner is the
        one whose *final* aspect ratio is closest to the image's. Ties go to
        the entry whose native area is nearest the request.
        """
        total = megapixels * 1024 * 1024
        img_log_ar = math.log(img_w / img_h)

        def fitted(res: tuple[int, int]) -> tuple[int, int]:
            w, h = res
            scale_by = math.sqrt(total / (w * h))
            return cls._snap(w * scale_by, multiple), cls._snap(h * scale_by, multiple)

        def score(res: tuple[int, int]) -> tuple[float, float]:
            fw, fh = fitted(res)
            return (abs(math.log(fw / fh) - img_log_ar), abs(res[0] * res[1] - total))

        return fitted(min(resolutions, key=score))

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(
        self,
        image: torch.Tensor,
        model: str,
        interpolation_mode: str,
        megapixels: float = 0.0,
        resize_longest_side: int = 0,
    ):
        # Resolve interpolation enum
        interp_key = interpolation_mode.upper().replace(" ", "_")
        interp_enum = getattr(InterpolationMode, interp_key)

        # image tensor shape: (B, H, W, C)
        _, img_h, img_w, _ = image.shape

        model_key = model.replace(" ", "_")
        resolutions = self.MODEL_RESOLUTIONS[model_key]
        multiple = self.MODEL_MULTIPLE.get(model_key, 8)

        if megapixels > 0:
            target_w, target_h = self._fit_megapixels(
                img_w, img_h, resolutions, megapixels, multiple
            )
        elif resize_longest_side > 0:
            target_w, target_h = self._fit_longest_side(
                img_w, img_h, resolutions, resize_longest_side, multiple
            )
        else:
            target_w, target_h = self._closest_resolution(img_w, img_h, resolutions)

        # Resize to the chosen resolution. Aspect ratio may shift slightly -
        # this is intentional; output is always a model-valid size.
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

        # "ui" feeds the on-node readout (see web/res_display.js)
        return {
            "ui": {"text": [f"{target_w} x {target_h}"]},
            "result": (image, target_w, target_h),
        }
