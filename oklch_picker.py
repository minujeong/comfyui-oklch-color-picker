import comfy.model_management
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import math


def linear_to_srgb(x):
    return 12.92 * x if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055


def oklch_to_srgb8(L, C, H):
    a_ = math.cos(math.radians(H)) * C
    b_ = math.sin(math.radians(H)) * C
    L_ = L

    l_ = L_ + 0.3963377774 * a_ + 0.2158037573 * b_
    m_ = L_ - 0.1055613458 * a_ - 0.0638541728 * b_
    s_ = L_ - 0.0894841775 * a_ - 1.2914855480 * b_

    l_ = l_**3
    m_ = m_**3
    s_ = s_**3

    r = +4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_

    def clamp(x):
        return max(0, min(1, x))

    r, g, b = clamp(r), clamp(g), clamp(b)
    r, g, b = linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)
    r, g, b = (int(r * 255), int(g * 255), int(b * 255))
    return (r, g, b)


def hue_lerp_short(h0, h1, t):
    dh = ((h1 - h0 + 540.0) % 360.0) - 180.0
    return (h0 + dh * t) % 360.0


required = {
    "candidates": ("STRING", {"multiline": True, "default": "red,green,blue"}),
    "count": ("INT", {"default": 3}),
    "lch0": ("TUPLE",),
    "lch1": ("TUPLE",),
}


class OKLCHPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": required}

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("labels", "gradient")
    FUNCTION = "convert"
    CATEGORY = "OKLCH/Color"

    _clip_model = None
    _clip_proc = None

    def _load_clip(self):
        if OKLCHPicker._clip_model is None:
            device = comfy.model_management.get_torch_device()
            OKLCHPicker._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            OKLCHPicker._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return OKLCHPicker._clip_model, OKLCHPicker._clip_proc

    def convert(self, candidates, count, lch0, lch1):
        L0, C0, H0 = lch0
        L1, C1, H1 = lch1
        width, height = 224, 224
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        inv = 1.0 / (width - 1)
        for x in range(width):
            t = x * inv
            L = (L1 - L0) * t + L0
            C = (C1 - C0) * t + C0
            H = hue_lerp_short(H0, H1, t)
            rgb = oklch_to_srgb8(L, C, H)
            for y in range(height):
                pixels[x, y] = rgb

        tensor = transforms.ToTensor()(img).permute(1, 2, 0).unsqueeze(0)

        model, processor = self._load_clip()

        candidates_split = [x.strip() for x in candidates.split(",") if x.strip()]
        inputs = processor(text=candidates_split, images=img, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits_per_image[0]
        top = torch.topk(scores, min(count, len(candidates_split)))

        return (",".join([candidates_split[i.item()] for i in top.indices]), tensor)


class OKLCH:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "l": ("FLOAT", {"min": 0.000, "max": 1.000, "step": 0.001}),
                "c": ("FLOAT", {"min": 0.000, "max": 0.400, "step": 0.001}),
                "h": ("FLOAT", {"min": 0.000, "max": 360.0, "step": 1.000}),
            }
        }

    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = "oklch"
    FUNCTION = "concat"
    CATEGORY = "OKLCH/Primitives"

    def concat(self, l, c, h):
        return ((l, c, h),)


NODE_CLASS_MAPPINGS = {"OKLCHPicker": OKLCHPicker, "OKLCH": OKLCH}
NODE_DISPLAY_NAME_MAPPINGS = {"OKLCHPicker": "OKLCHPicker", "OKLCH": "OKLCH"}
