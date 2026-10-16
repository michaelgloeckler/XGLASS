"""BioMedCLIP rendering, prompts and encoders used in the paper."""
import time
from typing import List
import nibabel as nib
import numpy as np
import torch
from PIL import Image

def log(m: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def get_slice_as_image(
    ct_volume: nib.Nifti1Image,
    slice_index: int,
    window_center: float = 0.0,
    window_width: float = 2000.0
) -> Image.Image:
    data_3d = ct_volume.get_fdata()
    axial_slice = data_3d[:, :, slice_index]

    lo, hi = window_center - window_width / 2.0, window_center + window_width / 2.0
    axial_slice = np.clip(axial_slice, lo, hi)
    axial_slice = (axial_slice - lo) / (hi - lo + 1e-6)

    img_uint8 = (axial_slice * 255).astype(np.uint8)
    img_pil = Image.fromarray(np.rot90(img_uint8), mode="L").convert("RGB")
    return img_pil


def load_biomedclip(device: torch.device):
    """
    Requires: pip install open_clip_torch
    """
    try:
        import open_clip
    except Exception as e:
        raise ImportError(
            "Missing dependency open_clip_torch. Install with:\n"
            "  pip install open_clip_torch\n"
        ) from e

    model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    log(f"Loading BioMedCLIP via open_clip: {model_id}")

    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)

    model = model.to(device).eval()
    log("BioMedCLIP loaded.")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text(model, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
    tokens = tokenizer(texts)
    if isinstance(tokens, dict):
        tokens = {k: v.to(device) for k, v in tokens.items()}
        txt = model.encode_text(**tokens)
    else:
        txt = model.encode_text(tokens.to(device))
    txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
    return txt


@torch.no_grad()
def encode_images_in_chunks(model, preprocess, images: List[Image.Image],
                            device: torch.device, chunk_size: int = 32) -> torch.Tensor:
    feats = []
    for start in range(0, len(images), chunk_size):
        batch_imgs = images[start:start + chunk_size]
        batch = torch.stack([preprocess(im) for im in batch_imgs], dim=0).to(device)
        img = model.encode_image(batch)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(img.detach().cpu())
    return torch.cat(feats, dim=0)  # [N, D] on CPU


def cosine_scores(img_feats: torch.Tensor, txt_feat: torch.Tensor) -> np.ndarray:
    """
    img_feats: [N, D] CPU tensor, L2-normalized
    txt_feat:  [M, D] CPU tensor, L2-normalized (single prompt or selected prompt set)
    returns: [N] np scores averaged over prompts
    """
    if txt_feat.dim() == 1:
        txt_feat = txt_feat[None, :]
    sims = (img_feats @ txt_feat.T).numpy()   # [N, M]
    return sims.mean(axis=1)                  # [N]


def canonicalize_category(raw: str) -> str:
    s = raw.strip().lower()
    mapping = {
        "groundglass opacity": "ground-glass opacity",
        "ground glass opacity": "ground-glass opacity",
        "lung nodule or mass": "pulmonary nodule",
        "pleural abnormality": "pleural effusion",
        "high lung volume": "hyperinflation",
    }
    return mapping.get(s, raw.strip())


def build_prompts(category_raw: str, prompt_mode: str = "selected5_sentence_ensemble") -> List[str]:
    c = canonicalize_category(category_raw)
    if prompt_mode == "radiological_finding_is":
        return [f"The radiological finding is {c}."]
    if prompt_mode == "axial_ct_shows":
        return [f"This axial CT image shows {c}."]
    if prompt_mode == "ct_positive_for":
        return [f"The CT slice is positive for {c}."]
    if prompt_mode == "there_is_in_ct":
        return [f"There is {c} in this CT slice."]
    if prompt_mode == "finding_present":
        return [f"A finding of {c} is present on this CT image."]
    if prompt_mode == "selected5_sentence_ensemble":
        return [
            f"The radiological finding is {c}.",
            f"This axial CT image shows {c}.",
            f"The CT slice is positive for {c}.",
            f"There is {c} in this CT slice.",
            f"A finding of {c} is present on this CT image.",
        ]
    raise ValueError(f"Unsupported clip_prompt_mode: {prompt_mode}")

