from .common_metrics_on_video_quality.calculate_fvd import calculate_fvd

import open_clip
import torch
from typing import List
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize


def clip_scores(video: torch.Tensor, prompt: List[str]):
    transforms = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    model = open_clip.create_model('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    video = transforms(video)
    text = tokenizer(prompt)
    with torch.no_grad():
        video_features = model.encode_image(video)
        text_features = model.encode_text(text)
        video_features /= video_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        scores = 100.0 * video_features @ text_features.T

    naive_score = scores.mean().item()

    video_sim_score = 100.0 * video_features[:-1] @ video_features[1:].T
    advanced_score = video_sim_score.diag().mean().item()

    return naive_score, advanced_score
