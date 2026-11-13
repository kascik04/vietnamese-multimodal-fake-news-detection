"""
Single-post / small-batch inference for VMTF-Net (what an extension backend or a demo calls).

    from vmtf_net.predict import VMTFPredictor
    predictor = VMTFPredictor.from_checkpoint("checkpoints/<exp>/best_vmtf_model.pt")
    result = predictor.predict(
        "KHẨN CẤP!!! Chia sẻ gấp ...", images=["post.jpg"],
        like_count=1200, comment_count=340, share_count=95,
    )
    # {'prob_fake': 0.87, 'label': 1, 'threshold': 0.42, 'beta': 0.31, 'style_signals': [...], ...}

Preprocessing is NOT re-implemented here: posts go through the same VMTFDataset as training
(word segmentation, tokenization / truncation, image transforms, OCR text, style features) with the
social / style normalizers stored in the checkpoint, so predictions match the evaluation numbers.

Unknown engagement counts are treated as 0, exactly like empty CSV cells. A freshly published post
has almost no engagement, which a model trained WITH social features has rarely seen: for
early-detection use, serve a checkpoint trained with --no_social.

CLI:
    python -m vmtf_net.predict --checkpoint <ckpt> --text "..." --image a.jpg --share_count 95
    python -m vmtf_net.predict --checkpoint <ckpt> --text "..." --kb_dir data/kb/index --post_time 2026-05-01
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

try:
    from .config import VMTFConfig
    from .dataset import (
        OCR_TEXT_COL,
        SocialFeatureNormalizer,
        VMTFCollate,
        VMTFDataset,
        load_image_processor,
        load_tokenizer,
    )
    from .evaluate import (
        json_default,
        load_checkpoint,
        load_style_normalizer,
        resolve_amp_dtype,
        run_inference,
        setup_cuda_precision,
    )
    from .style_features import clickbait_score, explain_style
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vmtf_net.config import VMTFConfig
    from vmtf_net.dataset import (
        OCR_TEXT_COL,
        SocialFeatureNormalizer,
        VMTFCollate,
        VMTFDataset,
        load_image_processor,
        load_tokenizer,
    )
    from vmtf_net.evaluate import (
        json_default,
        load_checkpoint,
        load_style_normalizer,
        resolve_amp_dtype,
        run_inference,
        setup_cuda_precision,
    )
    from vmtf_net.style_features import clickbait_score, explain_style

logger = logging.getLogger(__name__)

COUNT_FIELDS = ("like_count", "comment_count", "share_count", "angry_count", "haha_count")


class VMTFPredictor:
    """Loads a trained checkpoint once and scores posts one by one or in small batches."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: VMTFConfig,
        social_normalizer: SocialFeatureNormalizer,
        style_normalizer: Optional[SocialFeatureNormalizer] = None,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        batch_size: int = 16,
    ) -> None:
        self.config = config
        self.config.cache_images = False
        self.config.cache_dir = None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.social_normalizer = social_normalizer
        self.style_normalizer = style_normalizer
        self.threshold = float(threshold)
        self.batch_size = batch_size
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(config.amp_dtype) if self.use_amp else None

        # Loaded once (the slow part of building a dataset)
        self.tokenizer = tokenizer
        if config.uses_text and self.tokenizer is None:
            self.tokenizer = load_tokenizer(config)
        self.image_processor = image_processor
        if config.uses_image and self.image_processor is None:
            self.image_processor = load_image_processor(config)
        pad = getattr(self.tokenizer, "pad_token_id", None)
        self._collate = VMTFCollate(pad_token_id=int(pad) if pad is not None else 1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        text_model_name: Optional[str] = None,
        visual_model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> "VMTFPredictor":
        """
        Args:
            checkpoint_path: `best_vmtf_model.pt` written by training.
            device: 'cuda' / 'cpu' (auto if None).
            text_model_name / visual_model_name: local backbone paths for offline machines.
            threshold: decision threshold; default = the one tuned on validation (else 0.5).
        """
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, cfg, normalizer, checkpoint = load_checkpoint(checkpoint_path, dev, text_model_name, visual_model_name)
        if normalizer is None:
            raise ValueError("Checkpoint without social normalizer (old format): retrain or re-export it.")
        if threshold is None:
            threshold = checkpoint.get("best_threshold") or 0.5
        setup_cuda_precision(cfg.allow_tf32)
        return cls(model, cfg, normalizer, load_style_normalizer(checkpoint), threshold, dev, **kwargs)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _to_frame(self, posts: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        """Posts -> the CSV schema understood by VMTFDataset (missing fields stay empty)."""
        rows = []
        for i, post in enumerate(posts):
            images = post.get("images") or ([post["image"]] if post.get("image") else [])
            if isinstance(images, str):
                images = [images]
            row: Dict[str, Any] = {
                "sample_id": str(post.get("id") or f"post_{i}"),
                "text": post.get("text") or "",
                OCR_TEXT_COL: post.get("ocr_text") or "",
            }
            for slot in range(3):
                # absolute paths: independent of the image_dir stored in the training config
                row[f"image_{slot + 1}_local_path"] = os.path.abspath(str(images[slot])) if slot < len(images) else ""
            for field in COUNT_FIELDS:
                row[field] = post.get(field)
            rows.append(row)
        return pd.DataFrame(rows)

    def predict_batch(self, posts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Args:
            posts: dicts with 'text' and optionally 'id', 'images' (list of file paths) or 'image',
                   'ocr_text', and the engagement counts in COUNT_FIELDS.
        Returns:
            One result dict per post, in the same order.
        """
        if not posts:
            return []
        dataset = VMTFDataset(
            self._to_frame(posts),
            self.config,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            social_normalizer=self.social_normalizer,
            is_train=False,
            style_normalizer=self.style_normalizer,
        )
        if len(dataset) != len(posts):
            raise ValueError("Some posts were dropped by the dataset (ids must not start with 'EXAMPLE_').")
        batches = [
            self._collate([dataset[i] for i in range(start, min(start + self.batch_size, len(dataset)))])
            for start in range(0, len(dataset), self.batch_size)
        ]
        out = run_inference(
            self.model, batches, self.device, self.use_amp, disable_tqdm=True, amp_dtype=self.amp_dtype
        )

        results = []
        for i, post in enumerate(posts):
            prob = float(out["prob_fake"][i])
            text = post.get("text") or ""
            result: Dict[str, Any] = {
                "id": str(out["ids"][i]),
                "prob_fake": prob,
                "label": int(prob >= self.threshold),          # 1 = likely fake / misleading
                "threshold": self.threshold,
                "has_image": bool(out["has_image"][i] > 0.5) if out["has_image"] is not None else False,
                "clickbait_score": round(clickbait_score(text), 3),
                "style_signals": explain_style(text),
            }
            if out["beta"] is not None:
                # reliability weights of the gate (text vs image); beta = 0 when there is no image
                result["alpha_text"] = float(out["alpha"][i])
                result["beta_image"] = float(out["beta"][i])
            if out["cosine"] is not None and result["has_image"]:
                result["text_image_cosine"] = float(out["cosine"][i])
            results.append(result)
        return results

    def predict(self, text: str, images: Optional[Sequence[str]] = None, **fields: Any) -> Dict[str, Any]:
        """Score one post. `fields`: id, ocr_text, like_count, comment_count, share_count, angry_count, haha_count."""
        return self.predict_batch([{"text": text, "images": list(images or []), **fields}])[0]

    def predict_and_check(
        self,
        text: str,
        checker: Any,
        images: Optional[Sequence[str]] = None,
        post_time: Any = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        """
        Model score + evidence layer (vmtf_net.retrieval.PostChecker): matches against fact-checked
        claims, source credibility and a cautious user-facing message.
        """
        prediction = self.predict(text, images, **fields)
        checker.model_threshold = self.threshold
        report = checker.check(text, post_time=post_time, model_prob_fake=prediction["prob_fake"])
        report["prediction"] = prediction
        return report


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Score one post with a trained VMTF-Net checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text_file", default=None, help="Read the post text from a UTF-8 file")
    parser.add_argument("--image", nargs="*", default=[], help="Image file(s) of the post")
    parser.add_argument("--ocr_text", default=None, help="Text read inside the images (models trained with --use_ocr_text)")
    for field in COUNT_FIELDS:
        parser.add_argument(f"--{field}", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override the validation-tuned threshold")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_model_name", default=None, help="Local path of the text backbone (offline)")
    parser.add_argument("--visual_model_name", default=None, help="Local path of the visual backbone (offline)")
    parser.add_argument("--kb_dir", default=None, help="Claim knowledge base (scripts/build_claim_kb.py) for evidence")
    parser.add_argument("--post_time", default=None, help="Ignore KB entries published after this date")
    args = parser.parse_args()

    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    if not text and not args.image:
        parser.error("Provide --text / --text_file and/or --image")

    predictor = VMTFPredictor.from_checkpoint(
        args.checkpoint, args.device, args.text_model_name, args.visual_model_name, args.threshold
    )
    fields = {f: getattr(args, f) for f in COUNT_FIELDS if getattr(args, f) is not None}
    if args.ocr_text:
        fields["ocr_text"] = args.ocr_text

    if args.kb_dir:
        from vmtf_net.retrieval import ClaimKnowledgeBase, PostChecker

        checker = PostChecker(ClaimKnowledgeBase.load(args.kb_dir))
        result = predictor.predict_and_check(text or "", checker, args.image, post_time=args.post_time, **fields)
    else:
        result = predictor.predict(text or "", args.image, **fields)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
