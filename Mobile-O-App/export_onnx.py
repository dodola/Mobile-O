#!/usr/bin/env python3
"""
Export Mobile-O models to ONNX format for Android deployment.

Exports 5 components:
  1. LLM (Qwen2)       → ONNX  (llm.onnx)             with all hidden states
  2. DiT Transformer   → ONNX  (transformer.onnx)
  3. VAE Decoder       → ONNX  (vae_decoder.onnx)
  4. Connector         → ONNX  (connector.onnx)
  5. Vision Encoder    → ONNX  (vision_encoder.onnx)

Loading strategy mirrors export.py exactly:
    MobileOForInferenceLM.from_pretrained(path, torch_dtype=torch.float16, local_files_only=True)

Usage:
    # Download first, then export from local path:
    python export_onnx.py checkpoints --skip-download --output-dir onnx_models

    # Download + export in one step (uses HF_ENDPOINT env var for mirror):
    HF_ENDPOINT=https://hf-mirror.com python export_onnx.py --output-dir onnx_models

    # Export only specific components:
    python export_onnx.py checkpoints --skip-download --only dit vae

    # With INT8 quantization:
    python export_onnx.py checkpoints --skip-download --quantize
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Constants — identical to export.py
# ---------------------------------------------------------------------------

SEQ_LEN_MIN     = 77
SEQ_LEN_MAX     = 512
SEQ_LEN_DEFAULT = 77

DIT_LATENT_CHANNELS  = 32
DIT_LATENT_SIZE      = 16
DIT_TEXT_HIDDEN_SIZE = 2304

VAE_IMAGE_SIZE   = 512
VAE_LATENT_SCALE = 32

VISION_IMAGE_RES = 1024

DEFAULT_MODEL  = "Amshaker/Mobile-O-0.5B"
ALL_COMPONENTS = ["dit", "vae", "connector", "vision", "llm"]

# ---------------------------------------------------------------------------
# Model wrappers — same as export.py, adapted for ONNX (no JIT trace freeze)
# ---------------------------------------------------------------------------

class DiTWrapper(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, latent, timestep, encoder_hidden_states, encoder_attention_mask):
        return self.dit(
            hidden_states=latent,
            timestep=timestep.float(),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        ).sample


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


class ConnectorWrapper(nn.Module):
    def __init__(self, connector, num_layers):
        super().__init__()
        self.connector = connector
        self.num_layers = num_layers

    def forward(self, stacked_hidden_states):
        # [batch, num_layers, seq_len, hidden_dim] → list of [batch, seq_len, hidden_dim]
        hidden_states_list = [stacked_hidden_states[:, i, :, :] for i in range(self.num_layers)]
        return self.connector(hidden_states_list)


class LLMWrapper(nn.Module):
    """Export Qwen2 LLM with all hidden states (needed by Connector on device)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Return (logits, hidden_state_0, hidden_state_1, ..., hidden_state_N)
        return (outputs.logits,) + outputs.hidden_states

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def to_fp32_cpu(module):
    return module.to(torch.float32).eval().cpu()


def onnx_export(wrapper, dummy_inputs, output_path, input_names, output_names,
                dynamic_axes, opset=18):
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,   # use TorchScript path: embeds weights inline, honours dynamic_axes
        )
    # Fix Transpose nodes whose perm attribute contains -1 values produced by the
    # TorchScript exporter when it fails to constant-fold a dynamic dimension.
    # These are invalid ONNX and rejected by ORT's type-inference pass.
    _fix_negative_transpose_perms(output_path)
    # Repack any scattered external-data files into a single <name>.onnx.data sidecar.
    # Android ORT cannot open hundreds of separate weight files reliably (fd limit,
    # path resolution issues).  A single sidecar with byte offsets works on all platforms.
    _repack_external_data(output_path)
    size_mb = output_path.stat().st_size / 1024**2
    data_file = Path(str(output_path) + ".data")
    if data_file.exists():
        data_mb = data_file.stat().st_size / 1024**2
        print(f"  Saved: {output_path} ({size_mb:.1f} MB) + {data_file.name} ({data_mb:.1f} MB)")
    else:
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def _fix_negative_transpose_perms(onnx_path: Path):
    """Fix Transpose nodes whose perm values contain -1.

    The TorchScript ONNX exporter sometimes fails to constant-fold a dynamic
    channel dimension, leaving -1 in the perm attribute.  This is invalid ONNX
    and causes ORT to raise TypeInferenceError at session creation time.

    The VAE decoder has 46 such nodes arising from NCHW↔NHWC conversions:
      [0, -1, 1, 2]  → [0, 3, 1, 2]   (NHWC→NCHW, channel dim was unresolved)
      [0,  1, 2, -1] → [0, 3, 1, 2]   (same logical operation, different encoding)

    The rule: replace every -1 in a perm with `ndim - 1` where ndim is the
    length of the perm tuple.  This recovers the intended permutation because
    -1 always represents "the last dimension" in NumPy-style indexing.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name != "perm":
                continue
            perm = list(attr.ints)
            if not any(v < 0 for v in perm):
                continue
            ndim = len(perm)
            new_perm = [v if v >= 0 else ndim + v for v in perm]
            del attr.ints[:]
            attr.ints.extend(new_perm)
            fixed += 1

    if fixed:
        print(f"  Fixed {fixed} Transpose node(s) with negative perm values.")
        onnx.save_model(model, str(onnx_path))


def _repack_external_data(onnx_path: Path):
    """Consolidate scattered per-tensor external-data files into a single
    '<model>.onnx.data' sidecar and update the protobuf references.

    If the model has no external data this is a no-op.  After repacking, all
    loose weight files are deleted so only two files remain:
        <model>.onnx          — protobuf graph (small)
        <model>.onnx.data     — all weight bytes, contiguous with offsets
    """
    import onnx
    from onnx.external_data_helper import (
        load_external_data_for_model,
        convert_model_to_external_data,
    )

    model = onnx.load(str(onnx_path), load_external_data=False)

    # Collect names of existing loose external-data files so we can clean them up.
    loose_files: set = set()
    has_external = False
    for t in model.graph.initializer:
        if t.data_location == 1:   # EXTERNAL
            has_external = True
            for entry in t.external_data:
                if entry.key == "location":
                    loose_files.add(onnx_path.parent / entry.value)

    if not has_external:
        return  # nothing to do — model is already fully inline

    # Load all weights into memory, then re-save with a single sidecar.
    print(f"  Repacking external data → {onnx_path.name}.data ...")
    load_external_data_for_model(model, str(onnx_path.parent))

    data_file_name = onnx_path.name + ".data"
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_file_name,    # relative path stored in protobuf
        size_threshold=0,           # always external (even small tensors)
        convert_attribute=False,
    )
    onnx.save_model(model, str(onnx_path))

    # Remove the now-stale loose files.
    for f in loose_files:
        try:
            f.unlink()
        except FileNotFoundError:
            pass


def maybe_quantize(onnx_path: Path, output_path: Path):
    """Apply INT8 dynamic quantization (MatMul/Gemm only — safe for NNAPI)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"  Quantizing → {output_path.name} ...")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    size_mb = output_path.stat().st_size / 1024**2
    print(f"  Quantized:  {output_path} ({size_mb:.1f} MB)")

# ---------------------------------------------------------------------------
# Per-component export functions
# ---------------------------------------------------------------------------

def export_llm(model, output_dir: Path, quantize: bool = False, opset: int = 18):
    print("\n--- LLM (Qwen2) → ONNX ---")
    output_path = output_dir / "llm.onnx"

    llm = to_fp32_cpu(LLMWrapper(model))

    dummy_ids  = torch.randint(0, 1000, (1, SEQ_LEN_DEFAULT), dtype=torch.long)
    dummy_mask = torch.ones((1, SEQ_LEN_DEFAULT), dtype=torch.long)

    # Probe output count to build output_names dynamically
    with torch.no_grad():
        sample_out = llm(dummy_ids, dummy_mask)
    num_hidden = len(sample_out) - 1   # first element is logits

    output_names = ["logits"] + [f"hidden_state_{i}" for i in range(num_hidden)]
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits":         {0: "batch", 1: "seq_len"},
    }
    for name in output_names[1:]:
        dynamic_axes[name] = {0: "batch", 1: "seq_len"}

    print(f"  hidden states: {num_hidden}, seq_len range: {SEQ_LEN_MIN}–{SEQ_LEN_MAX}")
    path = onnx_export(
        llm, (dummy_ids, dummy_mask), output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset=opset,
    )
    if quantize:
        maybe_quantize(path, output_dir / "llm_int8.onnx")
    return path


def export_dit(model, output_dir: Path, quantize: bool = False, opset: int = 18):
    print("\n--- DiT Transformer → ONNX ---")
    output_path = output_dir / "transformer.onnx"

    dit = to_fp32_cpu(DiTWrapper(model.model.dit))

    dummy = (
        torch.randn(1, DIT_LATENT_CHANNELS, DIT_LATENT_SIZE, DIT_LATENT_SIZE),
        torch.tensor([500.0]),
        torch.randn(1, SEQ_LEN_DEFAULT, DIT_TEXT_HIDDEN_SIZE),
        torch.ones(1, SEQ_LEN_DEFAULT),
    )
    dynamic_axes = {
        "latent":                 {0: "batch"},
        "timestep":               {0: "batch"},
        "encoder_hidden_states":  {0: "batch", 1: "seq_len"},
        "encoder_attention_mask": {0: "batch", 1: "seq_len"},
        "noise_pred":             {0: "batch"},
    }
    path = onnx_export(
        dit, dummy, output_path,
        input_names=["latent", "timestep", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["noise_pred"],
        dynamic_axes=dynamic_axes,
        opset=opset,
    )
    if quantize:
        maybe_quantize(path, output_dir / "transformer_int8.onnx")
    return path


def export_vae(model, output_dir: Path, opset: int = 18):
    print("\n--- VAE Decoder → ONNX ---")
    output_path = output_dir / "vae_decoder.onnx"

    vae  = model.model.vae
    c    = vae.config.latent_channels
    h = w = VAE_IMAGE_SIZE // VAE_LATENT_SCALE
    dummy = (torch.randn(1, c, h, w),)

    wrapper = to_fp32_cpu(VAEDecoderWrapper(vae))
    return onnx_export(
        wrapper, dummy, output_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
        opset=opset,
    )


def export_connector(model, output_dir: Path, opset: int = 18):
    print("\n--- Connector → ONNX ---")
    output_path = output_dir / "connector.onnx"

    connector  = model.model.diffusion_connector
    num_layers = connector.num_layers
    input_dim  = connector.input_dim
    print(f"  num_layers={num_layers}, input_dim={input_dim}, output_dim={connector.output_dim}")

    wrapper = to_fp32_cpu(ConnectorWrapper(connector, num_layers))
    dummy   = (torch.randn(1, num_layers, SEQ_LEN_DEFAULT, input_dim),)

    return onnx_export(
        wrapper, dummy, output_path,
        input_names=["stacked_hidden_states"],
        output_names=["conditioning"],
        dynamic_axes={
            "stacked_hidden_states": {0: "batch", 2: "seq_len"},
            "conditioning":          {0: "batch", 1: "seq_len"},
        },
        opset=opset,
    )


def export_vision(model, output_dir: Path, opset: int = 18):
    print("\n--- Vision Encoder → ONNX ---")
    output_path = output_dir / "vision_encoder.onnx"

    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is None:
        raise ValueError("Vision tower not found in model")
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    vision_tower = to_fp32_cpu(vision_tower)
    dummy = (torch.rand(1, 3, VISION_IMAGE_RES, VISION_IMAGE_RES),)

    return onnx_export(
        vision_tower, dummy, output_path,
        input_names=["images"],
        output_names=["image_features"],
        dynamic_axes={"images": {0: "batch"}, "image_features": {0: "batch"}},
        opset=opset,
    )


def copy_tokenizer_files(src_dir: Path, dst_dir: Path):
    """Copy tokenizer files so the Android app can load them alongside llm.onnx."""
    import shutil
    files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "added_tokens.json", "special_tokens_map.json", "config.json",
    ]
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in files:
        src = src_dir / f
        if src.exists():
            shutil.copy2(src, dst_dir / f)
            copied.append(f)
    print(f"  Tokenizer files → {dst_dir}: {copied}")

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def setup_hf_mirror(mirror):
    import urllib.request
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
        print(f"  Mirror: {mirror}")
        return
    if "HF_ENDPOINT" in os.environ:
        print(f"  HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        return
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=3)
    except Exception:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("  HuggingFace unreachable, switched to https://hf-mirror.com")


def download_model(model_id, local_dir):
    from huggingface_hub import snapshot_download
    print(f"\nDownloading {model_id} → {local_dir}")
    print("  Already-downloaded files are skipped automatically.\n")
    path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=local_dir,
        resume_download=True,
    )
    print(f"\n  Downloaded to: {path}")
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export Mobile-O models to ONNX for Android",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_path", nargs="?", default=DEFAULT_MODEL,
                        help=f"HuggingFace repo ID or local path (default: {DEFAULT_MODEL})")
    parser.add_argument("--output-dir",    default="onnx_models",
                        help="Output directory (default: onnx_models)")
    parser.add_argument("--cache-dir",     default="checkpoints",
                        help="Download cache directory (default: checkpoints)")
    parser.add_argument("--only",          nargs="+", choices=ALL_COMPONENTS,
                        help="Export only these components")
    parser.add_argument("--mirror",        default=None,
                        help="HuggingFace mirror URL, e.g. https://hf-mirror.com")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, load directly from model_path or --cache-dir")
    parser.add_argument("--quantize",      action="store_true",
                        help="Apply INT8 dynamic quantization to LLM and DiT")
    parser.add_argument("--opset",         type=int, default=18,
                        help="ONNX opset version (default: 18)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = args.only or ALL_COMPONENTS

    # --- Locate model weights ---
    raw_path   = args.model_path
    local_path = Path(raw_path).expanduser()
    is_local   = local_path.exists()

    if is_local or args.skip_download:
        model_dir = str(local_path) if is_local else args.cache_dir
        if not Path(model_dir).exists():
            print(f"ERROR: {model_dir} does not exist. Run without --skip-download first.")
            sys.exit(1)
    else:
        setup_hf_mirror(args.mirror)
        model_dir = download_model(raw_path, args.cache_dir)

    if not (Path(model_dir) / "config.json").exists():
        print(f"ERROR: config.json not found in {model_dir}. Download may be incomplete.")
        sys.exit(1)

    print(f"\nModel dir:  {model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Components: {', '.join(components)}")

    # --- Load model — exact same pattern as export.py ---
    from mobileo.model import MobileOForInferenceLM
    print("\nLoading model...")
    model = MobileOForInferenceLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    print("  Model loaded.")

    # --- Export ---
    if "llm" in components:
        export_llm(model, output_dir, quantize=args.quantize, opset=args.opset)
        copy_tokenizer_files(Path(model_dir), output_dir / "llm")

    if "dit" in components:
        export_dit(model, output_dir, quantize=args.quantize, opset=args.opset)

    if "vae" in components:
        export_vae(model, output_dir, opset=args.opset)

    if "connector" in components:
        export_connector(model, output_dir, opset=args.opset)

    if "vision" in components:
        export_vision(model, output_dir, opset=args.opset)

    # --- Summary ---
    component_files = {
        "llm":       "llm.onnx",
        "dit":       "transformer.onnx",
        "vae":       "vae_decoder.onnx",
        "connector": "connector.onnx",
        "vision":    "vision_encoder.onnx",
    }
    print("\n" + "=" * 60)
    print("ONNX Export complete!")
    print("=" * 60)
    for c in components:
        p    = output_dir / component_files[c]
        size = f"{p.stat().st_size / 1024**2:.1f} MB" if p.exists() else "not found"
        print(f"  {c:<10} {p}  ({size})")


if __name__ == "__main__":
    main()
