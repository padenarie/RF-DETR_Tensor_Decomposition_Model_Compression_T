# Soccer Broadcast AI — RF-DETR with Tensor Decomposition

A research pipeline for **soccer broadcast video analytics** using a heavily customised version of [RF-DETR](https://github.com/roboflow/rf-detr) with a DINOv2 backbone, extended with two families of **neural network compression via tensor decomposition**: Tensor Train (TT) and CP (CANDECOMP/PARAFAC).

---

## 🏆 Key Results — Compression Benchmarks

> **Benchmarked on `RFDETRSegNano` (CPU, PyTorch 2.1+cpu), `dog.jpg` validation image.**  
> All figures are relative to the uncompressed FP32 baseline model.

| Method | Parameter count | Memory at rest | Inference latency |
|--------|----------------|----------------|-------------------|
| Baseline (FP32, no compression) | 100 % | 100 % | 1.0 × |
| TT — MLP only (`ε = 0.3`) | ~40 % | ~40 % | +5 – 15 % |
| TT — MLP + Attention (`ε = 0.3`) | ~33 % | ~33 % | +15 – 30 % |
| TT — Full (`ε = 0.3`) | ~29 % | ~29 % | +20 – 40 % |
| CP — Rank 16 (layers 8 – 11) | ~45 % | ~45 % | +5 – 20 % |
| **TT (MLP, `ε = 0.3`) + FP16 cores** | ~40 % | **~20 % 🔻** | +5 – 15 % |
| **TT (MLP, `ε = 0.3`) + BF16 cores** | ~40 % | **~20 % 🔻** | +5 – 15 % |

### 🥇 Best Combined Result

> **TT (`ε = 0.3`, MLP layers 8 – 11) + FP16 core storage → ~5× backbone memory compression**  
> Detection quality remains comparable to the full-precision baseline; the `.predict()` API is unchanged.

> ⚠️ FP16 / BF16 memory savings apply to stored core tensors only. On CPU, cores are temporarily upcast to FP32
> during TT contraction — compute therefore always runs in FP32. See [§8 Disclaimer](#8-️-production-readiness-disclaimer).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Custom RF-DETR Package (`rfdetr_pierre`)](#3-custom-rf-detr-package-rfdetr_pierre)
4. [Tensor Decomposition Methods](#4-tensor-decomposition-methods)
   - [4.1 Tensor Train (TT)](#41-tensor-train-tt-decomposition)
   - [4.2 CP Decomposition](#42-cp-candecompparafac-decomposition)
   - [4.3 Compression Strategy for DINOv2 MLP Weights](#43-compression-strategy-for-dinov2-mlp-weights)
   - [4.4 TT + Core Quantization (FP16 / BF16)](#44-tt--core-quantization-fp16--bf16)
5. [Model Variants](#5-model-variants)
6. [Quick Start](#6-quick-start)
7. [Notebooks](#7-notebooks)
8. [⚠️ Production Readiness Disclaimer](#8-️-production-readiness-disclaimer)

---

## 1. Project Overview

This project builds a complete computer-vision pipeline for soccer broadcast video:

- **Detection**: identify players, balls, referees, and other objects in video frames
- **Segmentation**: produce pixel-accurate instance masks for each detected object
- **Model compression**: reduce the memory footprint of the DINOv2 backbone by 2–5× with minimal accuracy loss, enabling deployment on resource-constrained hardware

The core insight is that the weight matrices inside a Vision Transformer's MLP blocks have **rapidly decaying singular values** — they are effectively low-rank — making them ideal candidates for tensor decomposition.

---

## 2. Repository Structure

```
Soccer_Broadcast_LVM/
├── data/
│   ├── raw/                    # Input videos & test images (dog.jpg, football_play.mp4, …)
│   ├── processed/              # Annotated output videos, detection JSON
│   └── external/
├── models/                     # Saved model checkpoints
├── notebooks/
│   ├── tensor_decomposition_showcase.ipynb  ← Main showcase (runnable end-to-end)
│   ├── tt_decomposition.ipynb               ← TT research notebook
│   ├── dino_decomposition.ipynb             ← CP research notebook
│   ├── build_detection_stage.ipynb          ← Video detection pipeline
│   └── release-demo_1-5.ipynb               ← Demo reel
├── src/
│   ├── rfdetr_pierre/          # Custom RF-DETR package (see §3)
│   ├── soccer_ai/              # Soccer-domain utilities & config
│   └── tntorch_pierre/        # Custom tntorch fork for TT/CP/Tucker
└── tests/
```

> **Note:** The `.pt` / `.pth` weight files currently live in `notebooks/` for convenience during development. For a production setup, move them to `models/` and update the paths in `src/rfdetr_pierre/assets/model_weights.py`.

---

## 3. Custom RF-DETR Package (`rfdetr_pierre`)

`rfdetr_pierre` is a fork of Roboflow's RF-DETR, extended with:

### 3.1 Windowed DINOv2 Backbone

**File:** `src/rfdetr_pierre/models/backbone/dinov2_with_windowed_attn.py`

We replace the standard DINOv2 global self-attention in non-output layers with **local windowed attention** (`WindowedDinov2WithRegistersBackbone`). This reduces the attention complexity from $O(N^2)$ to $O(N \cdot W)$ where $W$ is the window size, enabling higher-resolution inputs without quadratic memory cost.

Key config parameters per model variant:
- `patch_size` — ViT patch size (12 or 16 pixels)
- `num_windows` — number of local windows per spatial dimension (1, 2, or 4)
- `out_feature_indexes` — which layers use global attention (used as feature pyramid outputs)

### 3.2 Tensor Decomposition (`tt_factorize` / `cp_factorize`)

Both methods are available as **one-line model transforms**:

```python
from rfdetr_pierre import RFDETRSegNano

model = RFDETRSegNano()

# Tensor Train compression
model_tt = model.tt_factorize(
    layer_ids=[8, 9, 10, 11],          # which DINOv2 layers to compress
    factorize_backbone_mlp=True,        # compress MLP blocks
    factorize_backbone_attention=True,  # compress Q/K/V/Out attention projections
    factorize_decoder_ffn=False,        # compress transformer decoder FFN
    eps_backbone_mlp=0.3,               # TT truncation tolerance (higher → smaller model)
    eps_backbone_attention=0.3,
)

# CP decomposition
model_cp = model.cp_factorize(
    cp_rank=16,
    layer_ids=[8, 9, 10, 11],
    factorize_backbone_mlp=True,
    init=("svd", "hosvd"),              # multi-start ALS initialisation
    als_random_starts=3,
)

# Inference is identical after compression
detections = model_tt.predict(image, threshold=0.35)
```

### 3.3 Segmentation Head

`RFDETRSeg*` variants attach a lightweight **mask head** (`segmentation_head.py`) on top of the standard detection head, producing binary instance masks at `1/mask_downsample_ratio` of the input resolution.

### 3.4 Model Variants

| Class | Config | Resolution | Parameters (approx.) |
|-------|--------|------------|----------------------|
| `RFDETRNano` | Nano | 384 | ~7M |
| `RFDETRSmall` | Small | 512 | ~11M |
| `RFDETRMedium` | Medium | 576 | ~16M |
| `RFDETRSegNano` | Seg-Nano | 312 | ~9M |
| `RFDETRSegSmall` | Seg-Small | 384 | ~13M |
| `RFDETRSegMedium` | Seg-Medium | 432 | ~18M |
| `RFDETRSegLarge` | Seg-Large | 504 | ~25M |

---

## 4. Tensor Decomposition Methods

### Background

A neural network weight matrix $W \in \mathbb{R}^{m \times n}$ can be viewed as a 2-way tensor. Tensor decompositions generalise matrix factorisation to higher-order structures, enabling **controlled lossy compression** with a single accuracy–compression trade-off parameter.

We target the DINOv2 backbone's **MLP blocks** which account for ~75% of backbone parameters:
- `fc1.weight` ∈ ℝ^{1536×384} and `fc2.weight` ∈ ℝ^{384×1536} per layer
- 12 layers × 2 matrices ≈ **14.2M parameters** in the backbone MLP alone

---

### 4.1 Tensor Train (TT) Decomposition

The **Tensor Train** (also called Matrix Product State) decomposes an $N$-way tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ as a chain of 3-way cores:

$$\mathcal{X}[i_1, \ldots, i_N] = G_1[i_1] \cdot G_2[i_2] \cdots G_N[i_N]$$

where $G_k \in \mathbb{R}^{r_{k-1} \times I_k \times r_k}$ and boundary ranks $r_0 = r_N = 1$.

**Parameter count:**  
Dense: $\prod_{k} I_k$  
TT: $\sum_{k} r_{k-1} \cdot I_k \cdot r_k$

The ranks $(r_1, \ldots, r_{N-1})$ are the primary compression knobs. We support two control modes:
- **Fixed ranks** via `ranks_tt=[r1, r2, …]`
- **Adaptive truncation** via `eps=ε`, which uses truncated SVD sweeps to guarantee that the TT approximation error $\leq \varepsilon \cdot \|\mathcal{X}\|_F$

**Algorithm:** TT-SVD (also called DMRG / density matrix renormalisation group) — a left-to-right sweep of successive SVD truncations.

**Implementation:** `src/tntorch_pierre/tensor.py`, `Tensor(data, eps=0.3)` or `Tensor(data, ranks_tt=[...])`

#### Joint MLP Compression

Rather than compressing $W_1$ and $W_2$ independently, we stack them into a single 3D tensor:
$$F = \text{stack}(W_1, W_2^\top) \in \mathbb{R}^{2 \times 1536 \times 384}$$

This allows the TT decomposition to discover **shared low-rank structure** between the two weight matrices, yielding higher compression at the same reconstruction quality. After decomposition, we recover $\hat{W}_1 = \hat{F}[0,:,:]$ and $\hat{W}_2 = \hat{F}[1,:,:]^\top$.

**Forward pass (TTDinov2WithRegistersMLP):** TT cores are stored compressed. During inference, the effective weight matrix is reconstructed on-the-fly via batched einsum contractions over the TT chain, with no additional storage overhead.

#### Typical results (RFDETRSegNano, layers 8–11, ε=0.3)
| Target | Compression | Latency overhead |
|--------|-------------|-----------------|
| MLP only | ~2.5× | +5–15% |
| MLP + Attention | ~3.0× | +15–30% |
| MLP + Attention + Decoder FFN | ~3.5× | +20–40% |

---

### 4.2 CP (CANDECOMP/PARAFAC) Decomposition

The **CP decomposition** expresses a tensor as a sum of $R$ rank-1 outer products:

$$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \cdot a_r^{(1)} \otimes a_r^{(2)} \otimes \cdots \otimes a_r^{(N)}$$

For a matrix $W \in \mathbb{R}^{m \times n}$, this reduces to a rank-$R$ approximation:
$$W \approx \sum_{r=1}^{R} u_r v_r^\top = U V^\top, \quad U \in \mathbb{R}^{m \times R},\ V \in \mathbb{R}^{n \times R}$$

**Parameter count:** $(m + n) \cdot R$ vs $m \cdot n$ → compression when $R < \frac{mn}{m+n}$

CP rank $R$ is the compression knob: lower rank = smaller model, higher reconstruction error.

#### Supported Solvers

| Solver | Notes |
|--------|-------|
| `tensorly_als` | Alternating Least Squares via TensorLy; supports multi-start init (HOSVD, SVD, random) |
| `scipy_nls` | SciPy TRF nonlinear least squares; robust losses (`soft_l1`, `huber`, `cauchy`) for noisy weight matrices |
| `torch_lbfgs` | PyTorch L-BFGS on CP factor matrices with robust loss and ALS warm start |

Multi-start initialization (`als_init_modes=("svd", "hosvd", "random")`) runs multiple candidates and keeps the best reconstruction, substantially increasing odds of finding the global optimum.

#### CP Variants

| Variant | Description |
|---------|-------------|
| `standard` | Replace each MLP module with a CPDinov2WithRegistersMLP storing explicit CP factors |
| `layer_joint` | Joint CP over all weight matrices in one layer (fc1, fc2, Q, K, V, Out) with dense write-back |
| `global_mlp` | One global CP over all selected fc1/fc2 matrices stacked together; dense write-back |

**Implementation:** `src/rfdetr_pierre/detr.py::cp_factorize()`, `src/tntorch_pierre/tensor.py`

---

### 4.3 Compression Strategy for DINOv2 MLP Weights

The recommended workflow for finding the best compressed model:

1. **Single-layer analysis** — sweep ε (TT) or rank (CP) on one MLP layer, plot compression ratio vs R² to identify the knee point
2. **Model-level ablation** — choose which components to compress (MLP / attention / decoder FFN) and which layers (last-4 is usually best)
3. **Automated search** (see `tensor_decomposition_showcase.ipynb`) — sweep a config grid, evaluate on a validation image, score via $\text{score} = \frac{CR}{\text{latency ratio}} \times \min(1, \text{detection ratio})$
4. **Inference** — the compressed model's `.predict()` interface is identical to the baseline

---

### 4.4 TT + Core Quantization (FP16 / BF16)

After TT compression, the individual TT core tensors can be **cast to a lower-precision dtype** (FP16 or BF16) to achieve a further **~2× reduction in stored bytes** with negligible impact on reconstruction quality.

**How it works:**
- TT cores are stored as `torch.float16` or `torch.bfloat16` tensors (2 bytes/element vs 4)
- During the forward pass, cores are temporarily upcast to `float32` for the TT contraction chain (batched einsum / matmul), then the reconstructed weight matrix is cast to the desired compute dtype
- Memory savings are real for model storage and transfer; on CPU, the einsum contraction always runs in FP32

**Cumulative memory reduction table:**

| Stage | Memory vs FP32 baseline |
|-------|--------------------------|
| FP32 baseline | 100 % |
| After TT (`ε = 0.3`, MLP layers 8 – 11) | ~40 % |
| + FP16 core storage | **~20 %** (≈ 5× total reduction) |
| + BF16 core storage | **~20 %** (≈ 5× total reduction) |

**API (see §9 of `tensor_decomposition_showcase.ipynb`):**

```python
QUANT_CONFIGS = [
    {"name": "TT-FP32", "core_dtype": torch.float32},
    {"name": "TT-FP16", "core_dtype": torch.float16},
    {"name": "TT-BF16", "core_dtype": torch.bfloat16},
]
```

The automated sweep in the showcase notebook evaluates all three configs on `dog.jpg`, selects the best by a combined score of compression ratio × detection quality, and passes the winner directly to the video segmentation section.

> **Note:** True mixed-precision TT contraction (FP16 einsum) requires structural changes to tntorch's contraction kernels or GPU execution. The upcast-on-contraction approach is a CPU research workaround — see [§8 Disclaimer](#8-️-production-readiness-disclaimer).

---

## 5. Model Variants

All variants share the same base class `RFDETR` and support both decomposition APIs.

| Method | Signature | Key Parameters |
|--------|-----------|---------------|
| `tt_factorize()` | `model.tt_factorize(layer_ids, eps_backbone_mlp, ...)` | `eps_*` controls truncation per component; `layer_ids="all"` or list |
| `cp_factorize()` | `model.cp_factorize(cp_rank, layer_ids, ...)` | `cp_rank` sets compression; `init` for ALS init strategy; `cp_variant` for decomposition mode |
| `predict()` | `model.predict(image, threshold)` | Returns `supervision.Detections` with optional `.mask` field |
| `optimize_for_inference()` | `model.optimize_for_inference()` | Fuses BN, sets eval mode |

---

## 6. Quick Start

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src")))

from PIL import Image
from rfdetr_pierre import RFDETRSegNano
from rfdetr_pierre.util.coco_classes import COCO_CLASSES

# Load model
model = RFDETRSegNano()

# (Optional) Apply TT compression to last 4 backbone layers
model = model.tt_factorize(
    layer_ids=[8, 9, 10, 11],
    factorize_backbone_mlp=True,
    eps_backbone_mlp=0.3,
    verbose=True,
)

# Run inference
image = Image.open("data/raw/dog.jpg")
detections = model.predict(image, threshold=0.35)

labels = [COCO_CLASSES[c] for c in detections.class_id]
print(labels)
```

---

## 7. Notebooks

| Notebook | Purpose |
|----------|---------|
| [`tensor_decomposition_showcase.ipynb`](notebooks/tensor_decomposition_showcase.ipynb) | **Main showcase** — fully runnable, automated config search, plots, video output |
| [`tt_decomposition.ipynb`](notebooks/tt_decomposition.ipynb) | TT research: single-layer analysis, stacked-tensor strategy, latency benchmarks |
| [`dino_decomposition.ipynb`](notebooks/dino_decomposition.ipynb) | CP research: multi-start ALS, rank sweep, CP variant comparison |
| [`build_detection_stage.ipynb`](notebooks/build_detection_stage.ipynb) | Soccer video detection pipeline: frame-by-frame inference, JSON serialisation |
| [`release-demo_1-5.ipynb`](notebooks/release-demo_1-5.ipynb) | Release demo reel |

---

## References

- **RF-DETR**: Roboflow, 2025 — [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr)

---

## 8. ⚠️ Production Readiness Disclaimer

**This is a research prototype — not production-ready code.**

The tensor decomposition and quantization capabilities demonstrated in this repo required significant customisation to upstream libraries. The following limitations must be resolved before this work can be considered production-ready.

### 8.1 Custom / Forked Packages

| Package | Based on | Modifications |
|---------|----------|---------------|
| `rfdetr_pierre` | Roboflow RF-DETR | Added `tt_factorize()`, `cp_factorize()`, `TTDinov2WithRegistersMLP`, `CPDinov2WithRegistersMLP` — not upstream |
| `tntorch_pierre` | upstream tntorch | Extended CP solver backends (TensorLy ALS, SciPy NLS, PyTorch L-BFGS); modified contraction paths |

Neither package tracks upstream. Upgrading the base libraries without re-integrating these changes will silently break the decomposition APIs.

### 8.2 Runtime Monkeypatching

FP16 / BF16 core quantization is implemented by **replacing `_dense_weights` on individual module instances at runtime** using `types.MethodType`. This is a workaround, not a structural fix:

```python
# Simplified view of the patch applied per TTDinov2WithRegistersMLP module
def _patched_dense_weights(self_m, compute_dtype, device):
    low_prec_cores = self_m.tt_tensor.cores
    self_m.tt_tensor.cores = [c.float() for c in low_prec_cores]  # upcast to FP32
    stacked = self_m.tt_tensor.torch()                             # FP32 contraction
    self_m.tt_tensor.cores = low_prec_cores                        # restore stored dtype
    stacked = stacked.to(device=device, dtype=compute_dtype)
    return stacked[0], stacked[1].T
```

**Root cause:** PyTorch's CPU kernels do not support `float16` / `bfloat16` for `torch.einsum` / `matmul`. The TT contraction chain in tntorch collapses cores via einsum, which forces a float32 upcast on every forward pass. **Memory savings are real at rest; compute always runs in FP32 on CPU.**

### 8.3 What Structural Changes Would Be Needed

| Gap | Required work |
|-----|---------------|
| True FP16 TT contraction | Rewrite tntorch contraction to accept `compute_dtype`; use CUDA FP16 kernels on GPU |
| `torch.ao.quantization` integration | Replace manual dtype casting with PyTorch native PTQ / QAT pipelines |
| ONNX / TorchScript export | `tntorch.Tensor.torch()` is not ONNX-traceable; contraction must be unrolled into explicit ops |
| Package consolidation | Merge `rfdetr_pierre` and `tntorch_pierre` changes back to upstream or maintain proper forks |

### 8.4 Summary

| Limitation | Practical impact |
|-----------|------------------|
| CPU FP16/BF16 matmul unsupported | Compute runs in FP32; only storage and transfer benefit from low precision |
| `_dense_weights` monkeypatch | Fragile; breaks if module internals change between versions |
| Non-upstream packages | Manual maintenance burden; no upstream security patches |
| No ONNX / TorchScript export | Cannot use standard serving infrastructure (TorchServe, Triton, ONNX Runtime) |
| No torch.ao.quantization | Cannot use PyTorch's native quantization toolchain or INT8 acceleration |
- **DINOv2**: Oquab et al., 2023 — *DINOv2: Learning Robust Visual Features without Supervision*
- **Tensor Train**: Oseledets, 2011 — *Tensor-Train Decomposition*, SIAM J. Sci. Comput.
- **CP/PARAFAC**: Carroll & Chang, 1970; Harshman, 1970
- **TensorLy**: Kossaifi et al., 2019 — [tensorly.org](http://tensorly.org)
- **tntorch**: Alberini & Sanz-Alonso, 2021 — [github.com/rballester/tntorch](https://github.com/rballester/tntorch) (forked as `tntorch_pierre`)

