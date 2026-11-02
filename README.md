# Cross-Dataset 2D Object Detection Evaluation

A comparative evaluation of three pretrained 2D object detectors on **COCO 2017 validation** and **nuImages validation**, with an emphasis on detection accuracy, cross-domain generalization, inference efficiency, and failure modes. This is a **six-class evaluation**, not a comparison of the models' official 80-class COCO benchmark scores.

## Project overview

The study compares three detector families without additional training or fine-tuning:

| Detector | Family | Pretrained checkpoint |
|---|---|---|
| YOLO11s | YOLO | `yolo11s.pt` |
| RT-DETR-L | RT-DETR | `rtdetr-l.pt` |
| Faster R-CNN ResNet50-FPN v2 | Faster R-CNN | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` |

**Evaluation datasets:** COCO 2017 validation (5,000 images) and nuImages validation (16,445 keyframes). Both datasets were converted to a shared taxonomy of **person, car, truck, bus, bicycle, motorcycle**, with IDs 1–6. The notebook documents the COCO category remapping and nuImages category mapping.

## Key results

The following are six-class **COCO-style AP** results computed against the project-specific mapped ground truth. Values are shown as proportions (e.g., 0.466 = 46.6%).

| Dataset | Detector | mAP@[.50:.95] | AP50 | AP75 | AP small | AP medium | AP large | AR@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COCO | YOLO11s | 0.466 | 0.653 | 0.500 | 0.218 | 0.493 | 0.708 | 0.598 |
| COCO | RT-DETR-L | **0.537** | 0.749 | 0.570 | 0.309 | 0.566 | 0.767 | 0.682 |
| COCO | Faster R-CNN | 0.519 | 0.765 | 0.558 | 0.347 | 0.538 | 0.692 | 0.638 |
| nuImages | YOLO11s | 0.225 | 0.365 | 0.215 | 0.021 | 0.141 | 0.467 | 0.346 |
| nuImages | RT-DETR-L | 0.288 | 0.477 | 0.272 | 0.060 | 0.217 | 0.514 | 0.460 |
| nuImages | Faster R-CNN | **0.319** | 0.551 | 0.303 | 0.109 | 0.265 | 0.504 | 0.454 |

**Observed cross-dataset mAP changes (COCO → nuImages):** YOLO11s −51.8%; RT-DETR-L −46.4%; Faster R-CNN −38.5%. These differences should **not** be interpreted as a controlled measurement of domain shift alone: annotation policies, scene composition, and dataset distributions differ.

### Inference efficiency

| Dataset / GPU | Detector | Median latency (ms/image) | Approx. FPS* | Peak allocated GPU memory (MB)** |
|---|---|---:|---:|---:|
| COCO / NVIDIA A100-SXM4-80GB | YOLO11s | 14.125 | 70.8 | 111.6 |
| COCO / NVIDIA A100-SXM4-80GB | RT-DETR-L | 45.126 | 22.2 | 243.4 |
| COCO / NVIDIA A100-SXM4-80GB | Faster R-CNN | 25.765 | 38.8 | 695.8 |
| nuImages / NVIDIA RTX PRO 6000 Blackwell Server Edition | YOLO11s | 5.688 | 175.8 | 146.6 |
| nuImages / NVIDIA RTX PRO 6000 Blackwell Server Edition | RT-DETR-L | 16.329 | 61.2 | 285.1 |
| nuImages / NVIDIA RTX PRO 6000 Blackwell Server Edition | Faster R-CNN | 23.133 | 43.2 | 1602.8 |

\* FPS is calculated as `1000 / median_latency_ms`, not measured end-to-end streaming throughput.  
\** Peak **PyTorch-allocated** GPU memory, not total GPU utilization or process memory.

**Hardware caveat:** COCO and nuImages were evaluated on **different GPUs** (and slightly different Python/Ultralytics versions). Compare model speed primarily **within** each dataset/hardware environment; cross-dataset latency differences are not a controlled hardware comparison.

## Repository contents

```text
.
├── 2D Object Detection Evaluation.ipynb   # Main Google Colab notebook
├── README.md
├── configs/                              # Six model × dataset run configurations
├── results/                              # Metric CSVs, domain-shift tables, plots
└── failure_analysis/                     # Sample matching statistics, configuration, figures
```

The presentation PDF is submitted **separately** and is intentionally not stored in this repository.

### Results and failure-analysis files

- `results/full_combined_summary.csv` and `results/full_combined_per_class_ap.csv`: combined cross-dataset metrics.
- `results/full_summary.csv`, `results/full_per_class_ap.csv`: COCO results.
- `results/full_nuimages_summary.csv`, `results/full_nuimages_per_class_ap.csv`: nuImages results.
- `results/full_domain_shift_map.csv`, `results/full_domain_shift_by_class.csv`: accuracy comparisons.
- `results/full_cross_dataset_map.png`, `full_coco_per_class_ap.png`, `full_nuimages_per_class_ap.png`, `full_speed_accuracy_tradeoff.png`, `full_gpu_memory_comparison.png`: analysis plots.
- `failure_analysis/sample_failure_counts_by_model.csv`, `sample_failure_counts_by_class.csv`: sample-based detection matching statistics.
- `failure_analysis/failure_analysis_config.json`: failure-analysis thresholds and sampling settings.
- `failure_analysis/coco_failure_example_{1,2,3}.png` and `nuimages_failure_example_{1,2,3}.png`: ground-truth/model prediction comparisons.

## Methodology

1. **Taxonomy alignment:** Retain the six shared classes and map COCO and nuImages annotations into a common COCO-format representation. nuImages uses annotated keyframes and explicit category mapping.
2. **Inference:** Evaluate COCO-pretrained checkpoints without fine-tuning. Ultralytics models use 640-pixel inputs; Faster R-CNN uses its checkpoint-native resize (minimum 800, maximum 1333).
3. **Official-style evaluation:** Use `pycocotools.COCOeval` for AP@[.50:.95], AP50, AP75, AP by object area, average recall, and per-class AP, with up to 100 detections per image.
4. **Efficiency:** Record inference latency and peak allocated GPU memory; derive FPS from median latency.
5. **Failure analysis:** On a deterministic sample of 250 images per dataset (seed 42), perform class-aware greedy matching at IoU ≥ 0.50 using an **analysis-only** confidence threshold of 0.25. Report TP, unmatched predictions, and missed annotations; create side-by-side visual comparisons.

**Important:** Failure-analysis TP/FP/FN counts are **descriptive custom matching results**, not COCOeval AP statistics. An unmatched prediction is not necessarily a genuinely incorrect detection, especially when datasets have differing annotation or ignore policies.

## Experimental configuration

| Setting | Value |
|---|---|
| Seed | `42` |
| Batch size | `1` |
| Inference confidence threshold | `0.001` |
| Maximum detections | `100` |
| YOLO NMS IoU | `0.7` |
| RT-DETR NMS | Not applicable |
| Faster R-CNN NMS | Torchvision checkpoint defaults |
| COCO GPU | NVIDIA A100-SXM4-80GB |
| nuImages GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| PyTorch | `2.11.0+cu130` (both runs) |
| Torchvision | `0.26.0+cu130` (both runs) |
| Ultralytics | COCO `8.4.183`; nuImages `8.4.189` |
| Python | COCO `3.13.15`; nuImages `3.13.16` |

For model-specific settings, paths, and recorded run environments, see the six JSON files in [`configs/`](configs/).

## Running the notebook (Google Colab)

**Prerequisites:** Google Colab with a CUDA GPU; Google Drive mounted; access to COCO 2017 validation images/annotations and nuImages validation metadata/images; enough storage for the datasets and generated predictions.

1. Open [`2D Object Detection Evaluation.ipynb`](2D%20Object%20Detection%20Evaluation.ipynb) in Google Colab.
2. Install dependencies using the notebook's setup cell. Core dependencies include `torch`, `torchvision`, `ultralytics`, `pycocotools`, `numpy`, `pandas`, `matplotlib`, `Pillow`, and `tqdm`. The notebook's pip cell does **not** fully pin the historical run environment; consult `configs/` for recorded versions.
3. Adjust the notebook's `ROOT`, `COCO_DIR`, and `NUIMAGES_ROOT` paths to your Drive/dataset locations. The original run used:

   ```python
   ROOT = Path('/content/drive/MyDrive/ColabNotebooks/detection_benchmark')
   COCO_DIR = Path('/content/datasets/coco')
   NUIMAGES_ROOT = Path('/content/datasets/nuimages')
   ```

4. Obtain COCO 2017 validation images and `instances_val2017.json` from the [official COCO dataset](https://cocodataset.org/#download). Obtain nuImages validation metadata and images from the [official nuImages dataset](https://www.nuscenes.org/nuimages). The notebook's extraction code expects nuImages archives at its configured Drive location.
5. Run the setup, taxonomy, and dataset-conversion cells as needed. Set `PILOT = False` for full evaluation.
6. **Inference caution:** The notebook currently sets `RUN_INFERENCE = False` to avoid expensive reruns. Set it to `True` only when intentionally running inference. **The included Section 5 inference loop currently handles nuImages only**; COCO metrics were generated in an earlier run and are reused from saved outputs. Reproducing both datasets from scratch requires adapting that inference loop to include COCO.
7. **Saved-output caution:** Sections 6 and 7 load prediction JSONs and ground-truth files from `ROOT` on Google Drive. The repository's CSVs and figures let you inspect the reported results, but **they are not a substitute for the raw prediction JSONs** if you wish to rerun evaluation or failure analysis. The image-restoration helper cells additionally rely on missing-image lists produced by a prior analysis run.

The notebook can be reviewed without rerunning inference. To regenerate all reported results independently, obtain the datasets and checkpoints, configure paths, and regenerate or supply the saved ground-truth and prediction files.

## Main findings and limitations

- **Accuracy ranking changed across datasets:** RT-DETR-L achieved the highest six-class mAP on COCO, while Faster R-CNN led on nuImages.
- **YOLO11s offered the lowest median inference latency** in both measured environments, at the cost of lower mAP.
- **Small-object detection remained challenging**, especially on nuImages.
- **Qualitative failures included occlusion and crowded scenes.** Some apparent false positives may correspond to visible but unannotated objects.
- **Comparability limitations:** Differences in image preprocessing, annotation policies, object distributions, hardware, and package versions limit causal claims about model architecture or domain shift. Results are based on pretrained checkpoints with no target-domain adaptation.

## References and resources

- [COCO: Common Objects in Context (Lin et al., 2014)](https://arxiv.org/abs/1405.0312)
- [nuScenes: A Multimodal Dataset for Autonomous Driving (Caesar et al., 2020)](https://arxiv.org/abs/1903.11027) — related dataset ecosystem; see official nuImages documentation for the dataset used here.
- [Faster R-CNN (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)
- [Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144)
- [RT-DETR (Zhao et al.)](https://arxiv.org/abs/2304.08069)
- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Torchvision Faster R-CNN documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)
- [COCO API / COCOeval](https://github.com/cocodataset/cocoapi)

## Attribution

Course project: **Track A — Experimental Object Detection Evaluation**. All evaluation metrics reported here are specific to the six-class dataset mapping and experimental configurations documented in this repository.
