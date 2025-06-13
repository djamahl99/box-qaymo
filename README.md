# Box-QAymo

**A box-referring VQA dataset and benchmark for evaluating vision-language models on spatial and temporal reasoning in autonomous driving scenarios.**

Box-QAymo addresses a critical gap in autonomous driving AI: the ability to understand and respond to user queries about specific objects in complex driving scenes. Rather than relying on full-scene descriptions, our dataset enables users to express intent by drawing bounding boxes around objects of interest, providing a fast and intuitive interface for focused queries.

🌐 **[Project Page](https://djamahl99.github.io/qaymo-pages/)**

## Why Box-QAymo?

Current vision-language models (VLMs) struggle with localized, user-driven queries in real-world autonomous driving scenarios. Existing datasets focus on:
- ❌ Full-scene descriptions without spatial specificity
- ❌ Waypoint prediction rather than interpretable communication
- ❌ Idealized assumptions that don't reflect real user needs

**Box-QAymo enables:**
- ✅ **Spatial reasoning** about user-specified objects via bounding boxes
- ✅ **Temporal understanding** of object motion and inter-object dynamics
- ✅ **Hierarchical evaluation** from basic perception to complex spatiotemporal reasoning
- ✅ **Real-world complexity** with crowd-sourced fine-grained annotations

## Dataset Highlights

- **202 driving scenes** from Waymo Open Dataset validation split
- **50% of objects** enhanced with crowd-sourced fine-grained semantic labels
- **Hierarchical question taxonomy** spanning 3 complexity levels:
  1. **Binary sanity checks** (movement status, orientation)
  2. **Instance-grounded questions** (fine-grained classification, color recognition)
  3. **Motion reasoning** (trajectory analysis, relative motion, path conflicts)
- **Robust quality control** through negative sampling, temporal consistency, and difficulty stratification

## Question Categories

| Category | Description | Example |
|----------|-------------|---------|
| 🔍 **VLM Sanity Check** | Binary questions testing basic scene understanding | *"Are there any stationary vehicles?"* |
| 📦 **Instance-Grounded** | Questions about specific box-referred objects | *"What type of object is in the red box?"* |
| 🏃 **Motion Reasoning** | Spatiotemporal understanding across frames | *"Are the ego vehicle and truck on a collision course?"* |

## Quick Start

### Prerequisites
- Python 3.10
- Access to Waymo Open Dataset v1.4.3

### Installation

1. **Clone and install requirements:**
   ```bash
   git clone https://github.com/your-username/box-qaymo
   cd box-qaymo
   pip install -r requirements.txt
   ```

2. **Download required data:**
   - Waymo Open Dataset v1.4.3 validation tfrecords
   - Metadata files from [Google Drive](https://drive.google.com/drive/folders/1hgEu0n3TdDilA0nc01DHFo9I1kNfRNf2?usp=sharing)

3. **Process Waymo data:**
   ```bash
   python waymo_extract.py --validation-path /path/to/waymo --output-dir /path/to/output
   ```

4. **Generate VQA dataset:**
   ```bash
   python vqa_generator.py --dataset_path /path/to/output
   ```

## Dataset Structure

Our three-stage construction methodology:

### 1. Enhanced Object Annotation
- **Base dataset**: Waymo Open Dataset (superior scene diversity and LiDAR density)
- **Crowd-sourced labeling**: Fine-grained semantic categories following Argoverse 2.0 taxonomy
- **Multi-view annotation**: 3×3 object galleries from best visibility crops
- **Color attributes**: Vehicle color labels for color-based reasoning

### 2. Box-Referring VQA Generation
- **Visual markers**: Red bounding boxes instead of numerical coordinates
- **Hierarchical complexity**: From binary questions to complex spatiotemporal reasoning
- **Motion analysis**: Both implicit (single-frame) and explicit (multi-frame) approaches

### 3. Quality Control & Balancing
- **Negative sampling**: Balanced positive/negative examples
- **Temporal consistency**: Logical consistency across frame sequences
- **Difficulty stratification**: Granular complexity levels
- **Answer formats**: 2-4 option multiple choice to prevent binary guessing

## Evaluation Framework

Our hierarchical evaluation protocol systematically tests VLM capabilities:

```
Binary Sanity Checks → Instance-Grounded Questions → Motion Reasoning
     (Basic VLM)              (Spatial Focus)           (Temporal Understanding)
```

## Model Integration

Evaluate your models using our provided scripts:

```bash
# LLaVA evaluation
python llava_predict.py --dataset_path /path/to/data

# Qwen-VL evaluation  
python qwenvl_predict.py --dataset_path /path/to/data

# Evaluate all models
python eval_all_csv.py --dataset_path /path/to/data
```

## Key Findings

Our comprehensive evaluation reveals significant limitations in current VLMs:
- **Spatial referencing**: Models struggle to correctly identify box-referred objects
- **Fine-grained classification**: Poor performance on detailed object categorization
- **Motion understanding**: Difficulty with temporal reasoning and trajectory analysis
- **Real-world gap**: Performance drops significantly under realistic conditions

## Research Applications

Box-QAymo enables research in:
- **Interpretable autonomous driving** systems
- **Spatial-aware vision-language models**
- **Human-AI interaction** in safety-critical domains
- **Temporal reasoning** in dynamic environments
- **User intent understanding** through visual references

## Citation

If you use Box-QAymo in your research, please cite:

```bibtex
@article{box_qaymo_2024,
  title={Box-QAymo: A Box-Referring VQA Dataset for Spatial and Temporal Reasoning in Autonomous Driving},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## API Reference

<details>
<summary>Core Components</summary>

### Data Processing
- `WaymoDatasetLoader`: Extracts and processes Waymo scenes
- `SceneInfo`: Complete scene representation with temporal data
- `ObjectInfo`: Enhanced object annotations with fine-grained labels

### Question Generation
- `BasePromptGenerator`: Abstract base for question generators
- `ObjectBinaryPromptGenerator`: Binary sanity check questions
- `ObjectDrawnBoxPromptGenerator`: Instance-grounded questions
- `EgoRelativeObjectTrajectoryPromptGenerator`: Motion reasoning questions

### Evaluation
- `MultipleChoiceMetric`: Accuracy, Recall, Precision, F1 evaluation for MCQ

</details>

## Contributing

We welcome contributions! Areas of particular interest:
- New question types and complexity levels
- Additional evaluation metrics
- Model integration scripts
- Analysis tools and visualizations

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Note**: This project processes data from the Waymo Open Dataset, which requires separate licensing from Waymo.

## Acknowledgments

- Waymo Open Dataset team for providing high-quality autonomous driving data
- Crowd-sourcing annotators for fine-grained semantic labels
- Argoverse team for the semantic taxonomy

---

**🚀 Ready to evaluate your VLM on real-world driving scenarios? Get started with Box-QAymo today!**