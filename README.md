# FoodExtract Vision — Fine-tuned SmolVLM2 for Structured Food & Drink Extraction

A resume-style end-to-end project that fine-tunes a small Vision Language Model (VLM) to:

- **Detect whether an image contains food/drinks**
- **Extract visible food/drink items into a strict JSON schema**

This repository is organized around three core files:

- **`app.py`**: Hugging Face Spaces **Gradio** demo entrypoint (base vs. fine-tuned comparison)
- **`vlm_fine_tune_jarvis.ipynb`**: training code (Supervised Fine-Tuning / SFT)
- **`vlm_SmolVLM2_500M_Test.ipynb`**: inference/testing notebook for the uploaded fine-tuned model

## Demo

The Gradio demo (`app.py`) runs two `transformers` pipelines:

- **Base**: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- **Fine-tuned**: `CreatorJarvis/FoodExtract-Vision-SmolVLM2-500M-fine-tune`

It uses the same prompt for both and displays the generated JSON output for quick qualitative comparison.

## Project Highlights

- **Structured extraction**: image → strict JSON (`is_food`, `image_title`, `food_items`, `drink_items`).
- **Model comparison UI**: base vs. fine-tuned outputs in one interface.
- **Production-minded inference**:
  - Automatic `cuda` / `cpu` selection.
  - Optional precision toggles (`USE_FP16`, `USE_BF16`).
  - CUDA GEMM failure fallback to CPU.
- **Reproducible workflow**: training notebook documents dataset loading, preprocessing, and SFT setup.

## Tech Stack

- **Model / Training**: PyTorch, Transformers, TRL (SFTTrainer)
- **Experiment tracking**: Weights & Biases (optional)
- **Demo**: Gradio, Hugging Face Spaces

## Repository Structure

```text
.
├── app.py                       # HF Spaces Gradio app (demo entrypoint)
├── examples/                     # Sample images for the demo
├── vlm_fine_tune_jarvis.ipynb     # Training notebook (SFT)
└── vlm_SmolVLM2_500M_Test.ipynb   # Testing notebook (inference on fine-tuned model)
```

## How to Run

### Run `app.py` locally (optional)

This is mainly intended for Hugging Face Spaces, but you can also run it locally.

```bash
pip install torch transformers gradio pillow
python app.py
```

Notes:

- If you want to use the `@spaces.GPU` decorator locally, you may also need `pip install spaces`.
- `torch` installation varies by OS/CUDA. If installation fails, install the correct build from https://pytorch.org/.

### Deploy to Hugging Face Spaces

Recommended workflow:

1. Create a new **Gradio** Space.
2. Upload/copy the following into the Space repo:
   - `app.py`
   - `examples/` (optional)
3. Configure hardware (CPU/GPU) in Space settings.

If the Space build complains about missing Python dependencies, add them in the Space repository configuration (commonly via dependency files in the Space repo). **Do not hardcode tokens/keys.**

## Environment Variables

- `FORCE_CPU=1`
  - Force CPU even if CUDA is available.
- `USE_FP16=1`
  - On CUDA, run inference with FP16.
- `USE_BF16=1`
  - On CUDA, run inference with BF16 if supported.

## Output Contract (JSON Schema)

The app prompts both models to return **only valid JSON** in the following shape:

```json
{
  "is_food": 1,
  "image_title": "Fresh Garden Salad",
  "food_items": ["lettuce", "tomatoes", "cucumber"],
  "drink_items": []
}
```

## Training Notes

Training is documented in `vlm_fine_tune_jarvis.ipynb`.

- **Dataset**: `mrdbourke/FoodExtract-1k-Vision`
  - ~1000 food images + ~500 non-food images
  - Labels generated using a larger VLM and converted into a Hugging Face image dataset
- **Method**: Supervised Fine-Tuning (SFT) using `trl.SFTTrainer`

### What this notebook does

- Loads and inspects the dataset
- Builds the message/prompt format for image → structured JSON extraction
- Runs SFT training and saves the fine-tuned checkpoint

> Authentication (Hugging Face / W&B) should be done via environment variables or interactive login. Avoid committing any secrets.

## Testing Notes

The notebook `vlm_SmolVLM2_500M_Test.ipynb` is used to:

- Load the **uploaded fine-tuned model** from Hugging Face Hub
- Run inference on sample images
- Validate output formatting and compare behavior

> Token/auth setup (Hugging Face / W&B) should be provided via environment variables. Avoid committing any secrets.

## Links

- **Base model**: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- **Fine-tuned model**: https://huggingface.co/CreatorJarvis/FoodExtract-Vision-SmolVLM2-500M-fine-tune
- **Dataset**: https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision

## Roadmap

- Reduce prompt dependence (train towards direct image → JSON behavior)
- Add lightweight evaluation (schema validity rate, extraction accuracy)
- Extend to multi-language extraction

---

If you’re reviewing this as a portfolio project:

- The demo shows **before/after fine-tuning** behavior under the same prompt.
- The notebook shows the **training pipeline** and key hyperparameters used.
