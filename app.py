import os
import torch
import gradio as gr
import spaces
from transformers import pipeline

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
FINE_TUNED_MODEL_ID = "CreatorJarvis/FoodExtract-Vision-SmolVLM2-500M-fine-tune"
OUTPUT_TOKENS = 256
original_pipeline = None
ft_pipe = None

FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
DEVICE_TYPE = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
if DEVICE_TYPE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def _get_dtype(device: str):
    if device == "cuda":
        if os.getenv("USE_FP16", "0") == "1":
            return torch.float16
        if os.getenv("USE_BF16", "0") == "1":
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(is_bf16_supported) and is_bf16_supported():
                return torch.bfloat16
        return torch.float32
    return torch.float32

def _make_pipe(model_id: str, device_type: str):
    dtype = _get_dtype(device_type)
    device_arg = 0 if device_type == "cuda" else -1
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        device=device_arg,
        torch_dtype=dtype,
    )
    model = getattr(pipe, "model", None)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.max_new_tokens = OUTPUT_TOKENS
        try:
            generation_config.max_length = None
        except Exception:
            pass
    return pipe

ACTIVE_DEVICE_TYPE = DEVICE_TYPE

def _load_pipes(device_type: str):
    global original_pipeline, ft_pipe, ACTIVE_DEVICE_TYPE
    ACTIVE_DEVICE_TYPE = device_type
    print(f"[INFO] Using device_type={ACTIVE_DEVICE_TYPE}")
    original_pipeline = _make_pipe(BASE_MODEL_ID, ACTIVE_DEVICE_TYPE)
    ft_pipe = _make_pipe(FINE_TUNED_MODEL_ID, ACTIVE_DEVICE_TYPE)

_load_pipes(DEVICE_TYPE)

def _extract_generated_text(pipe_output) -> str:
    try:
        item0 = pipe_output[0]
        if isinstance(item0, dict) and "generated_text" in item0:
            gt = item0["generated_text"]
        else:
            gt = pipe_output[0][0]["generated_text"]

        if isinstance(gt, str):
            return gt
        if isinstance(gt, list) and gt:
            last = gt[-1]
            if isinstance(last, dict) and "content" in last:
                return last["content"]
        return str(gt)
    except Exception:
        return str(pipe_output)

def create_message(input_image):
    return [{'role': 'user',
 'content': [{'type': 'image',
   'image': input_image},
  {'type': 'text',
   'text': "Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.\n\nOnly return valid JSON in the following form:\n\n```json\n{\n  'is_food': 0, # int - 0 or 1 based on whether food/drinks are present (0 = no foods visible, 1 = foods visible)\n  'image_title': '', # str - short food-related title for what foods/drinks are visible in the image, leave blank if no foods present\n  'food_items': [], # list[str] - list of visible edible food item nouns\n  'drink_items': [] # list[str] - list of visible edible drink item nouns\n}\n```\n"}]}]

@spaces.GPU
def extract_foods_from_image(input_image):
    if input_image is None:
        return "Please upload an image", "Please upload an image"

    input_image = input_image.convert("RGB")
    input_image = input_image.resize(size=(512, 512))
    input_message = create_message(input_image=input_image)

    try:
        original_pipeline_output = original_pipeline(text=[input_message])
        outputs_pretrained = _extract_generated_text(original_pipeline_output)

        ft_pipe_output = ft_pipe(text=[input_message])
        outputs_fine_tuned = _extract_generated_text(ft_pipe_output)
    except RuntimeError as e:
        msg = str(e)
        is_cuda_linear_failure = (
            "CUBLAS_STATUS_INVALID_VALUE" in msg
            or "cublasGemmEx" in msg
            or ("CUDA error" in msg and "CUBLAS" in msg)
        )
        if ACTIVE_DEVICE_TYPE == "cuda" and is_cuda_linear_failure:
            try:
                print("[WARN] CUDA GEMM failed, falling back to CPU.")
                _load_pipes("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                original_pipeline_output = original_pipeline(text=[input_message])
                outputs_pretrained = _extract_generated_text(original_pipeline_output)
                ft_pipe_output = ft_pipe(text=[input_message])
                outputs_fine_tuned = _extract_generated_text(ft_pipe_output)
            except Exception:
                raise e
        else:
            raise

    return outputs_pretrained, outputs_fine_tuned

CUSTOM_CSS = """
/* Global Theme */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header Styling */
.header-container {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
}

.header-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: white !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-subtitle {
    font-size: 1.1rem !important;
    color: rgba(255,255,255,0.9) !important;
    font-weight: 400 !important;
}

/* Card Styling */
.info-card {
    background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.info-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    transform: translateY(-2px);
}

.tech-badge {
    display: inline-block;
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
}

/* Output Section */
.output-section {
    background: #fafbfc;
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
}

.comparison-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Button Styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Links */
.resource-link {
    color: #4f46e5;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}

.resource-link:hover {
    color: #7c3aed;
    text-decoration: underline;
}

/* Footer */
.footer-section {
    text-align: center;
    padding: 1.5rem;
    margin-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
    font-size: 0.9rem;
}

/* Accordion */
.accordion-header {
    font-weight: 600 !important;
    color: #1f2937 !important;
}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(), title="FoodExtract Vision | Fine-tuned VLM Demo") as demo:
    
    # Header Section
    gr.HTML("""
    <div class="header-container">
        <h1 class="header-title">🍽️ FoodExtract Vision</h1>
        <p class="header-subtitle">Fine-tuned SmolVLM2-500M for Structured Food & Drink Extraction</p>
    </div>
    """)
    
    # Project Overview Cards
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="info-card">
                <h3 style="margin: 0 0 0.75rem 0; color: #1f2937; font-size: 1rem;">🎯 Project Objective</h3>
                <p style="margin: 0; color: #4b5563; font-size: 0.9rem; line-height: 1.6;">
                    Extract food and drink items from images in a <strong>structured JSON format</strong>. 
                    This demo compares the base model vs. fine-tuned model to showcase the improvement in output consistency.
                </p>
            </div>
            """)
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="info-card">
                <h3 style="margin: 0 0 0.75rem 0; color: #1f2937; font-size: 1rem;">🛠️ Tech Stack</h3>
                <div>
                    <span class="tech-badge">SmolVLM2-500M</span>
                    <span class="tech-badge">Transformers</span>
                    <span class="tech-badge">LoRA Fine-tuning</span>
                    <span class="tech-badge">PyTorch</span>
                    <span class="tech-badge">Gradio</span>
                    <span class="tech-badge">HF Spaces</span>
                </div>
            </div>
            """)
    
    # Resources Section
    with gr.Accordion("📚 Model & Dataset Resources", open=False):
        gr.Markdown("""
        | Resource | Link |
        |----------|------|
        | **Base Model** | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |
        | **Fine-tuned Model** | [CreatorJarvis/FoodExtract-Vision-SmolVLM2-500M-fine-tune](https://huggingface.co/CreatorJarvis/FoodExtract-Vision-SmolVLM2-500M-fine-tune) |
        | **Training Dataset** | [mrdbourke/FoodExtract-1k-Vision](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision) (1k food + 500 non-food images) |
        """)
    
    gr.Markdown("---")
    
    # Main Demo Section
    gr.HTML('<h2 style="text-align: center; color: #1f2937; margin-bottom: 1rem;">🔬 Live Demo: Model Comparison</h2>')
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📷 Upload Food Image", height=350)
            submit_btn = gr.Button("🚀 Extract Food Items", variant="primary", elem_classes=["primary-btn"])
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.HTML('<div class="comparison-header">❌ Base Model (No Fine-tuning)</div>')
                output_original = gr.Textbox(
                    lines=6, 
                    label="",
                    placeholder="Base model output will appear here...",
                    show_label=False
                )
            with gr.Group():
                gr.HTML('<div class="comparison-header">✅ Fine-tuned Model</div>')
                output_finetuned = gr.Textbox(
                    lines=6, 
                    label="",
                    placeholder="Fine-tuned model output will appear here...",
                    show_label=False
                )
    
    submit_btn.click(
        fn=extract_foods_from_image,
        inputs=[input_image],
        outputs=[output_original, output_finetuned]
    )
    
    # Examples Section
    gr.Markdown("### 📸 Example Images")
    gr.Examples(
        examples=[
            ["examples/food1.jpeg"],
            ["examples/food2.jpg"],
            ["examples/food3.jpg"],
            ["examples/food4.jpeg"]
        ],
        inputs=[input_image],
        outputs=[output_original, output_finetuned],
        fn=extract_foods_from_image,
        cache_examples=False
    )
    
    gr.Markdown("---")
    
    # Technical Details Accordion
    with gr.Accordion("📋 Input Prompt & Expected Output Format", open=False):
        gr.Markdown("""
        **Both models receive the same input prompt:**
        
        ```text
        Classify the given input image into food or not and if edible food or drink items 
        are present, extract those to a list. If no food/drink items are visible, return empty lists.

        Only return valid JSON in the following form:
        ```
        
        **Expected JSON Output Structure:**
        ```json
        {
          "is_food": 1,
          "image_title": "Fresh Garden Salad",
          "food_items": ["lettuce", "tomatoes", "cucumber"],
          "drink_items": []
        }
        ```
        """)
    
    with gr.Accordion("🔮 Future Improvements", open=False):
        gr.Markdown("""
        - **Remove Input Prompt**: Train the model for direct image → JSON conversion to reduce inference tokens
        - **Expand Training Data**: Current dataset is limited to 1.5k images; real-world data would improve generalization
        - **Fix Repetitive Generation**: Address occasional repetitive outputs (e.g., "onions", "onions", "onions")
        - **Multi-language Support**: Extend to support food extraction in multiple languages
        """)
    
    # Footer
    gr.HTML("""
    <div class="footer-section">
        <p style="margin: 0;">Built with ❤️ by <strong>Jarvis Zhang</strong> | 
        <a href="https://huggingface.co/CreatorJarvis" target="_blank" style="color: #4f46e5;">🤗 Hugging Face</a> | 
        <a href="https://github.com/CreatorJarvis" target="_blank" style="color: #4f46e5;">💻 GitHub</a>
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #9ca3af;">Fine-tuning Demo • Vision Language Model • Structured Output Generation</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False)
