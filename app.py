import os
import re
import json
import base64
from io import BytesIO
from zipfile import ZipFile
import streamlit as st
from PIL import Image
from groq import Groq
from huggingface_hub import InferenceClient
import time
import torch
from transformers import CLIPProcessor, CLIPModel

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Storytelling System",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        font-size: 2.4rem;
        margin-bottom: 8px;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.95;
    }

    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 10px;
    }

    .scene-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        margin-bottom: 18px;
    }

    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="main-header">
        <h1>AI Interactive Storytelling System</h1>
        <p>AI Story Generation · Text + Image · Powered by LLaMA-3 and Hugging Face</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SECRETS
# =========================================================
def get_secret(key: str) -> str:
    """
    Read secrets from Streamlit Secrets first.
    If not found, fallback to environment variables.
    """
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None

    if not value:
        value = os.getenv(key)

    return value


GROQ_API_KEY = get_secret("GROQ_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")


if not GROQ_API_KEY or not HF_TOKEN:
    st.error(
        """
        Missing API keys.

        Please add your keys in Streamlit Secrets:

        GROQ_API_KEY = "your-groq-api-key"
        HF_TOKEN = "your-huggingface-token"
        """
    )
    st.stop()


# =========================================================
# CLIENTS
# =========================================================
@st.cache_resource
def load_groq_client(api_key: str):
    return Groq(api_key=api_key)


@st.cache_resource
def load_hf_client(api_key: str):
    return InferenceClient(
        provider="auto",
        api_key=api_key
    )


groq_client = load_groq_client(GROQ_API_KEY)
hf_client = load_hf_client(HF_TOKEN)

# =========================================================
# CLIP MODEL FOR IMAGE EVALUATION
# =========================================================
@st.cache_resource
def load_clip_model():
    """
    Load CLIP model for image-text alignment evaluation.
    Cached to avoid reloading on every interaction.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


with st.spinner("Loading CLIP model for image evaluation..."):
    clip_model, clip_processor = load_clip_model()


# =========================================================
# CONSTANTS
# =========================================================
GENRES = [
    "Fantasy",
    "Adventure",
    "Educational",
    "Fable",
    "Sci-Fi",
    "Mystery",
    "Folklore"
]

AGE_GROUPS = [
    "Toddlers (2-4)",
    "Young Children (5-7)",
    "Children (8-10)",
    "Pre-teens (11-13)"
]

THEMES = [
    "Friendship",
    "Courage",
    "Honesty",
    "Kindness",
    "Perseverance",
    "Curiosity",
    "Family"
]

LANGUAGES = ["English"]


# =========================================================
# JSON HELPERS
# =========================================================
def clean_json_output(raw_text: str) -> str:
    """
    Cleans LLM output and attempts to isolate JSON content.
    """
    raw_text = raw_text.strip()

    raw_text = re.sub(r"```json", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"```", "", raw_text)
    raw_text = raw_text.strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start != -1 and end != -1 and end > start:
        raw_text = raw_text[start:end + 1]

    return raw_text


def safe_json_loads(raw_text: str) -> dict:
    """
    Parse JSON safely from model output.
    """
    cleaned = clean_json_output(raw_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned_no_newlines = cleaned.replace("\n", " ").replace("\r", " ")
        return json.loads(cleaned_no_newlines)


# =========================================================
# TEXT GENERATION
# =========================================================
def generate_story(
    genre: str,
    age_group: str,
    theme: str,
    characters: str,
    num_scenes: int,
    language: str
) -> dict:
    """
    Generate story using Groq LLaMA model.
    """

    if "Toddlers" in age_group:
        age_style = (
            "Use very short sentences of 5-7 words. "
            "Use very simple words and repetition suitable for toddlers."
        )
    elif "Young Children" in age_group:
        age_style = (
            "Use simple sentences and clear vocabulary suitable for young children."
        )
    elif "Children" in age_group:
        age_style = (
            "Use engaging storytelling with moderate vocabulary suitable for children."
        )
    else:
        age_style = (
            "Use richer vocabulary and more complex storytelling suitable for pre-teens."
        )

    lang_instruction = "Respond entirely in English."

    system_prompt = f"""
You are a master storyteller for {age_group} audiences.

Writing Style Instructions:
{age_style}

Make sure the vocabulary and sentence complexity strictly match the selected age group.
{lang_instruction}

Your task:
Write a {genre} story split into exactly {num_scenes} scenes.

Each scene must include a highly detailed image_prompt describing:
- character appearance
- character emotions
- environment
- actions happening in the scene

Structure the story with:
1. Introduction
2. Problem or conflict
3. Resolution

Create a consistent visual description for the main character and reuse it in every image prompt.

OUTPUT FORMAT — strict JSON only, no extra text:
{{
  "title": "story title",
  "moral": "one sentence moral",
  "character_profile": "detailed visual description of the main character",
  "scenes": [
    {{
      "scene_number": 1,
      "title": "scene title",
      "narrative": "story text",
      "image_prompt": "scene description WITHOUT repeating character description"
    }}
  ]
}}
"""

    user_prompt = f"""
Genre: {genre}
Age Group: {age_group}
Theme: {theme}
Main Characters: {characters}
Number of scenes: {num_scenes}

Write the full story now.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.85,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()
    story_data = safe_json_loads(raw)

    return story_data


# =========================================================
# IMAGE GENERATION
# =========================================================
def generate_image_hf(prompt: str):
    """
    Generate image using Hugging Face InferenceClient.
    """
    try:
        image = hf_client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )
        return image.convert("RGB")

    except Exception as e:
        st.warning(f"Image generation failed: {e}")
        return None


def create_placeholder_image() -> Image.Image:
    """
    Placeholder image if HF image generation fails.
    """
    image = Image.new("RGB", (768, 512), color=(210, 210, 230))
    return image



# =========================================================
# CLIP SCORE (Image-Text Alignment Evaluation)
# =========================================================
def compute_clip_score(image, text_prompt: str):
    """
    Computes CLIP score: cosine similarity between image and text embeddings.
    Returns a score from 0 to 1 (higher = better alignment).
    """
    try:
        inputs = clip_processor(
            text=[text_prompt],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize and compute cosine similarity
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_embeds @ text_embeds.T).item()

        return round(similarity, 4)

    except Exception as e:
        st.warning(f"CLIP scoring failed: {e}")
        return None


# =========================================================
# EVALUATION
# =========================================================
def evaluate_story(story_data: dict, language: str) -> dict:
    """
    Evaluates story using a DIFFERENT model than the generator
    to avoid self-evaluation bias (LLM-as-judge best practice).
    
    Generator: llama-3.3-70b-versatile
    Evaluator: openai/gpt-oss-120b
    """

    lang_instruction = "Respond in English."

    story_text = "\n\n".join(
        [
            f"Scene {scene['scene_number']}: {scene['narrative']}"
            for scene in story_data["scenes"]
        ]
    )

    eval_prompt = f"""
{lang_instruction}

Evaluate this AI-generated story on these criteria.

Return JSON only:
{{
  "coherence_score": <1-10>,
  "age_appropriateness": <1-10>,
  "creativity_score": <1-10>,
  "language_quality": <1-10>,
  "overall_score": <1-10>,
  "strengths": ["point1", "point2", "point3"],
  "limitations": ["limit1", "limit2"],
  "improvement_suggestions": ["sug1", "sug2"]
}}

Story Title: {story_data["title"]}

Story Text:
{story_text}
"""


    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    eval_data = safe_json_loads(raw)

    return eval_data


# =========================================================
# FORMATTING HELPERS
# =========================================================
def build_story_markdown(story_data: dict) -> str:
    """
    Convert story JSON into Markdown.
    """

    story_md = f"# {story_data.get('title', 'Generated Story')}\n\n"
    story_md += f"> **Moral / Value:** {story_data.get('moral', '')}\n\n"
    story_md += "---\n\n"

    for scene in story_data.get("scenes", []):
        story_md += (
            f"## Scene {scene.get('scene_number')}: {scene.get('title')}\n\n"
            f"{scene.get('narrative')}\n\n"
            "---\n\n"
        )

    return story_md


def image_to_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    """
    Convert PIL image to bytes.
    """
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return buffer.getvalue()


def create_images_zip(images_with_captions: list) -> bytes:
    """
    Create ZIP file containing generated images.
    """
    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w") as zip_file:
        for idx, item in enumerate(images_with_captions, start=1):
            image = item["image"]
            caption = item["caption"]

            safe_caption = re.sub(r"[^a-zA-Z0-9_-]+", "_", caption)
            filename = f"{idx:02d}_{safe_caption}.png"

            img_bytes = image_to_bytes(image)
            zip_file.writestr(filename, img_bytes)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def render_score_bar(score):
    """
    Render score using simple text bar.
    """
    try:
        score_int = int(score)
    except Exception:
        return "N/A"

    score_int = max(0, min(10, score_int))
    return "█" * score_int + "░" * (10 - score_int)


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_storytelling_pipeline(
    genre: str,
    age_group: str,
    theme: str,
    characters: str,
    num_scenes: int,
    language: str
):
    """
    Full pipeline:
    1. Story generation (with latency tracking)
    2. Image generation (with latency tracking + CLIP scoring)
    3. Evaluation (using different LLM)
    """

    progress_bar = st.progress(0)
    status_box = st.empty()

    # ── Step 1: Generate story (with timing) ──
    status_box.info("Generating story with LLaMA-3...")
    progress_bar.progress(10)

    story_start = time.time()
    story_data = generate_story(
        genre=genre,
        age_group=age_group,
        theme=theme,
        characters=characters,
        num_scenes=int(num_scenes),
        language=language
    )
    story_gen_time = round(time.time() - story_start, 2)

    progress_bar.progress(20)

    # ── Step 2: Generate images + CLIP scoring (with timing) ──
    images_with_captions = []
    clip_scores = []
    image_gen_times = []
    scenes = story_data.get("scenes", [])
    total_scenes = len(scenes)

    for idx, scene in enumerate(scenes, start=1):
        current_progress = 20 + int(55 * idx / max(total_scenes, 1))

        status_box.info(f"Generating image {idx}/{total_scenes}...")
        progress_bar.progress(current_progress)

        character_desc = story_data.get("character_profile", "")
        image_prompt = scene.get("image_prompt", "")

        full_prompt = (
            f"{character_desc}, {image_prompt}, "
            f"children book illustration, beautiful, colorful, soft lighting"
        )

        # Measure image generation time
        img_start = time.time()
        image = generate_image_hf(full_prompt)
        img_end = time.time()
        image_gen_times.append(round(img_end - img_start, 2))

        if image is None:
            image = create_placeholder_image()
            clip_score = None
            caption = f"Scene {scene.get('scene_number')}: {scene.get('title')} (image unavailable)"
        else:
            # Compute CLIP score
            clip_score = compute_clip_score(image, image_prompt)
            score_text = f"CLIP: {clip_score}" if clip_score is not None else "CLIP: N/A"
            caption = f"Scene {scene.get('scene_number')}: {scene.get('title')} | {score_text}"

        clip_scores.append(clip_score)

        images_with_captions.append(
            {
                "image": image,
                "caption": caption,
                "prompt": full_prompt,
                "clip_score": clip_score
            }
        )

    # ── Step 3: Evaluation ──
    status_box.info("Evaluating story quality...")
    progress_bar.progress(85)

    eval_data = evaluate_story(story_data, language)

    progress_bar.progress(100)
    status_box.success("Done!")

    # Performance metrics
    performance_data = {
        "story_gen_time": story_gen_time,
        "image_gen_times": image_gen_times,
        "total_image_time": round(sum(image_gen_times), 2),
        "total_pipeline_time": round(story_gen_time + sum(image_gen_times), 2),
        "clip_scores": clip_scores
    }

    return story_data, images_with_captions, eval_data, performance_data

# =========================================================
# SESSION STATE
# =========================================================
if "story_data" not in st.session_state:
    st.session_state.story_data = None

if "images_with_captions" not in st.session_state:
    st.session_state.images_with_captions = None

if "eval_data" not in st.session_state:
    st.session_state.eval_data = None

if "performance_data" not in st.session_state:
    st.session_state.performance_data = None


# =========================================================
# SIDEBAR INPUTS
# =========================================================
with st.sidebar:
    st.header("Story Parameters")

    genre = st.selectbox(
        "Genre",
        GENRES,
        index=GENRES.index("Fantasy")
    )

    age_group = st.selectbox(
        "Age Group",
        AGE_GROUPS,
        index=AGE_GROUPS.index("Children (8-10)")
    )

    theme = st.selectbox(
        "Theme",
        THEMES,
        index=THEMES.index("Friendship")
    )

    language = st.selectbox(
        "Language",
        LANGUAGES,
        index=0
    )

    characters = st.text_area(
        "Main Characters",
        value="Luna the brave girl, Pip the talking fox",
        placeholder="e.g. Luna the brave girl, Pip the talking fox",
        height=90
    )

    num_scenes = st.slider(
        "Number of Scenes",
        min_value=2,
        max_value=5,
        value=3,
        step=1
    )

    generate_button = st.button(
        "Generate Story",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Quick Examples")

    example_1 = st.button("Fantasy · Courage", use_container_width=True)
    example_2 = st.button("Fable · Honesty", use_container_width=True)
    example_3 = st.button("Educational · Curiosity", use_container_width=True)


# =========================================================
# QUICK EXAMPLES LOGIC
# =========================================================
if example_1:
    genre = "Fantasy"
    age_group = "Children (8-10)"
    theme = "Courage"
    characters = "Aria the dragon, Ben the knight"
    num_scenes = 3
    language = "English"
    generate_button = True

if example_2:
    genre = "Fable"
    age_group = "Young Children (5-7)"
    theme = "Honesty"
    characters = "A clever fox, a wise owl"
    num_scenes = 2
    language = "English"
    generate_button = True

if example_3:
    genre = "Educational"
    age_group = "Pre-teens (11-13)"
    theme = "Curiosity"
    characters = "Zara the scientist, Robot R7"
    num_scenes = 4
    language = "English"
    generate_button = True


# =========================================================
# RUN BUTTON
# =========================================================
if generate_button:
    if not characters.strip():
        st.error("Please enter at least one main character.")
    else:
        try:
            story_data, images_with_captions, eval_data, performance_data = run_storytelling_pipeline(
                genre=genre,
                age_group=age_group,
                theme=theme,
                characters=characters,
                num_scenes=num_scenes,
                language=language
            )

            st.session_state.story_data = story_data
            st.session_state.images_with_captions = images_with_captions
            st.session_state.eval_data = eval_data
            st.session_state.performance_data = performance_data

        except Exception as e:
            st.error(f"Pipeline failed: {e}")


# =========================================================
# OUTPUT TABS
# =========================================================
story_data = st.session_state.story_data
images_with_captions = st.session_state.images_with_captions
eval_data = st.session_state.eval_data
performance_data = st.session_state.performance_data

tab_story, tab_images, tab_eval, tab_json, tab_downloads = st.tabs(
    [
        "Story",
        "Illustrations",
        "Evaluation",
        "Raw JSON",
        "Downloads"
    ]
)


# =========================================================
# STORY TAB
# =========================================================
with tab_story:
    if story_data is None:
        st.info("Your story will appear here after generation.")
    else:
        
        st.markdown(f"# {story_data.get('title', 'Generated Story')}")
        st.markdown(f"> **Moral / Value:** {story_data.get('moral', '')}")
        st.markdown("---")

        
        scenes = story_data.get("scenes", [])
        
        for idx, scene in enumerate(scenes):
            st.markdown(
                f"## Scene {scene.get('scene_number')}: {scene.get('title')}"
            )
            
            col_text, col_image = st.columns([1, 1])
            
            with col_text:
                st.markdown(scene.get("narrative", ""))
            
            with col_image:
                if images_with_captions and idx < len(images_with_captions):
                    item = images_with_captions[idx]
                    st.image(
                        item["image"],
                        caption=item["caption"],
                        use_container_width=True
                    )
                    
                    with st.expander("View image prompt"):
                        st.write(item["prompt"])
                else:
                    st.info("Image not available")
            
            st.markdown("---")


# =========================================================
# IMAGES TAB
# =========================================================
with tab_images:
    if images_with_captions is None:
        st.info("Scene illustrations will appear here after generation.")
    else:
        cols = st.columns(2)

        for idx, item in enumerate(images_with_captions):
            with cols[idx % 2]:
                st.image(
                    item["image"],
                    caption=item["caption"],
                    use_container_width=True
                )

                with st.expander("View image prompt"):
                    st.write(item["prompt"])


# =========================================================
# EVALUATION TAB
# =========================================================
with tab_eval:
    if eval_data is None:
        st.info("Evaluation results will appear here after generation.")
    else:
        # ─── 4A: Text Quality (LLM-as-Judge) ───
        st.subheader("Text Quality Evaluation (LLM-as-Judge)")
        st.caption(
            "*Evaluator model: `openai/gpt-oss-120b` "
            "(different from generator to reduce self-bias)*"
        )

        metrics = {
            "Coherence": eval_data.get("coherence_score"),
            "Age Appropriateness": eval_data.get("age_appropriateness"),
            "Creativity": eval_data.get("creativity_score"),
            "Language Quality": eval_data.get("language_quality"),
            "Overall": eval_data.get("overall_score"),
        }

        metric_cols = st.columns(len(metrics))
        for col, (metric_name, score) in zip(metric_cols, metrics.items()):
            with col:
                st.metric(metric_name, f"{score}/10")

        st.markdown("### Score Table")
        for metric_name, score in metrics.items():
            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>{metric_name}</strong><br>
                    {score}/10 &nbsp; {render_score_bar(score)}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ─── 4B: Image Quality (CLIP Score) ───
        st.markdown("---")
        st.subheader("Image Quality Evaluation (CLIP Score)")
        st.caption(
            "*CLIP measures image-text alignment "
            "(0=poor, 1=perfect match)*"
        )

        if images_with_captions:
            clip_scores_list = [item.get("clip_score") for item in images_with_captions]
            valid_clip_scores = [s for s in clip_scores_list if s is not None]

            clip_table_md = "| Scene | CLIP Score | Quality |\n|-------|-----------|--------|\n"
            for i, score in enumerate(clip_scores_list):
                if score is not None:
                    if score >= 0.30:
                        quality = "Excellent"
                    elif score >= 0.25:
                        quality = "Good"
                    elif score >= 0.20:
                        quality = "Fair"
                    else:
                        quality = "Poor"
                    clip_table_md += f"| Scene {i+1} | {score} | {quality} |\n"
                else:
                    clip_table_md += f"| Scene {i+1} | N/A | - |\n"

            st.markdown(clip_table_md)

            if valid_clip_scores:
                avg_clip = round(sum(valid_clip_scores) / len(valid_clip_scores), 4)
                st.markdown(f"**Average CLIP Score:** {avg_clip}")

        # ─── 4C: Performance Metrics ───
        st.markdown("---")
        st.subheader("Performance Metrics (Latency)")

        if performance_data:
            perf_md = "| Stage | Time (seconds) |\n|-------|----------------|\n"
            perf_md += f"| Story Generation | {performance_data['story_gen_time']} s |\n"
            for i, t in enumerate(performance_data['image_gen_times']):
                perf_md += f"| Image {i+1} Generation | {t} s |\n"
            perf_md += f"| **Total Image Time** | **{performance_data['total_image_time']} s** |\n"
            perf_md += f"| **Total Pipeline Time** | **{performance_data['total_pipeline_time']} s** |\n"
            st.markdown(perf_md)

        # ─── 4D: Qualitative Evaluation ───
        st.markdown("---")
        st.subheader("Qualitative Evaluation")

        st.markdown("### Strengths")
        for item in eval_data.get("strengths", []):
            st.write(f"- {item}")

        st.markdown("### Limitations")
        for item in eval_data.get("limitations", []):
            st.write(f"- {item}")

        st.markdown("### Improvement Suggestions")
        for item in eval_data.get("improvement_suggestions", []):
            st.write(f"- {item}")

        # ─── 4E: System Limitations (Full version) ───
        st.markdown("---")
        st.markdown(
            """
            ## System-Level Limitations & Discussion

            ### Text Generation Limitations
            - **Hallucination:** LLMs may generate factually inconsistent plot elements
            - **Age Calibration:** Age-appropriateness relies on prompt adherence, not verified filtering
            - **Cultural Bias:** Training data may bias narrative styles toward Western storytelling

            ### Image Generation Limitations
            - **Visual Inconsistency:** Characters may appear differently across scenes (no cross-scene memory in diffusion models)
            - **Prompt Adherence:** FLUX.1-schnell prioritizes speed over fidelity; complex prompts may be partially rendered
            - **Computational Cost:** HF inference can take 20-60 sec/image on free-tier APIs

            ### Evaluation Limitations
            - **LLM-as-Judge Bias:** Although we use a different model for evaluation, LLM judges still share systematic biases (verbosity preference, position bias)
            - **CLIP Score Limitations:** CLIP was trained on internet images; it may underestimate quality for stylized children's illustrations
            - **No Human Evaluation:** Production deployment would require human raters for ground-truth quality assessment

            ### Mitigations Applied
            - Used different evaluator model (gpt-oss-120b) vs generator (llama-3.3-70b)
            - Added objective CLIP score for image-text alignment
            - Tracked latency for reproducibility
            - Character profile reused across scenes to improve consistency
            """
        )


# =========================================================
# RAW JSON TAB
# =========================================================
with tab_json:
    if story_data is None:
        st.info("Raw JSON will appear here after generation.")
    else:
        st.subheader("Story Data JSON")
        st.json(story_data)

        st.subheader("Evaluation JSON")
        st.json(eval_data)


# =========================================================
# DOWNLOADS TAB
# =========================================================
with tab_downloads:
    if story_data is None:
        st.info("Download files will appear here after generation.")
    else:
        story_md = build_story_markdown(story_data)
        story_json = json.dumps(story_data, indent=2, ensure_ascii=False)
        eval_json = json.dumps(eval_data, indent=2, ensure_ascii=False)

        st.download_button(
            label="Download Story Markdown",
            data=story_md,
            file_name="generated_story.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.download_button(
            label="Download Story JSON",
            data=story_json,
            file_name="story_data.json",
            mime="application/json",
            use_container_width=True
        )

        st.download_button(
            label="Download Evaluation JSON",
            data=eval_json,
            file_name="evaluation_data.json",
            mime="application/json",
            use_container_width=True
        )

        if images_with_captions:
            images_zip = create_images_zip(images_with_captions)

            st.download_button(
                label="Download All Images ZIP",
                data=images_zip,
                file_name="story_images.zip",
                mime="application/zip",
                use_container_width=True
            )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "AI Interactive Storytelling System · Built with Streamlit, Groq, and Hugging Face"
)
