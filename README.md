# AI Interactive Storytelling System

This project is a Streamlit-based AI storytelling application that generates:

- A complete children's story
- Scene-by-scene illustrations
- Quantitative and qualitative story evaluation
- Downloadable JSON, Markdown, and image ZIP files

The system uses:

- Groq LLaMA model for story generation and evaluation
- Hugging Face Inference API for image generation
- Streamlit for the user interface

---

## Features

- Generate stories by genre, age group, theme, characters, and number of scenes
- Generate illustrations for each scene
- Evaluate the generated story using AI-based scoring
- Display results in separate Streamlit tabs
- Download generated story as Markdown
- Download raw story JSON
- Download evaluation JSON
- Download all generated images as ZIP

---

## Project Structure

```text
ai_storytelling_streamlit/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── secrets.toml.example
    └── config.toml
