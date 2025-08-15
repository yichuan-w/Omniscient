---
title: Omniscient
emoji: 👁️‍🗨️
colorFrom: indigo
colorTo: purple
sdk: streamlit
python_version: 3.11
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

<div align="center">

# 🧠 Omniscient 
### *"The all-knowing AI that sees everything, knows everything"*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow.svg)](https://huggingface.co/spaces/Omniscient001/Omniscient)

*A versatile AI bot for image analysis and dataset curation with support for multiple AI models*

🎮 **[Try it Live on HuggingFace!](https://huggingface.co/spaces/Omniscient001/Omniscient)** *(Actively WIP)*

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🗃️ **Dataset Curation**
Generate and curate high-quality image datasets with intelligent filtering and categorization.

### 🔍 **Image Analysis with multiple angle** 
Benchmark different AI models on individual images with detailed performance metrics.

</td>
<td width="50%">

### 🤖 **Agentic Analysis**
Multi-step AI reasoning and analysis with advanced decision-making capabilities.

### 🌐 **Multiple AI Providers**
Seamless integration with OpenAI, Anthropic, and Google AI platforms.

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📋 **Step 1: Setup Environment**

```bash
cd simple_G_ai_bot
```

Create a `.env` file in the project root:

```bash
# 🔐 .env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 📦 **Step 2: Install Dependencies**

```bash
uv sync
```

### 🎯 **Step 3: Usage Examples**

<details>
<summary><b>🏗️ Data Collection (Collect Mode)</b></summary>

Generate 50 samples into a named dataset (thumbnails + labels):
```bash
python main.py --mode collect --dataset my_dataset --samples 50 --headless
```

</details>

<details>
<summary><b>⚡ Single Image Analysis (Benchmark Mode)</b></summary>

Benchmark GPT-4o on 5 samples:
```bash
python main.py --mode benchmark --models gpt-4o --samples 5
```

</details>

<details>
<summary><b>🧠 Agentic Analysis (Agent Mode)</b></summary>

Run multi-step analysis with Gemini:
```bash
python main.py --mode agent --model gemini-2.5-pro --steps 10 --samples 5
```

</details>

<details>
<summary><b>🔬 Multi-Run, Per-Step Evaluation (Test Mode)</b></summary>

Run per-step evaluation across models with multiple runs and logging:
```bash
python main.py --mode test --models gpt-4o claude-3-7-sonnet --dataset test --steps 10 --samples 30 --runs 5
```

Outputs per-step accuracy and average distance, and saves logs under `results/test/<timestamp>/<model>/`.

</details>

---

## ⚙️ Configuration

### 🔑 **Environment Variables**

| Variable | Description | Status |
|:---------|:------------|:------:|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | 🔶 Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | 🔶 Optional |
| `GOOGLE_API_KEY` | Google AI API key for Gemini models | 🔶 Optional |

### 🛠️ **Command Line Options**

#### 🌟 **Common Options**
- `--mode` → Operation mode (`agent`, `benchmark`, `collect`, `test`)
- `--dataset` → Dataset name to use or create *(default: `default`)*
- `--samples` → Number of samples to process *(default: 50)*
- `--headless` → Run browser in headless mode

#### 📊 **Benchmark Mode**
- `--models` → One or more models, e.g. `--models gpt-4o claude-3-7-sonnet`
- `--temperature` → LLM sampling temperature *(default: 0.0)*

#### 🤖 **Agent Mode**
- `--model` → Single model, e.g. `--model gemini-2.5-pro`
- `--steps` → Max reasoning steps *(default: 10)*
- `--temperature` → LLM sampling temperature *(default: 0.0)*

#### 🔬 **Test Mode**
- `--models` → One or more models to compare
- `--steps` → Max steps per sample *(records per-step metrics)*
- `--runs` → Repeats per model to stabilize metrics *(default: 3)*
- `--temperature` → LLM sampling temperature *(default: 0.0)*
- `--id` → Run only the sample with this specific ID *(e.g., `--id 09ce31a1-a719-4ed9-a344-7987214902c1`)*

---

## 🎯 Supported Models

<div align="center">

| Provider | Models | Status |
|:--------:|:-------|:------:|
| **🔵 OpenAI** | GPT-4o, GPT-4, GPT-3.5-turbo | ✅ Active |
| **🟣 Anthropic** | Claude-3-opus, Claude-3-sonnet, Claude-3-haiku | ✅ Active |
| **🔴 Google** | Gemini-2.5-pro, Gemini-pro, Gemini-pro-vision | ✅ Active |

</div>

---

## 📋 Requirements

> **Prerequisites:**
> - 🐍 Python 3.8+
> - 📦 UV package manager  
> - 🔑 Valid API keys for desired AI providers

---

## 🔧 Installation

<table>
<tr>
<td>

**1️⃣** Clone the repository
```bash
git clone <repository-url>
```

**2️⃣** Navigate to project directory
```bash
cd simple_G_ai_bot
```

</td>
<td>

**3️⃣** Create `.env` file with your API keys
```bash
touch .env
# Add your API keys
```

**4️⃣** Install dependencies
```bash
uv sync
```

</td>
</tr>
</table>

**5️⃣** Run the bot with desired mode and options! 🎉

---

## 💡 Examples

### 🏗️ **Basic Data Collection**
```bash
python main.py --mode collect --dataset my_dataset --samples 20 --headless
```

### ⚔️ **Model Comparison (Benchmark)**
```bash
# GPT-4o Analysis
python main.py --mode benchmark --models gpt-4o --samples 10

# Claude-3 Analysis  
python main.py --mode benchmark --models claude-3-opus --samples 10
```

### 🧠 **Advanced Agentic Workflow**
```bash
python main.py --mode agent --model gemini-2.5-pro --steps 15 --samples 3
```

### 🔬 **Per-Step Curves and Logs (Test Mode)**
```bash
python main.py --mode test --models gpt-4o gemini-2.5-pro --dataset test --steps 10 --samples 30 --runs 5
```

This saves JSON logs per model in `results/test/<timestamp>/<model>/` and prints per-step accuracy and average distance.

**Single Sample Testing**: To test a specific sample by ID:
```bash
python main.py --mode test --models gpt-4o --dataset test --steps 10 --runs 3 --id 09ce31a1-a719-4ed9-a344-7987214902c1
```

**Quick ID Test**: Run a specific sample by ID(like: 09ce31a1-a719-4ed9-a344-7987214902c1):
```bash
python main.py --mode test --models gpt-4o --dataset test --steps 10 --runs 3 --id <sample_id>
```

---

## 🧭 Modes

- **Agent**: Multi-step agent that explores and then makes a final guess. Uses a simpler prompt for action selection and a final `GUESS` at the last step. Good for end-to-end agent behavior.
- **Benchmark**: Single-image baseline (no multi-step exploration). Good for quick, pure recognition comparisons between models.
- **Test**: Multi-model, multi-run evaluation that records a prediction at every step and logs detailed results. Produces per-step accuracy and average-distance curves; logs saved under `results/test/<timestamp>/<model>/`.
- **Collect**: Generates datasets by sampling locations from MapCrunch and saving `datasets/<name>/golden_labels.json` plus thumbnails. Use this for data generation.

---

## 🔐 Security Note

> ⚠️ **Important**: Never commit your `.env` file to version control. Add `.env` to your `.gitignore` file to keep your API keys secure.

---

<div align="center">

## 📜 License

**MIT License** - see [LICENSE](LICENSE) file for details.

---

<img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love">
<img src="https://img.shields.io/badge/AI%20Powered-🤖-blue.svg" alt="AI Powered">
<img src="https://img.shields.io/badge/Open%20Source-💚-green.svg" alt="Open Source">

**⭐ Star this repo if you find it useful!**

</div>