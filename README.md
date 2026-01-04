# AgenticVerifier

An MCP-based dual-agent framework for interactive code generation and verification. AgenticVerifier implements a **Generator-Verifier** pattern where a Generator agent creates code/content based on target specifications, and a Verifier agent evaluates outputs and provides feedback for iterative refinement.

## Supported Domains

| Mode | Description | Output |
|------|-------------|--------|
| **BlenderGym** | 3D scene modification from initial code | Blender Python code |
| **BlenderStudio** | Progressive 3D scene generation (Level 1-3) | Blender Python code |
| **Static Scene** | 3D scene creation from scratch using target image | Blender scene (.blend) |
| **Dynamic Scene** | Animated 3D scene generation with keyframes | Blender animation |
| **AutoPresent** | Slide generation from descriptions | PowerPoint (PPTX) |
| **Design2Code** | HTML/CSS generation from design images | HTML/CSS files |

## Project Structure

```
AgenticVerifier/
├── main.py                 # Main entry point for dual-agent loop
├── agents/                 # Core dual-agent implementation
│   ├── generator.py        # Generator agent logic
│   ├── verifier.py         # Verifier agent logic
│   ├── tool_client.py      # MCP tool client
│   └── prompt_builder.py   # System/user prompt construction
├── runners/                # Dataset runners for batch execution
│   ├── blendergym.py       # BlenderGym runner
│   ├── blenderstudio.py    # BlenderStudio runner
│   ├── static_scene.py     # Static scene runner
│   ├── dynamic_scene.py    # Dynamic scene runner
│   ├── autopresent.py      # AutoPresent runner
│   └── design2code.py      # Design2Code runner
├── tools/                  # MCP tool servers
│   ├── exec_blender.py     # Blender execution & rendering
│   ├── exec_slides.py      # PowerPoint generation
│   ├── exec_html.py        # HTML/CSS rendering
│   ├── investigator.py     # Scene investigation & analysis
│   ├── meshy.py            # 3D asset generation via Meshy API
│   ├── sam_init.py         # 3D scene reconstruction (SAM)
│   ├── generator_base.py   # Base generator tools
│   └── verifier_base.py    # Base verifier tools
├── prompts/                # System and user prompts per mode
├── data/                   # Datasets for each mode
├── evaluators/             # Mode-specific evaluation scripts
├── utils/                  # Utility modules
├── requirements/           # Environment requirement files
└── output/                 # Output directory for runs
```

## Installation

### 1. Create Virtual Environments

Since MCP decouples agents and tools, we recommend creating separate environments:

```bash
# Agent environment (required)
conda create -n agent python=3.10
conda activate agent
pip install -r requirements/requirement_agent.txt

# Blender environment (for 3D modes)
conda create -n blender python=3.10
conda activate blender
pip install -r requirements/requirement_blender.txt

# PPTX environment (for AutoPresent)
conda create -n pptx python=3.10
conda activate pptx
pip install -r requirements/requirement_pptx.txt

# Web environment (for Design2Code)
conda create -n web python=3.10
conda activate web
pip install -r requirements/requirement_web.txt
```

### 2. External Dependencies

#### Blender (for 3D modes)

For BlenderGym, you need a specific version of Infinigen:

```bash
cd utils
git clone git@github.com:richard-guyunqi/infinigen.git
cd infinigen
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh
```

For other 3D modes (static_scene, dynamic_scene, blenderstudio):

```bash
cd utils
git clone https://github.com/princeton-vl/infinigen.git
bash scripts/install/interactive_blender.sh
```

#### LibreOffice (for AutoPresent)

```bash
sudo apt install -y libreoffice unoconv
# Verify installation
/usr/bin/python3 /usr/bin/unoconv --version
```

#### Google Chrome (for Design2Code)

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
```

### 3. Configuration

Create configuration files in `utils/`:

**`utils/_api_keys.py`**:
```python
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
CLAUDE_API_KEY = "your-claude-api-key"
CLAUDE_BASE_URL = "https://api.anthropic.com/v1"
GEMINI_API_KEY = "your-gemini-api-key"
MESHY_API_KEY = "your-meshy-api-key"  # For 3D asset generation
VA_API_KEY = "your-va-api-key"
```

**`utils/_path.py`**:
```python
# Configure paths for different tool environments
path_to_cmd = {
    "tools/exec_blender.py": "/path/to/blender/env/bin/python",
    "tools/exec_slides.py": "/path/to/pptx/env/bin/python",
    "tools/exec_html.py": "/path/to/web/env/bin/python",
    # Add other tool paths as needed
}
```

## Usage

### Main Entry Point

```bash
python main.py --mode <mode> --model <model> [options]
```

#### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Mode: blendergym, blenderstudio, static_scene, dynamic_scene, autopresent, design2code | Required |
| `--model` | Vision model (gpt-4o, claude-sonnet-4, gemini-2.5-pro, etc.) | gpt-4o |
| `--api-key` | API key for the model | From env |
| `--api-base-url` | API base URL | From env |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--memory-length` | Chat history length | 12 |
| `--output-dir` | Output directory | None |
| `--init-code-path` | Path to initial code file | None |
| `--init-image-path` | Path to initial images | None |
| `--target-image-path` | Path to target images | None |
| `--target-description` | Natural language task description | None |
| `--generator-tools` | Comma-separated generator tool scripts | tools/generator_base.py |
| `--verifier-tools` | Comma-separated verifier tool scripts | tools/verifier_base.py |
| `--no-tools` | Disable tool calling mode | False |
| `--clear-memory` | Clear memory between rounds | False |

#### Blender-specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--blender-command` | Path to Blender executable | utils/infinigen/blender/blender |
| `--blender-file` | Blender template file | None |
| `--blender-script` | Blender execution script | data/blendergym/pipeline_render_script.py |
| `--gpu-devices` | GPU devices for rendering | From CUDA_VISIBLE_DEVICES |

---

## Runners

Runners provide batch execution for datasets with parallel processing support.

### BlenderGym Runner

Run 3D scene modification tasks:

```bash
python runners/blendergym.py \
    --dataset-path data/blendergym \
    --task all \
    --model gpt-4o \
    --max-rounds 10 \
    --max-workers 8
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to BlenderGym dataset | data/blendergym |
| `--task` | Task type: all, blendshape, geometry, lighting, material, placement | all |
| `--task-id` | Specific task ID (e.g., "1") | None |
| `--test-id` | Retest failed tasks from previous run | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--memory-length` | Memory length | 24 |
| `--max-workers` | Parallel workers | 8 |
| `--sequential` | Run sequentially | False |
| `--no-tools` | Disable tools | False |
| `--generator-tools` | Generator tool servers | tools/exec_blender.py,tools/generator_base.py,tools/initialize_plan.py |
| `--verifier-tools` | Verifier tool servers | tools/verifier_base.py,tools/investigator.py |

### BlenderStudio Runner

Run progressive 3D scene generation tasks:

```bash
python runners/blenderstudio.py \
    --dataset-path data/blenderstudio \
    --task all \
    --model gpt-4o \
    --max-workers 8
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to BlenderStudio dataset | data/blenderstudio |
| `--task` | Task level: all, level1, level2, level3 | all |
| `--task-id` | Specific task ID | None |
| `--test-id` | Retest failed tasks | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--max-workers` | Parallel workers | 8 |

### Static Scene Runner

Create 3D scenes from scratch using target images:

```bash
python runners/static_scene.py \
    --dataset-path data/static_scene \
    --task all \
    --model gpt-4o \
    --max-rounds 100
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to static scene dataset | data/static_scene |
| `--task` | Task name or "all" | all |
| `--test-id` | Test ID for output naming | None |
| `--max-rounds` | Maximum interaction rounds | 100 |
| `--model` | Vision model | gpt-5 |
| `--memory-length` | Memory length | 12 |
| `--prompt-setting` | Prompt setting: none, procedural, scene_graph, get_asset | none |
| `--init-setting` | Init setting: none, minimal, reasonable | none |
| `--max-workers` | Parallel workers | 1 |
| `--text-only` | Use only text as reference | False |
| `--generator-tools` | Generator tools | tools/exec_blender.py,tools/generator_base.py,tools/meshy.py,tools/initialize_plan.py |
| `--verifier-tools` | Verifier tools | tools/investigator.py,tools/verifier_base.py |

### Dynamic Scene Runner

Generate animated 3D scenes:

```bash
python runners/dynamic_scene.py \
    --dataset-path data/dynamic_scene \
    --task all \
    --model gpt-4o \
    --max-rounds 100
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to dynamic scene dataset | data/dynamic_scene |
| `--task` | Task name or "all" | all |
| `--test-id` | Test ID for output naming | None |
| `--max-rounds` | Maximum interaction rounds | 100 |
| `--model` | Vision model | gpt-5 |
| `--prompt-setting` | Prompt setting: none, init | none |
| `--text-only` | Use only text as reference | False |
| `--generator-tools` | Generator tools | tools/exec_blender.py,tools/generator_base.py,tools/initialize_plan.py |
| `--verifier-tools` | Verifier tools | tools/investigator.py,tools/verifier_base.py |

### AutoPresent Runner

Generate presentation slides:

```bash
python runners/autopresent.py \
    --dataset-path data/autopresent/examples \
    --task all \
    --model gpt-4o \
    --max-workers 8
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to AutoPresent dataset | data/autopresent/examples |
| `--task` | Task category: all, art_photos, business, design, entrepreneur, environment, food, marketing, social_media, technology | all |
| `--task-id` | Specific task ID | None |
| `--test-id` | Retest failed tasks | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--max-workers` | Parallel workers | 8 |
| `--sequential` | Run sequentially | False |
| `--generator-tools` | Generator tools | tools/exec_slides.py,tools/generator_base.py |
| `--verifier-tools` | Verifier tools | tools/verifier_base.py |

### Design2Code Runner

Generate HTML/CSS from design images:

```bash
python runners/design2code.py \
    --dataset-path data/design2code/Design2Code-HARD \
    --model gpt-4o \
    --max-workers 8
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-path` | Path to Design2Code dataset | data/design2code/Design2Code-HARD |
| `--case-id` | Specific case ID | None |
| `--max-rounds` | Maximum interaction rounds | 10 |
| `--model` | Vision model | gpt-4o |
| `--memory-length` | Memory length | 12 |
| `--max-workers` | Parallel workers | 8 |
| `--sequential` | Run sequentially | False |
| `--browser-command` | Browser for screenshots | google-chrome |
| `--generator-tools` | Generator tools | tools/exec_html.py,tools/generator_base.py |
| `--verifier-tools` | Verifier tools | tools/verifier_base.py |

---

## Supported Models

AgenticVerifier supports multiple vision-language models:

- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-4o-mini
- **Anthropic**: claude-sonnet-4, claude-opus-4.5
- **Google**: gemini-2.5-pro, gemini-2.0-flash
- **Qwen**: qwen-vl-max, qwen-vl-plus

---

## Output Structure

Each task generates output in the following structure:

```
output/<mode>/<timestamp>/<task_name>/
├── generator_thoughts/     # Generator reasoning logs
├── verifier_thoughts/      # Verifier feedback logs
├── renders/                # Rendered images per round
│   ├── 1/
│   ├── 2/
│   └── ...
├── codes/                  # Generated code per round
├── scores.json             # Evaluation scores
├── args.json               # Task arguments
└── blender_file.blend      # Final Blender scene (3D modes)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input (Target)                          │
│         (description / image / initial code)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Generator Agent                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Build prompt with context                          │   │
│  │ • Call vision model with tool definitions            │   │
│  │ • Parse and execute tool calls via MCP               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Servers (MCP)                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ exec_blender │ │ exec_slides  │ │  exec_html   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ investigator │ │    meshy     │ │   sam_init   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Verifier Agent                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Analyze visual differences from target             │   │
│  │ • Generate structured feedback                       │   │
│  │ • Provide improvement suggestions                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Feedback to Generator │
              │    (Loop or End)       │
              └───────────────────────┘
```

---

## License

MIT License - Copyright 2025 Fugtemypt123

---

## Citation

If you use AgenticVerifier in your research, please cite:

```bibtex
@software{agenticverifier2025,
  title = {AgenticVerifier: MCP-based Dual-Agent Framework for Interactive Code Generation and Verification},
  year = {2025},
  url = {https://github.com/Fugtemypt123/AgenticVerifier}
}
```
