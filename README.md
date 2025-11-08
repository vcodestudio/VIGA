# AgenticVerifier

MCP-based agent library for dual-agent (Generator/Verifier) interactive frameworks, supporting 3D (Blender), 2D (PPTX), and Design2Code (HTML/CSS) modes. Plug and play for automated code generation, execution, and verification workflows.

## Quick Start

Since mcp-framework can decouple agents and tools, we highly recommend you to create different envs to run the framework.

The official envs names are `[agent, blender, pptx, web]` for agent, exec_blender, exec_slides, exec_html. You can found the requirement under `requirements/`

### Blender

For blender, you need to further install executable `blender` and `infinigen` in the system:

```zsh
cd utils
git clone https://github.com/princeton-vl/infinigen.git
bash scripts/install/interactive_blender.sh
```

However, `BlenderGym` uses an old-version of `infinigen`, some code is no longer compatible with the new libraries.

To run `BlenderGym`, you have to:

```zsh
git clone git@github.com:richard-guyunqi/infinigen.git
cd infinigen
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh
```

### Slide

For slide, you need to further install `unoconv` in the system:

```zsh
sudo apt install -y libreoffice unoconv
# To verify the installation, run
/usr/bin/python3 /usr/bin/unoconv --version
```

### Web

For web design, you need to install `google-chrome` in the system:

```zsh
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
```

Complete `utils/_api_keys.py` and `utils/_path.py` to your system setting.