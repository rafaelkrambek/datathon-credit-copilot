#!/usr/bin/env bash
set -euo pipefail

echo "▶ Running post-create setup..."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install uv (fast Python package manager by Astral — 10-100x faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH for future shells
echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Install DVC with S3 support (caso a gente queira usar remote depois)
pip install --user "dvc[s3]"

# Install kaggle CLI (pra baixar dataset no Bloco 6)
pip install --user kaggle

# Give a helpful welcome message
cat >> ~/.bashrc << 'BASHRC'

# --- Datathon Credit Copilot ---
alias ll='ls -lah'
alias gs='git status'
alias gl='git log --oneline --graph --decorate -20'

echo ""
echo "🏦  Datathon Credit Copilot dev container"
echo "   Python:  $(python --version)"
echo "   uv:      $(command -v uv >/dev/null && uv --version || echo 'not found')"
echo "   Project: /workspaces/datathon-credit-copilot"
echo ""
echo "   Próximo passo sugerido: 'make setup' (depois que pyproject.toml existir)"
echo ""
BASHRC

echo "✓ Post-create setup complete."
echo ""
echo "  Abra um NOVO terminal (ou rode 'source ~/.bashrc') pra pegar o uv no PATH."
