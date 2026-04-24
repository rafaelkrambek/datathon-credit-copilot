#!/usr/bin/env bash
# Não usamos 'set -e' de propósito: queremos que o container suba mesmo se algum
# passo opcional falhar. Logamos tudo pra diagnóstico posterior.
set -u
exec > >(tee -a /tmp/post-create.log) 2>&1

echo "=== post-create.sh iniciado em $(date) ==="

echo "--- Atualizando pip ---"
python -m pip install --upgrade pip setuptools wheel || echo "WARN: pip upgrade falhou"

echo "--- Instalando uv (Astral) ---"
curl -LsSf https://astral.sh/uv/install.sh | sh || echo "WARN: uv install falhou"
echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "--- Instalando Kaggle CLI ---"
pip install --user kaggle || echo "WARN: kaggle install falhou"

echo "--- Instalando DVC ---"
pip install --user 'dvc[s3]' || echo "WARN: dvc install falhou"

echo "--- Aliases úteis ---"
cat >> ~/.bashrc << 'BASHRC'

# Datathon Credit Copilot aliases
alias ll='ls -lah'
alias gs='git status'
alias gd='git diff'
alias py='python'
alias jn='jupyter notebook --no-browser --ip=0.0.0.0'

BASHRC

echo "=== post-create.sh concluido em $(date) ==="
echo ""
echo "Ambiente pronto. Próximos passos:"
echo "  1. source ~/.bashrc"
echo "  2. Criar pyproject.toml"
echo "  3. uv pip install -e ."
