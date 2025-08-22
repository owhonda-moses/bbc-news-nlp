#!/usr/bin/env bash
set -euo pipefail


if [ -f pat.env ]; then
  chmod 600 pat.env
  source ./pat.env
  echo "PAT loaded."
else
  echo "pat.env not found." >&2
  exit 1
fi


# git config
echo "Configuring git"
git config --global user.name  "owhonda-moses"
git config --global user.email "owhondamoses7@gmail.com"
git config --global init.defaultBranch main

# ~/.netrc for https auth
cat > "$HOME/.netrc" <<EOF
machine github.com
  login x-access-token
  password $GITHUB_TOKEN
EOF
chmod 600 "$HOME/.netrc"

# bootstrap
echo "Bootstrapping git repo…"
git remote remove origin 2>/dev/null || true
if [ ! -d .git ]; then
  git init
fi
git remote add origin \
  https://x-access-token:${GITHUB_TOKEN}@github.com/owhonda-moses/bbc-news-nlp.git
git fetch origin main --depth=1 2>/dev/null || true
git checkout main 2>/dev/null || git checkout -b main


echo "Setting up Python with Poetry"
curl -sSL https://install.python-poetry.org | python3.11 -
export PATH="/root/.local/bin:$PATH"
echo "Poetry installed."


echo "Setting up environment and packages"
poetry config virtualenvs.in-project true
poetry env use python3.11
poetry lock
poetry install --no-root --no-interaction --no-ansi 2>/dev/null 

echo "Python env ready: $(poetry run python -V)"

echo "Setting up spaCy transformer model"
poetry run python -m spacy download en_core_web_trf 2>/dev/null 

poetry run python -m ipykernel install --user --name="bbc-nlp" --display-name="python-bbc-nlp"

# echo "Verifying setup"
# poetry run python -c "
# import torch
# import spacy
# print(f'PyTorch version: {torch.__version__}')
# gpu_available = torch.cuda.is_available()
# print(f'GPU available: {gpu_available}')
# if gpu_available:
#     print(f'CUDA version: {torch.version.cuda}')
# print('Loading spaCy model en_core_web_trf...')
# nlp = spacy.load('en_core_web_trf')
# print('SpaCy transformer model loaded.')
# "

echo "Setup complete"