> **Historical document.** This file records the initial project bootstrap procedure.
> It is not the setup or validation procedure for the final submission.
> For the current instructions, use the repository root `README.md`.
# README_NEXT_STEPS.md

# How to use these files

1. Create the project folder `IRP_BeyondUtility_Project/`.
2. Copy these files into the project root:
   - `PROJECT_SPEC.md`
   - `AGENTS.md`
   - `CODEX_PROMPTS.md`
3. Initialize Git.
4. Clone the original repository under `original_repo/`.
5. Copy the paper PDF under `paper/`.
6. Open the folder in VS Code.
7. Use Codex with the prompts in `CODEX_PROMPTS.md`, one at a time.
8. After every stable phase, run tests and commit.

Recommended first commands:

```bash
mkdir IRP_BeyondUtility_Project
cd IRP_BeyondUtility_Project
git init
mkdir -p original_repo paper
git clone https://github.com/JiangDeccc/EvaLLMasRecommender.git original_repo/EvaLLMasRecommender
# copy 2411.00331v1.pdf into paper/
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/experiments/exp00_smoke_test.py
pytest tests/
```
