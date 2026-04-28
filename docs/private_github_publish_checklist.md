# Private GitHub Publish Checklist

Use this checklist before creating a GitHub repository for this app.

## Repository Boundary

- Publish this folder as a standalone repository root.
- Do not publish the current parent repository Segmentations as-is.
- If needed, create a fresh repository from this directory or import it with a filtered history.

## Secrets and Sensitive Data

- Verify there are no API keys, tokens, passwords, private keys, or credentials.
- Verify there is no .env file, Streamlit secrets file, or other local secret material.
- Verify there are no hardcoded private local paths.

## Data Hygiene

- Keep segmentation_results and other generated outputs out of git.
- Keep raw and processed images out of git unless they are explicitly approved public samples.
- Remove or rename any sample identifiers that should not be public.
- Do not commit model weights or checkpoints.

## Git Hygiene

- Review git status manually before staging anything.
- Stage files explicitly; do not use git add -A blindly.
- Confirm mode-only changes are intentional.
- Re-check git history if publishing from any repository that predates this folder.

## Docker Hygiene

- Build only from this folder as the Docker context.
- Confirm .dockerignore excludes local data, caches, model files, and secrets.

## Final Gate

- Confirm only source code, docs, and approved config are staged.
- Confirm .local remains ignored.
- Confirm the audit report is not staged.
- Confirm the repository can be shared without exposing unrelated parent-repo history.