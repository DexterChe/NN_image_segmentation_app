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

## License Hygiene

- Confirm LICENSE, NOTICE, and THIRD_PARTY_NOTICES.md are present and up to date.
- Confirm README advertises AGPL-3.0 and links to LICENSE.
- Confirm docs/ultralytics_agpl_strategy.md records the AGPL-3.0 decision.
- If publishing a hosted app, include a visible source-code link to the exact
  public version being run.
- Re-check the licenses of direct dependencies before publishing a container, binary bundle, or hosted service.
- Review whether any included dependency, especially ultralytics, adds copyleft or commercial licensing obligations for your distribution model.
- Verify that no pretrained weights, checkpoints, fonts, logos, or sample assets are being published without confirmed upstream permission.

## Final Gate

- Confirm only source code, docs, and approved config are staged.
- Confirm .local remains ignored.
- Confirm the audit report is not staged.
- Confirm the repository can be shared without exposing unrelated parent-repo history.
