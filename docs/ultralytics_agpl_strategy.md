# Ultralytics AGPL Decision

This note records the licensing decision for publishing this repository
publicly.

It is a practical engineering checklist, not legal advice.

## Current State

- The repository source code is licensed under GNU Affero General Public License
  v3.0 only (AGPL-3.0-only).
- `requirements.txt` includes `ultralytics`.
- `segmentation.py` imports `SAM`, `YOLO`, and `FastSAM` from `ultralytics`.
- The app documentation names SAM, SAM 2, MobileSAM, FastSAM, and YOLOv8
  checkpoints.
- Model weights are not committed, and generated outputs are ignored.

This means the public repository does not ship Ultralytics code or weights, but
it does define an application intended to run with Ultralytics software.

## Official Ultralytics Position Reviewed

Reviewed on 2026-04-29:

- Ultralytics licensing page:
  https://www.ultralytics.com/license
- Ultralytics docs licensing section:
  https://docs.ultralytics.com/
- Ultralytics AGPL-3.0 license text:
  https://www.ultralytics.com/legal/agpl-3-0-software-license

Ultralytics presents two licensing paths:

- AGPL-3.0 for open-source use.
- Enterprise license for proprietary or commercial embedding without AGPL
  open-source obligations.

Their licensing FAQ states that use of Ultralytics YOLO code, models,
architectures, training pipelines, or trained/fine-tuned models requires either
open-sourcing the entire project under AGPL-3.0 or obtaining an Ultralytics
Enterprise License. Their FAQ also states this applies to SaaS/API/private
deployments and to projects that do not use pretrained weights.

## Decision

The repository uses the AGPL-3.0-only open-source path.

Completed actions:

- Replaced the repository `LICENSE` with AGPL-3.0.
- Changed README license badge and licensing text from Apache-2.0 to AGPL-3.0.
- Updated `NOTICE` and `THIRD_PARTY_NOTICES.md` to state that the application is
  AGPL-3.0-only because it is designed to run with Ultralytics.
- Added a visible source-code and license notice in the Streamlit sidebar.

Operational requirements:

- Keep all source needed to run the hosted version public, including scripts and
  configuration, except secrets and private data.
- If a modified version is offered as a public network service, make the
  corresponding source code for that running version available to users.
- Do not publish model weights unless their upstream license permits it.
