# Third-Party Notices

This file provides a best-effort summary of direct runtime dependencies and
related redistribution notes for this repository as audited on 2026-04-29.
It is informational only, is not legal advice, and does not replace the
original license texts, NOTICE files, or terms published by upstream projects.

## Repository License

- The source code in this repository is offered under GNU Affero General Public
  License v3.0 only (AGPL-3.0-only).
- See LICENSE for the project license text.
- See NOTICE for repository-level attribution and distribution notes.

## Direct Python Dependencies

The project currently declares the following direct dependencies in
requirements.txt. Versions are intentionally not pinned in this repository, so
license checks must be repeated against the exact lockfile, wheelhouse,
container image, or deployment environment being distributed.

| Package | Version spec | Declared license | Notes |
| --- | --- | --- | --- |
| streamlit | >=1.30.0 | Apache-2.0 | Streamlit package metadata exposes Apache-2.0 licensing in current releases. |
| ultralytics | >=8.1.0 | AGPL-3.0 or separate Ultralytics enterprise/commercial license | Important: redistributing or network-deploying an installation, container, bundled app, or hosted service that includes this dependency can trigger AGPL obligations unless you have a separate commercial license from Ultralytics. Review upstream terms before distribution or deployment. |
| opencv-python-headless | >=4.8.0 | MIT for packaging scripts; OpenCV core under Apache-2.0; bundled wheel components under additional third-party licenses | The headless package is used because this app does not rely on OpenCV GUI APIs. Upstream documentation states that wheels ship with FFmpeg under LGPL-2.1 and include other binaries covered by LICENSE-3RD-PARTY material. Review the exact wheel notices before binary redistribution. |
| numpy | >=1.24.0 | BSD-3-Clause plus notices for bundled components in binary wheels | Binary wheels can include OpenBLAS/LAPACK and compiler runtime libraries with their own notices. |
| pandas | >=2.0.0 | BSD-3-Clause | Retain upstream copyright and redistribution notices when redistributing. |
| matplotlib | >=3.7.0 | PSF-based license | Review upstream license text when redistributing a packaged application. |
| scikit-image | >=0.21.0 | Primarily BSD-style; includes some BSD-2-Clause and MIT components | Package metadata contains mixed notices for included files. |
| torch | >=2.0.0 | BSD-style plus notices for bundled components in binary distributions | PyTorch distributions ship separate notices for third-party components. |
| tqdm | >=4.65.0 | MPL-2.0 and MIT | Current package metadata reports combined MPL-2.0 and MIT licensing. |

## Key Compatibility Notes

- This repository is licensed under AGPL-3.0-only to align the public application
  with the open-source Ultralytics licensing path.
- If the app is modified and distributed or offered as a public network service,
  AGPL-3.0 requires the corresponding source code for the running version to be
  made available to users.
- Ultralytics' public licensing FAQ states that projects using Ultralytics YOLO
  code, models, architectures, training pipelines, or trained/fine-tuned models
  must either open-source the entire project under AGPL-3.0 or obtain an
  Ultralytics Enterprise License. This repository has selected the AGPL-3.0-only
  open-source path. See docs/ultralytics_agpl_strategy.md for the
  repository-level decision record.
- Using `opencv-python-headless` avoids Qt GUI wheel dependencies that are not
  needed by this Streamlit app. FFmpeg and other third-party wheel components
  still require notice and license review for binary redistribution.
- If you only publish this repository as source code and do not include model
  weights, built wheels, or containers, the immediate redistribution burden is
  lower. A packaged app, Docker image, internal wheelhouse, or public service
  requires a fresh scan of the exact artifacts.

## Models, Checkpoints, and Trademarks

- This repository does not include pretrained model weights.
- Model families referenced by the application and documentation include SAM,
  SAM 2, MobileSAM, FastSAM, and YOLOv8.
- Those checkpoints, brand names, and associated assets are governed by the
  separate license terms, acceptable use terms, and trademark policies of their
  respective publishers.
- Before publishing a Docker image, desktop bundle, internal wheelhouse, or
  hosted service based on this repository, verify the terms for every model
  checkpoint you download or redistribute.

## Audit Scope and Gaps

- This review covered repository files, direct dependencies listed in
  requirements.txt, locally available package metadata, and selected upstream
  project metadata available on 2026-04-29.
- Transitive dependencies were not exhaustively enumerated in this file.
- If you redistribute a built environment rather than source only, run a full
  SBOM or license scan against the exact lockfile, wheelhouse, or container
  image you ship.

## Upstream References Reviewed

- Ultralytics PyPI project metadata and licensing notes:
  https://pypi.org/project/ultralytics/
- OpenCV Python package licensing notes:
  https://github.com/opencv/opencv-python
- OpenCV Python PyPI licensing notes for wheel components:
  https://pypi.org/project/opencv-python/
