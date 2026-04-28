# Security Policy

## Supported Scope

This repository is intended to contain application source code and documentation only.

Do not include the following in issues, pull requests, or commits:

- secrets, tokens, passwords, or private keys
- Streamlit secrets files or local environment files
- model weights or checkpoints
- raw input images, customer data, sample data, or generated analysis outputs

## Reporting a Vulnerability

If you discover a security issue, do not open a public issue with exploit details or credentials.

Report it privately to the repository maintainer with:

- a short description of the issue
- affected file paths or components
- reproduction steps
- impact assessment
- suggested remediation, if available

If a report accidentally contains secrets or sensitive data, rotate or revoke the exposed material before any public discussion.

## Hardening Expectations

Before publishing or sharing the repository:

- verify that local data and generated outputs are excluded by ignore rules
- confirm that no secrets are present in current files or git history
- publish this app from its own repository root, not from an unrelated parent repository