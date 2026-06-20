# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | ✅ Active support  |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue, please email: **yacine.baghli@outlook.fr**

Include:
- A description of the vulnerability.
- Steps to reproduce.
- Potential impact.

You can expect an acknowledgement within **48 hours** and a resolution timeline within **7 days** for confirmed issues.

## Data & Model Security Notes

- All training data is synthetically generated at runtime — no private datasets are included.
- Model checkpoints (`.pt`, `.pth`) are excluded from version control via `.gitignore` and should never be committed.
- No API keys or secrets are required to run the project.
