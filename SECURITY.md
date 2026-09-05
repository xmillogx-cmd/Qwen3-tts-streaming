# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| Latest 0.1.x release on PyPI | ✅ |
| Anything older than the latest release | ❌ — upgrade to the latest version |

Only the most recent PyPI release of `qwen3-tts-streaming` is supported for security fixes.

## Reporting a vulnerability

- **Preferred:** use GitHub's private vulnerability reporting: open this repository → *Security* tab → *Report a vulnerability*. This keeps the report private until we decide to disclose it.
- If private reporting is not available, contact the maintainer directly through any channel published in the repository. Do **not** post security-sensitive details in public issues before they are fixed.

## Scope

In scope:

- The `fast_tts` package (streaming engine, audio player, CLI, `_patch/` CUDA-graph overlay) and this repository's build/packaging metadata.

Out of scope — report to the upstream maintainers instead:

- `qwen-tts` / Qwen3-TTS → https://github.com/QwenLM/Qwen3-TTS
- `transformers`, `torch`, and other third-party dependencies → their respective projects

## Response policy

We aim to acknowledge a valid report within 48 hours and ship a fix in the next release. Because PyPI versions are immutable, security fixes always land as a **new** version — users pinning exact versions should upgrade after a security release is published.
