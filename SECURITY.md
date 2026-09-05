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

## Known upstream vulnerabilities (transformers 4.x)

The `transformers` dependency is pinned to the 4.x line (`>=4.57,<5`) because `qwen-tts 0.1.1` hard-pins `transformers==4.57.3`. Three high-severity advisories affect that version:

| CVE | Advisory | Issue | Patched in |
|-----|----------|-------|------------|
| CVE-2026-4372 | [GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | RCE via `_attn_implementation_internal` field in a malicious `config.json` when loading a model from an untrusted Hub repository | 5.3.0 |
| CVE-2026-5241 | [GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r) | RCE during LightGlue model initialization — untrusted config overrides `trust_remote_code=False` | 5.5.0 |
| CVE-2026-9856 | [GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) | Path traversal in `save_pretrained()` via crafted chat-template keys from an untrusted tokenizer/processor repository | 5.10.0 |

**Why this does not affect normal use of this package:**

- `fast_tts` imports only `StaticCache` and the causal-mask helpers (`masking_utils`) from transformers — none of the vulnerable code paths is exercised by the streaming pipeline.
- The model is loaded by qwen-tts from a **local directory** you downloaded yourself; there is no LightGlue usage and no `save_pretrained()` call anywhere in this project.

**Mitigation:** download model weights only from the official [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) repository — never from third-party mirrors or re-uploads (this is what CVE-2026-4372 exploits).

**Plan:** when `qwen-tts` ships a release compatible with transformers ≥ 5.10, the pins will be raised and the full GPU test suite re-run before publishing a new package version.
