# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

`main` branch tracks the latest development; tagged releases are
the supported surface.

## Reporting a Vulnerability

If you discover a security vulnerability — particularly anything
involving arbitrary code execution from untrusted photogrammetry
input (camera images, SfMData JSON, `.mg` project files) — please
report it privately rather than opening a public issue.

**Contact**: open a [GitHub Security Advisory] in this repository
(Security → Advisories → "Report a vulnerability"). This keeps
the disclosure private until a fix ships.

[GitHub Security Advisory]: https://github.com/<placeholder>/alicevision-for-mac/security/advisories

Expect an initial acknowledgment within 5 business days. We'll
work with you on a disclosure timeline that balances user safety
with your finding's complexity.

## Scope

In scope:
- Memory-safety bugs in our C++ overlay code (`src/av_gpu/`,
  `src/depth_map_metal/`, kernel host code in `tests/` and
  `src/shaders/`).
- Sandbox-escape via SwiftUI native app input parsing
  (`meshroom-native/Sources/`).
- Build-time RCE via maliciously crafted CMake/patch files.
- Pipeline-binary input handling: validate-first behaviour on
  malformed SfMData JSON, EXR depth maps, etc.

Out of scope (file upstream or with Apple, not us):
- Vulnerabilities in upstream AliceVision (file at
  [alicevision/AliceVision](https://github.com/alicevision/AliceVision)).
- Vulnerabilities in upstream Meshroom (file at
  [alicevision/Meshroom](https://github.com/alicevision/Meshroom)).
- macOS / Metal driver bugs (file via Apple Feedback Assistant).
- Vulnerabilities in Homebrew dependencies (alembic, boost,
  eigen, etc.) — file with the upstream project.

## Known issues

None tracked in this section currently. Run-time
limitations are documented in `README.md` "Known issues"
and `docs/user/troubleshooting.md`.
