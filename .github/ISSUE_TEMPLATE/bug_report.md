---
name: Bug report
about: A reproducible problem with the pipeline, build, or Meshroom integration
title: '[BUG] '
labels: bug
---

## Summary
<!-- One sentence: what's broken? -->

## Repro steps
1. ...
2. ...
3. Observed: ...
4. Expected: ...

## Environment
- macOS version (Apple menu → About):
- Apple Silicon chip (M1 / M2 / M3 / M4):
- Xcode / Command Line Tools version (`xcode-select -p && pkgutil --pkg-info=com.apple.pkg.CLTools_Executables`):
- Homebrew prefix (`brew --prefix`):
- Branch / commit (`git rev-parse HEAD`):

## Component
- [ ] Build (`cmake` / `ninja`)
- [ ] `aliceVision_*` pipeline binary
- [ ] Meshroom integration (`meshroom-mac/`, `plugins/`)
- [ ] AI segmentation (`SegmentationBiRefNet`, `ai-models/`)
- [ ] Docs / install instructions

## Logs / output
<details>
<summary>Relevant excerpt (last 50 lines is usually enough)</summary>

```
<paste here>
```

</details>

## Tests
Did you run `ctest -j8` and `python -m pytest tests/python` on a fresh build
before reporting? If something failed there, paste the failing test's name
and output below; the fix might be a separate test issue.

```
<ctest output or empty>
```
