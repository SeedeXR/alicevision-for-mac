<!-- Thanks for the PR! Fill in every section honestly. PRs without these
sections will be asked for revisions before review. -->

## Summary
<!-- One paragraph. What does this change and WHY. -->

## Related issue(s)
Fixes # / Refs #

## Scope
- [ ] Code (`src/`, `tests/`, `meshroom-native/`)
- [ ] Build system (`CMakeLists.txt`, `cmake/`)
- [ ] Docs (`docs/`, top-level `.md` files, `mkdocs.yml`)
- [ ] Patches (`patches/`)
- [ ] CI / tooling (`.github/`, `scripts/`)
- [ ] Other: ___

## Testing
<!-- Required. Tick every box that applies; explain if blank. -->
- [ ] `ctest -j8` is 37/37 (or N/N for new tests).
- [ ] `swift test` (in `meshroom-native/`) is 151/151 (or N/N).
- [ ] New tests added for new functionality.
- [ ] Existing tests still pass.
- [ ] For perf changes: before/after numbers via `AV_PROFILE_ADAPTER=ON`
      on Monstree mini3 view 0.

### Before / after numbers (if perf-related)
```
<paste relevant rows from `tail -25 /tmp/dm_profile.txt`>
```

## Numerical correctness (if kernel-related)
- [ ] Test fixture validates against CPU-FP64 reference within documented budget.
- [ ] No precision changes (FP32 stays FP32).
- [ ] No silent algorithm changes.

## Docs
- [ ] User-facing change → updated `docs/user/` and / or top-level `*.md`.
- [ ] Dev-facing change → updated `docs/dev/`.
- [ ] Approach changed since the original todo item → struck through old
      approach in `memory/todo.md` with explicit reason.

## Checklist
- [ ] I read `CONTRIBUTING.md`.
- [ ] My commits explain WHY, not just WHAT.
- [ ] I did NOT modify anything in `upstream/` (it's a symlink to a
      read-only reference).
- [ ] If I touched LICENSE / third-party / `patches/`, I checked that
      the licensing remains compatible with MIT.
- [ ] If I added a dependency, I discussed it in an issue first.
