# ViiperHexBots Refactoring Plan

## Scope

This plan records the repository audit and the staged refactoring work. The
priority is behavior preservation: correctness-critical Windows input,
shutdown, legacy configuration migration, and detection heuristics are not to
be simplified without tests and evidence.

## Audit findings

### Immediate, low-risk cleanup

- Fix the deferred `exc` closure in `BotLifecycleManager.init_viiper()`.
- Remove confirmed unused imports and locals reported by static analysis.
- Keep legacy configuration and compatibility paths unless their callers and
  migration coverage are removed deliberately. Dependency defaults must use
  explicit ``None`` checks; never select a main implementation by truthiness.
- Do not use process-name fallbacks for destructive lifecycle operations: only
  terminate a process that this manager owns.

### Runtime architecture

- `HuntRuntimeContext` is a broad compatibility façade over gates, startup,
  tracks, policy, validation, wings, and danger services.
- `GateController` owns event gates, session arbitration, waits, startup
  transitions, and input admission. This complexity is correctness-sensitive
  because sit/storage/heal and shutdown share the input boundary.
- `HuntRuntime` and `BotLifecycleManager` contain substantial startup,
  shutdown, retry, and ownership coordination. Consolidation should be
  incremental and must retain bounded cleanup behavior.

### Hunt modes

- `walk` is a supported non-teleport waiting strategy and is covered by
  configuration tests.
- `hybrid` is intentionally a placeholder and has a regression test proving
  it does not teleport. Do not delete it without a product decision; document
  it instead.

### Detection pipeline

- Detector/descriptor code is large but mostly domain-driven. Split only pure,
  evidence-backed responsibilities after measuring performance and preserving
  fixture behavior.

## Runtime defect added to this plan

When damage arrives while the character is sitting, the sit worker must stand,
perform one danger escape for that request, reset area/tracking state, and sit
in the resulting area. Damage recorded during teleport settle is a new request,
not the original request. The old loop checked `danger_sit_requested.is_set()`
without consuming it after an escape. A request raised during settle therefore
remained set and caused repeated `sit_danger_request` teleports forever.

The fix is to consume the pending event at each recovery-loop handoff using the
worker's existing atomic `_sit_danger_detected()` helper. This preserves a
newly raised request for exactly one additional escape and prevents stale event
reprocessing.

## Staged implementation

1. **Phase 0 — safety and cleanup**
   - Fix the sitting danger request state machine.
   - Add regression coverage for damage during teleport settle.
   - Fix the deferred exception closure and confirmed dead imports/locals.
2. **Phase 1 — hunt mode clarity**
   - Keep `walk` and the tested `hybrid` placeholder behavior.
   - Improve documentation/naming rather than removing a configured mode.
3. **Phase 2 — lifecycle consolidation**
   - Extract only duplicated, testable shutdown/startup helpers after mapping
     ownership and retry behavior. Do not weaken bounded cleanup.
4. **Phase 3 — narrower runtime interfaces**
   - Introduce focused protocols/adapters incrementally while retaining the
     context façade for compatibility during migration.
5. **Phase 4 — UI responsibility extraction**
   - Extract cohesive configuration/control panel construction from
     `MainWindow`; retain top-level composition and event routing there.
6. **Phase 5 — detection review**
   - Measure and isolate pure algorithms only where duplication or coupling is
     demonstrated by tests/profiling. No speculative rewrite.
7. **Final validation**
   - Run Python tests/compile checks and Go tests/vet, review all diffs, and
     record deferred risks.

## Status

- [x] Audit recorded.
- [x] Sitting danger-loop root cause recorded.
- [x] Phase 0 implementation and validation (sitting loop fix, callback fix, safe import cleanup; targeted tests pass).
- [x] Phase 1 implementation and validation (hunt-mode contracts documented; hybrid placeholder retained intentionally).
- [x] Phase 2 implementation and validation (shared `_finalize_shutdown()` boundary with existing shutdown regression coverage).
- [x] Phase 3 implementation and validation (focused `SessionLifecycle` protocol added to session-owning workers; façade retained for compatibility).
- [x] Phase 4 implementation and validation (pure status-display formatting/comparison extracted from `MainWindow`; full panel extraction deferred to avoid risky UI churn).
- [x] Phase 5 review/low-risk implementation (detector complexity reviewed; pure CLI response/calibration transformations extracted; no speculative detector split; static cleanup completed).
- [x] Final validation and review (Python 315 passed/2 skipped; Go test and vet passed; compileall and pyflakes clean).
- [x] Deterministic fallback audit (explicit dependency defaults, no arbitrary
  VIIPER process kill fallback, and compatibility fallback behavior retained
  only where lightweight contexts require it).

## Deferred deliberately

- Full `MainWindow` panel construction extraction: the UI is tightly coupled to
  widget state and callbacks, and the pure presentation seam provides value
  without destabilizing startup/shutdown behavior.
- Detector/descriptor decomposition: current complexity is domain-driven and
  fixture-backed; split only after profiling or duplicated algorithm evidence.
  The pure CLI response/calibration layer is already extracted into
  `pybot/recognition/detection_response.py`; the detector core remains intact.
- Removal of the hybrid mode: it is persisted/configurable and covered by a
  no-teleport regression test, so product behavior must be specified first.
