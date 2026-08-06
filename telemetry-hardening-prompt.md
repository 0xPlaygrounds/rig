# Harden the telemetry completion-parent contract (follow-up on PR #2208, same branch)

Repo: rig workspace, branch `telemetry-parent-contract` (open PR #2208, base `main`).
Work in: `crates/rig-core/src/telemetry/mod.rs` and `crates/rig-agent/src/agent/runner.rs`.

## Context you need

PR #2208 made `rig_core::telemetry::completion_parent_span!` the single declarative
source of the completion-parent contract: it declares the versioned adoption marker
(`COMPLETION_PARENT_MARKER_FIELD = "rig.completion_parent.v1"`) plus every field in
`COMPLETION_PARENT_REQUIRED_FIELDS`. `CompletionSpanBuilder::build` adopts the current
span iff `is_adoptable_completion_parent` passes (marker present + all required fields
statically declared); otherwise it creates a fresh `rig::completions` child span.
Exact-set tests pin macro ↔ constant ↔ builder span against each other. A hidden helper
`__rig_canonical_completion_span!` owns the 10 always-`Empty` canonical fields and
already supports an optional `parent:` argument that the public macro does not expose.

Key constraint driving everything: `tracing` bakes field names into static metadata and
never lets callers read recorded *values* back, so the contract version must live in the
marker field *name*, and no `const` can be spliced into a span declaration. Hard rule
for this task: **do not introduce any new mirrored copy of the field list anywhere,
including inside tests or validation logic** — the whole point of #2208 was eliminating
mirrors.

Review found four residual weaknesses. Root cause of 1–3: the version is an unstructured
string literal appearing in several places, reconciled only by set-equality tests, and
every mismatch degrades silently. Implement the four fixes below.

## Fix 1 — three-state adoption gate with a stale/malformed-parent diagnostic

In `crates/rig-core/src/telemetry/mod.rs`:

- Add a private module-level constant `COMPLETION_PARENT_MARKER_PREFIX: &str =
  "rig.completion_parent"` (keep it private — no new public API surface). Add a unit
  test asserting the relationship: `COMPLETION_PARENT_MARKER_FIELD` equals the prefix
  followed by `.v` + digits. This makes prefix/version drift a test failure.
- In `CompletionSpanBuilder::build`, when adoption fails, detect a *near-miss* parent:
  the current span's metadata declares a field that is either exactly
  `COMPLETION_PARENT_MARKER_PREFIX` (the old pre-#2208 unversioned marker) or starts
  with `COMPLETION_PARENT_MARKER_PREFIX` + `"."` (any other version), OR it declares
  the current marker but is missing required fields (a malformed conforming attempt).
  For a near-miss, emit a one-time-per-process `tracing::warn!` (guard with
  `std::sync::Once`; must compile on `wasm32-unknown-unknown`) that names: the marker
  field found, the expected marker, and — for the malformed case — which required
  fields are missing. Then fall through to the existing fresh-child-span behavior
  unchanged. Ambient spans with no prefixed field stay completely silent (that is the
  normal path for bare completion calls — do not scan or log anything for them beyond
  the existing adoption check).
- Behavior must be observable in tests: add a test that a span declaring the old
  unversioned marker is NOT adopted (fresh child span is created). Testing the
  `Once`-guarded warn text itself is optional; the non-adoption behavior is the
  required assertion.

## Fix 2 — partial-marker test must assert its own premise

The deliberately non-conforming hand-written span test (search for the comment
"Deliberately hand-written", around line ~996 of `telemetry/mod.rs`) hardcodes
`rig.completion_parent.v1 = true`. Before its non-adoption assertions, add a
precondition assertion that the span's metadata declares
`COMPLETION_PARENT_MARKER_FIELD` (i.e. the hand-written literal still matches the
current constant). Purpose: on a future version bump, this test fails on the
precondition — pointing at the stale literal — instead of passing for the wrong
reason (marker mismatch rather than missing fields). Add a one-line comment saying so.

## Fix 3 — count assertions so duplicates can't hide, plus extras doc

- In all three exact-set tests — `completion_parent_span_macro_matches_the_contract_exactly`
  and `canonical_completion_span_declares_exactly_the_required_fields` in
  `telemetry/mod.rs`, and `chat_span_declares_the_full_completion_parent_contract` in
  `rig-agent`'s `span_safety_net` module (`crates/rig-agent/src/agent/runner.rs`) —
  additionally assert `metadata.fields().len() == expected.len()` next to the existing
  `HashSet` equality. Rationale (put in a short comment): duplicate field names collapse
  in a `HashSet`, so set equality alone cannot catch an extras collision or a macro bug
  that declares a field twice.
- In the `completion_parent_span!` doc comment, add one sentence: runtime-specific
  extra fields must not repeat the marker or any required contract field (a duplicate
  compiles but produces a span with two same-named fields, and `record` targets the
  first). Do NOT add a compile-time deny-list of field names inside the macro — that
  would recreate a mirrored field list.

## Fix 4 — expose optional `parent:` on the public macro

Give `completion_parent_span!` a second arm accepting
`target: ..., parent: <expr>, name: ...` (parent between target and name, mirroring the
hidden helper's argument order) that forwards the parent to
`__rig_canonical_completion_span!`. Implement the existing no-parent arm by delegating
to the new arm with `::tracing::Span::current()` so the two can't drift. `parent:`
accepts any expression tracing accepts (including `None`). Update the macro docs (the
default stays "explicitly parented on `Span::current()`"), and extend the macro
exact-set test with one invocation using an explicit `parent:` to prove the arm
declares the identical field set.

## Docs / changelog

- Extend the existing *(core)* [**breaking**] unreleased entry in both `CHANGELOG.md`
  and `crates/rig-core/CHANGELOG.md` (don't add a new entry — same unreleased change)
  with one sentence about the stale-marker warning and the optional `parent:`.
- In `MIGRATING.md`'s "Telemetry completion-parent marker is versioned" section, add
  that a span carrying an outdated or malformed marker now triggers a one-time
  `warn!` identifying the mismatch.

## Verification (all must pass)

- `cargo test -p rig-core --lib telemetry` and `cargo test -p rig-core --doc completion_parent_span`
- `cargo test -p rig-agent --all-features --lib span_safety_net`
- `cargo test --workspace --all-features`
- `cargo clippy --workspace --all-features --all-targets -- -D warnings` (CI enforces -D warnings)
- `cargo fmt --all --check`
- `cargo check -p rig-core --features wasm --target wasm32-unknown-unknown` (the `Once`
  guard and warn path must build there)

Commit on this branch as one commit, conventional-commit style, e.g.
`refactor(telemetry): stale-marker diagnostic, self-checking tests, optional parent`.
Do not add Co-Authored-By or any generated-with attribution lines. Do not rename any
`gen_ai.*` field and do not change the marker version — both are explicitly out of scope.
