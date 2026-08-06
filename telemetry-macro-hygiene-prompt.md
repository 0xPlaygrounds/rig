# Fix the exported telemetry macro's downstream dependency hygiene (P1 follow-up on PR #2208)

Repo: Rig workspace.
Branch: `telemetry-parent-contract`.
Open PR: #2208, base `main`.
Start from commit `c311118a8b4c57cc3959e7eb3dccecbaa96191f9`.

## Objective

Fix the remaining P1 review finding in the new public
`rig_core::telemetry::completion_parent_span!` API.

The macro must compile in a downstream crate whose only direct dependency is
`rig-core`. The downstream crate must not need to add `tracing` itself, and the
fix must also work when `rig-core` is renamed in the consumer's `Cargo.toml`.

## Confirmed failure

`completion_parent_span!` delegates to the exported hidden helper
`__rig_canonical_completion_span!`. Their expansions currently contain absolute
caller-side paths such as:

```rust
::tracing::info_span!(...)
::tracing::field::Empty
::tracing::Span::current()
```

Exported macro paths are resolved in the invoking crate. Therefore this minimal
consumer fails with `E0433: could not find tracing in the list of imported
crates`:

```toml
[dependencies]
rig-core = { path = "/path/to/rig/crates/rig-core" }
```

```rust
fn main() {
    let _span = rig_core::telemetry::completion_parent_span!(
        target: "consumer",
        name: "chat",
        operation: "chat",
        system_instructions: Option::<&str>::None,
    );
}
```

The existing macro doctest does not catch this because a `rig-core` doctest is
compiled with `rig-core`'s own dependency set, which already includes
`tracing`.

## Required implementation

Work primarily in:

- `crates/rig-core/src/telemetry/mod.rs`
- a new downstream compile fixture/test under `crates/rig-core/tests/`
- the existing unreleased telemetry changelog entries, if needed

Implement the fix as follows:

1. Re-export `tracing` through a public-but-doc-hidden, private-looking
   `rig-core` path intended only for exported macro expansion. For example, use
   a `#[doc(hidden)] pub use tracing as __tracing;` near the telemetry macros,
   or an equivalently narrow hidden module/re-export.
   - The path must be publicly reachable because macro expansion occurs in a
     different crate.
   - Do not ask consumers to add `tracing` directly.
   - Do not add a normal user-facing `tracing` API to the documented surface.

2. Replace every externally expanded `::tracing` reference in
   `__rig_canonical_completion_span!` and `completion_parent_span!` with a
   `$crate`-qualified path through that hidden re-export, including:
   - `info_span!`
   - every `field::Empty`
   - the default `Span::current()` parent

3. Keep `$crate` hygiene intact when `rig-core` is renamed by the downstream
   crate. Do not hardcode `rig_core` inside an exported macro expansion.

4. Do not change:
   - the marker name or version;
   - any `gen_ai.*` field;
   - the field set or field ordering;
   - adoption behavior;
   - default or explicit-parent behavior;
   - warning behavior;
   - the macro's public invocation syntax.

## Required regression fixture

Add a real downstream compile fixture rather than relying only on a doctest or
an in-crate unit test.

Follow the repository's existing nested-Cargo patterns, especially:

- `tests/tool_facade_features.rs`
- `tests/fixtures/tool_facade/`
- `crates/rig-derive/tests/dependency_rename.rs`
- `crates/rig-derive/tests/fixtures/`

Suggested shape:

```text
crates/rig-core/tests/macro_hygiene.rs
crates/rig-core/tests/fixtures/telemetry_macro_consumer/Cargo.toml
crates/rig-core/tests/fixtures/telemetry_macro_consumer/src/main.rs
```

The fixture manifest must:

- contain its own `[workspace]` so it is independent of the parent workspace;
- depend on the local `rig-core` path;
- have no direct or dev dependency on `tracing`;
- preferably rename the dependency, for example:

```toml
[dependencies]
rig_runtime_core = { package = "rig-core", path = "../../.." }
```

The fixture program must invoke the macro through the renamed dependency and
cover both supported arms:

- default current parent;
- explicit `parent: None`.

The integration test must run `cargo check --manifest-path ...` against the
fixture, use a deterministic repository-local target directory, capture stdout
and stderr, and print both on failure. Keep it consistent with the existing
nested-Cargo tests and avoid shell-specific commands.

This regression must fail on `c311118a` with the current `E0433` and pass only
after the macro paths are made dependency-hygienic.

## Documentation and changelog

- If the public macro docs discuss dependencies, make it explicit that callers
  do not need a direct `tracing` dependency merely to invoke this macro.
- Extend the existing unreleased telemetry entry in both `CHANGELOG.md` and
  `crates/rig-core/CHANGELOG.md`; do not add a separate duplicate entry.
- Do not change `MIGRATING.md` unless the implementation changes migration
  guidance beyond removing the accidental direct-dependency requirement.

## Verification

Run the smallest checks first, then the full required suite:

```bash
cargo fmt --all --check
cargo test -p rig-core --test macro_hygiene
cargo test -p rig-core --lib telemetry
cargo test -p rig-core --doc completion_parent_span
cargo test -p rig-agent --all-features --lib span_safety_net
cargo check -p rig-core --features wasm --target wasm32-unknown-unknown
cargo test --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
```

Also inspect the final expansion-related diff and confirm there are no remaining
externally expanded absolute `::tracing` paths in either exported macro.

## Commit and PR expectations

- Keep the change on `telemetry-parent-contract` so it updates PR #2208.
- Make one focused conventional commit, for example:
  `fix(telemetry): make completion-parent macro dependency-hygienic`.
- Do not stage either prompt Markdown file.
- Do not add `Co-Authored-By` or generated-by attribution.
- Before pushing, review the full PR diff against `main` and obtain an
  independent final review as required by `AGENTS.md`.
