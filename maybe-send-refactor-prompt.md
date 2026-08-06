# MaybeSend Refactor Prompt

Refactor Rig’s cross-platform Send/Sync compatibility vocabulary to use conventional Rust naming while preserving its current behavior.

Backwards compatibility and breaking semver are not concerns.

## Goals

1. Rename the project-local compatibility traits:
   - `WasmCompatSend` → `MaybeSend`
   - `WasmCompatSync` → `MaybeSync`
   - `WasmBoxedFuture` → `BoxFuture`

2. Preserve the exact existing contract:
   - On browser WASM (`all(target_arch = "wasm32", target_os = "unknown")`), `MaybeSend` and `MaybeSync` are no-op marker traits with blanket implementations.
   - On every other target, they require `Send` and `Sync`, respectively.
   - Do not broaden this to every `target_family = "wasm"` target.

3. Implement `BoxFuture` using the established `futures` aliases:
   - Native/non-browser targets: `futures::future::BoxFuture`
   - Browser WASM: `futures::future::LocalBoxFuture`

4. Review `WasmCompatSendStream` separately instead of mechanically renaming it:
   - Inspect every use and determine whether it should become a generic `MaybeSendStream` or a purpose-specific HTTP byte-stream abstraction.
   - Prefer existing `BoxStream`/`LocalBoxStream` aliases where boxing is already appropriate.
   - Preserve its exact item type and target-dependent Send behavior.
   - Avoid unnecessary new abstractions.

5. Keep this compatibility layer owned by Rig:
   - Do not add a “maybe-send” dependency.
   - Do not reuse RMCP’s `MaybeSend`; its semantics differ.
   - Do not introduce a global `local` Cargo feature.
   - Do not migrate Rig’s public traits to `trait-variant` in this change.

6. Audit the workspace for raw `Send`, `Sync`, boxed-future, and boxed-stream bounds:
   - Replace raw bounds only where the API is intended to work on browser WASM.
   - Retain raw `Send + Sync` for genuinely native-only or multithreaded integrations.
   - Pay particular attention to public provider traits, tools, hooks, completion models, vector stores, memory backends, HTTP streaming, erased trait objects, and stored error types.
   - Follow Rig’s platform-specific boxed-error rules.
   - Do not broaden the task into unrelated refactoring.

7. Update all public exports, imports, rustdocs, examples, READMEs, migration documentation, and changelog entries affected by the renamed public API.
   - Clearly document that “Maybe” is target-conditional, not runtime-optional.
   - Explain the exact browser-WASM predicate.
   - Remove stale references to the old names.
   - Since compatibility is not required, remove the old aliases rather than retaining deprecated shims.

8. Add regression coverage proving:
   - Native builds retain real `Send`/`Sync` guarantees.
   - Browser-WASM builds accept representative non-`Send` state such as `Rc`.
   - Boxed futures are `Send` natively and may be non-`Send` on browser WASM.
   - Key facade paths compile from both `rig-core` and the root `rig` crate.

Before editing, read the existing compatibility module and all relevant trait/object-erasure implementations. Keep the change mechanical where possible, but validate every raw bound rather than blindly replacing it.

## Verification

Run the smallest targeted tests and compile fixtures first, followed by at least:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo check -p rig-core --target wasm32-unknown-unknown
cargo check -p rig-agent --target wasm32-unknown-unknown
cargo check -p rig --target wasm32-unknown-unknown
```

Use `rg` afterward to confirm the old identifiers are gone except where intentionally mentioned in migration documentation.

Do not commit, push, or open a PR unless separately requested. In the final handoff, summarize the public API changes, raw bounds retained intentionally, verification results, and any remaining WASM limitations.
