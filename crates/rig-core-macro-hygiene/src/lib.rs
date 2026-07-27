//! Compile-only dependency-hygiene tripwire for `rig-core`'s exported macros.
//!
//! `completion_parent_span!` is the supported way for a third-party runtime to
//! declare a completion-parent span, so its expansion must resolve every path
//! through `rig-core` itself — a downstream crate should not need a direct
//! `tracing` dependency merely to invoke it. That property is invisible from
//! inside `rig-core`, where `tracing` is always in scope, so it can only be
//! checked from a crate that does not depend on `tracing`.
//!
//! This crate is that crate: it depends on `rig-core` alone, under a renamed
//! alias (so `$crate` resolution is exercised too), and does nothing but
//! invoke the macro without naming a single `tracing` path. A regression fails
//! the build directly — no nested `cargo` invocation, no second target
//! directory, no extra dependency resolution.
//!
//! CI compiles this crate through the `doctest` job's `cargo test --doc
//! --workspace --all-features`. Being a workspace member is not by itself
//! enough: this workspace's root manifest is the `rig` package and sets no
//! `default-members`, so a bare `cargo check`/`cargo nextest run` at the root
//! selects `rig` alone and never builds this crate. If that job loses
//! `--workspace`, the tripwire stops tripping.
//!
//! Checking it with a dedicated `cargo check -p rig-core-macro-hygiene` step
//! would look more explicit but cost more than it looks: this crate depends on
//! `rig-core` with `default-features = false`, a different feature resolution
//! from every other build in CI, so cargo would compile rig-core and its whole
//! tree again to typecheck three lines. Under the `--workspace` build, feature
//! unification means it rides along for free.

/// Declare a completion-parent span in each of the macro's two forms, using
/// each of the two portable spellings for "declared, no value yet".
///
/// Nothing is returned and nothing is observed: the point is that this
/// function *compiles*. `pub` keeps it reachable so it cannot be optimized
/// away as dead code.
pub fn declare_completion_parent_spans() {
    let _contextual_parent = rig_runtime_core::telemetry::completion_parent_span!(
        target: "rig_core_macro_hygiene",
        name: "chat",
        operation: "chat",
        system_instructions: Option::<&str>::None,
    );
    let _explicit_parent = rig_runtime_core::telemetry::completion_parent_span!(
        target: "rig_core_macro_hygiene",
        parent: None,
        name: "chat",
        // `Empty` is re-exported by `rig-core` precisely so this line does not
        // need a direct `tracing` dependency; naming it here keeps that
        // re-export inside the tripwire.
        operation: rig_runtime_core::telemetry::Empty,
        system_instructions: Option::<&str>::None,
        // Runtime-specific extras are appended after the named arguments.
        gen_ai.agent.name = "assistant",
    );
}
