//! The declarative per-wire-family conformance suite macro.
//!
//! [`streaming_conformance_suite!`](crate::streaming_conformance_suite)
//! expands the full canonical scenario set for one wire family — one named
//! `#[tokio::test]` per scenario — plus an anti-tamper test (the langchain
//! `standard-tests` precedent: capability flags gate skips inside test
//! bodies, never test deletion, and an inherited completeness check proves no
//! scenario was dropped):
//!
//! - `suite_is_complete`: the expanded scenario list equals
//!   [`CANONICAL_SCENARIOS`](crate::test_utils::streaming_conformance::CANONICAL_SCENARIOS),
//!   and every `xfail` entry names a canonical scenario with a reason. The
//!   list it checks (`EMITTED_SCENARIOS`) is *generated from the same entries
//!   as the test functions* by
//!   [`__streaming_conformance_scenarios`](crate::__streaming_conformance_scenarios),
//!   so it cannot be a hand-written twin that agrees with itself: deleting a
//!   scenario deletes it from both sides and the check fails (#2258 G6).
//!
//! Capability flags are not written in the invocation at all: each gated test
//! derives them from the fixture itself
//! ([`ProviderWireFixture::capabilities`](crate::test_utils::streaming_conformance::ProviderWireFixture::capabilities)),
//! so a flag structurally cannot drift from the wire fixture that backs it —
//! the shapes the fixture supplies *are* the declared capability set.
//!
//! Capability-gated scenarios expand to *visible named skips*, never absence
//! and never a vacuous pass: a scenario that skips while its capability is
//! declared fails, and one that runs while the capability is disclaimed also
//! fails (#2258 review, F8 corpus-honesty batch).
//!
//! The workspace registry test (`all_wire_families_have_conformance_suites`
//! in `tests/core/streaming_conformance_registry.rs`) enumerates
//! [`WIRE_FAMILIES`](crate::test_utils::streaming_conformance::WIRE_FAMILIES)
//! and fails CI when any family lacks an invocation naming it.

/// Expand the canonical wire-conformance suite for one wire family.
///
/// Invoke inside a dedicated module (the test names are fixed):
///
/// ```ignore
/// mod anthropic_suite {
///     use super::*;
///
///     rig_core::streaming_conformance_suite! {
///         provider: "anthropic",
///         fixture: anthropic::fixture(),
///         // Snapshot of the fixture-derived capability set (required); the
///         // `derived_capabilities_match_the_manifest` test keeps it honest.
///         manifest: [partial_tool_args, malformed_frame, unknown_event_frame],
///         // Sanctioned known failures only, each with a finding reference.
///         xfail: ["unknown_event_is_skipped: warn-skip pending F2 (#2258)"],
///     }
/// }
/// ```
///
/// `provider` is the wire-family name from
/// [`WIRE_FAMILIES`](crate::test_utils::streaming_conformance::WIRE_FAMILIES);
/// the workspace registry test matches invocations on it. `fixture` is an
/// expression producing a fresh
/// [`ProviderWireFixture`](crate::test_utils::streaming_conformance::ProviderWireFixture)
/// per test. The gating flags themselves derive from the fixture's populated
/// optional fields, so they cannot be written out of sync with the frames the
/// suite actually drives; `manifest:` is a required snapshot of that derived
/// set, so losing a fixture sample (and its scenario) is a loud diff rather
/// than a silent skip.
/// One scenario entry of [`__streaming_conformance_scenarios`].
///
/// `ungated <name>` drives the scenario unconditionally; `gated <name> =>
/// <field> && <field>…` reads the gate off the fixture-derived
/// [`SuiteCapabilities`](crate::test_utils::streaming_conformance::SuiteCapabilities)
/// fields it names, so the gate is written as data instead of hand-copied test
/// bodies.
///
/// `$name` is used *both* as the emitted `fn` name and (via `stringify!`) as
/// the scenario label passed to the outcome checkers, so the label cannot drift
/// from the function or from the shared scenario fn in
/// `crate::test_utils::streaming_conformance` that it calls.
#[doc(hidden)]
#[macro_export]
macro_rules! __streaming_conformance_scenario {
    (ungated $name:ident) => {
        #[tokio::test]
        async fn $name() {
            let result = $crate::test_utils::streaming_conformance::$name(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_ungated_outcome(
                stringify!($name),
                SUITE_XFAIL,
                result,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }
    };
    (gated $name:ident => $($field:ident)&&+) => {
        #[tokio::test]
        async fn $name() {
            let fixture = suite_fixture();
            let capabilities = fixture.capabilities();
            // Naming two fields means the scenario needs BOTH shapes to be
            // spellable on this wire; either gap is a visible named skip.
            let capability = $(capabilities.$field)&&+;
            let outcome =
                $crate::test_utils::streaming_conformance::$name(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                stringify!($name),
                capability,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }
    };
}

/// Expand a scenario list into the suite's test functions AND the
/// `EMITTED_SCENARIOS` const that `suite_is_complete` checks.
///
/// This exists to make `suite_is_complete` structural (#2258 G6). It used to
/// compare two hand-written lists — the nine `#[tokio::test] fn`s and a
/// literal `EMITTED_SCENARIOS` array — neither derived from the other, so
/// deleting a scenario meant deleting its name from both and the "anti-tamper"
/// test passed. Now each name is written ONCE and expands into both, so a
/// removed scenario disappears from `EMITTED_SCENARIOS` too and the comparison
/// against
/// [`CANONICAL_SCENARIOS`](crate::test_utils::streaming_conformance::CANONICAL_SCENARIOS)
/// fails.
///
/// The list is order-sensitive: `suite_is_complete` compares it to
/// `CANONICAL_SCENARIOS` with `assert_eq!`, so entries stay in canonical order
/// and gated/ungated entries interleave.
#[doc(hidden)]
#[macro_export]
macro_rules! __streaming_conformance_scenarios {
    ( $( $kind:ident $name:ident $(=> $($field:ident)&&+)? ; )+ ) => {
        $(
            $crate::__streaming_conformance_scenario!($kind $name $(=> $($field)&&+)?);
        )+

        /// The scenarios this suite actually expanded, generated from the same
        /// entries as the test functions above — not a parallel hand-written
        /// list.
        const EMITTED_SCENARIOS: &[&str] = &[ $( stringify!($name) ),+ ];
    };
}

#[macro_export]
macro_rules! streaming_conformance_suite {
    (
        provider: $family:literal,
        fixture: $fixture:expr,
        manifest: [$($cap:ident),* $(,)?]
        $(, xfail: [$($xfail:literal),* $(,)?])?
        $(,)?
    ) => {
        /// Wire family this suite covers; the workspace registry test keys
        /// invocations on the `provider:` field naming it.
        #[allow(dead_code)]
        pub const WIRE_FAMILY: &str = $family;

        const SUITE_XFAIL: &[&str] = &[$($($xfail),*)?];

        fn suite_fixture() -> $crate::test_utils::streaming_conformance::ProviderWireFixture {
            $fixture
        }

        /// Snapshot of the fixture-derived capability set. Flags still derive
        /// from the fixture (drift between flags and fixture is structurally
        /// impossible); this manifest exists so a capability LOSS is loud —
        /// dropping a fixture sample cannot silently turn a scenario into a
        /// passing named skip. Editing the fixture's shape requires editing
        /// this list, restoring the two-sided review diff.
        #[test]
        fn derived_capabilities_match_the_manifest() {
            let derived = suite_fixture().capabilities();
            let expected = match $crate::test_utils::streaming_conformance::SuiteCapabilities::from_names(&[
                $(stringify!($cap)),*
            ]) {
                Ok(expected) => expected,
                Err(error) => panic!("{error}"),
            };
            assert_eq!(
                derived, expected,
                "fixture-derived capabilities changed; update this suite's `manifest:` \
                 to acknowledge the new coverage set"
            );
        }

        // The canonical scenario set, in canonical order. Each entry expands
        // ONCE into both its `#[tokio::test] fn` and the `EMITTED_SCENARIOS`
        // const that `suite_is_complete` below compares against
        // `CANONICAL_SCENARIOS` — deleting a scenario here deletes it from the
        // completeness list too, so the check fails instead of agreeing with
        // itself (#2258 G6).
        $crate::__streaming_conformance_scenarios! {
            ungated truncation_preserves_content_without_terminal;
            ungated transport_error_after_tool_call_yields_err_then_end;
            gated malformed_frame_surfaces_err_and_terminal_still_completes => malformed_frame;
            gated unknown_event_is_skipped => unknown_event_frame;
            gated defective_known_event_surfaces_err => defective_known_frame;
            gated delta_less_choice_prelude_is_a_noop => delta_less_prelude;
            gated refusal_frames_deliver_text_without_error => refusal;
            gated bare_terminal_after_only_unparseable_frames_fabricates_nothing
                => bare_terminal && malformed_frame;
            ungated usage_variants_are_reported_or_zero_sentinel;
        }

        /// Anti-tamper: the expanded suite covers exactly the canonical
        /// scenario list, and every `xfail` entry is well-formed.
        #[test]
        fn suite_is_complete() {
            assert_eq!(
                EMITTED_SCENARIOS,
                $crate::test_utils::streaming_conformance::CANONICAL_SCENARIOS,
                "the {} suite's expanded scenarios diverge from the canonical list",
                $family,
            );
            let invalid = $crate::test_utils::streaming_conformance::invalid_xfail_entries(SUITE_XFAIL);
            assert!(
                invalid.is_empty(),
                "xfail entries must name a canonical scenario and carry a finding reference: {invalid:?}",
            );
        }
    };
}
