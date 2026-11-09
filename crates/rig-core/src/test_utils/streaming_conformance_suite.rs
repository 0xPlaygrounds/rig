//! The declarative per-wire-family conformance suite macro.
//!
//! [`streaming_conformance_suite!`](crate::streaming_conformance_suite)
//! expands the canonical scenario set for one wire family, one named
//! `#[tokio::test]` per scenario, plus `suite_is_complete`, which checks that
//! the expanded list equals
//! [`CANONICAL_SCENARIOS`](crate::test_utils::streaming_conformance::CANONICAL_SCENARIOS)
//! and that every `xfail` entry names a canonical scenario with a reason. The
//! list it checks is generated from the same entries as the test functions,
//! so deleting a scenario fails it.
//!
//! Each gated test derives its capability flags from the fixture
//! ([`ProviderWireFixture::capabilities`](crate::test_utils::streaming_conformance::ProviderWireFixture::capabilities)).
//! A scenario that skips while its capability is declared fails, and so does
//! one that runs while the capability is disclaimed. The registry test in
//! `tests/core/streaming_conformance_registry.rs` fails when a
//! [`WIRE_FAMILIES`](crate::test_utils::streaming_conformance::WIRE_FAMILIES)
//! entry has no invocation.

/// One scenario entry of [`__streaming_conformance_scenarios`].
///
/// `ungated <name>` drives the scenario unconditionally; `gated <name> =>
/// <field> && <field>…` reads the gate off the fixture-derived
/// [`SuiteCapabilities`](crate::test_utils::streaming_conformance::SuiteCapabilities)
/// fields it names. `$name` is both the emitted `fn` name and the scenario
/// label passed to the outcome checkers.
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

/// Expand a scenario list into the suite's test functions and the
/// `EMITTED_SCENARIOS` const that `suite_is_complete` checks. Each name is
/// written once and expands into both, so a removed scenario leaves the
/// completeness list too and the check fails instead of agreeing with itself.
/// Entries stay in canonical order, since `suite_is_complete` compares the
/// lists with `assert_eq!`.
#[doc(hidden)]
#[macro_export]
macro_rules! __streaming_conformance_scenarios {
    ( $( $kind:ident $name:ident $(=> $($field:ident)&&+)? ; )+ ) => {
        $(
            $crate::__streaming_conformance_scenario!($kind $name $(=> $($field)&&+)?);
        )+

        /// The scenarios this suite expanded, from the same entries as the
        /// test functions above.
        const EMITTED_SCENARIOS: &[&str] = &[ $( stringify!($name) ),+ ];
    };
}

/// Expand the canonical wire-conformance suite for one wire family.
///
/// Invoke inside a dedicated module, since the test names are fixed:
///
/// ```ignore
/// mod anthropic_suite {
///     use super::*;
///
///     rig_core::streaming_conformance_suite! {
///         provider: "anthropic",
///         fixture: anthropic::fixture(),
///         manifest: [partial_tool_args, malformed_frame, unknown_event_frame],
///         xfail: ["unknown_event_is_skipped: warn-skip pending a fix"],
///     }
/// }
/// ```
///
/// `provider` is the wire-family name from
/// [`WIRE_FAMILIES`](crate::test_utils::streaming_conformance::WIRE_FAMILIES).
/// `fixture` is an expression producing a fresh
/// [`ProviderWireFixture`](crate::test_utils::streaming_conformance::ProviderWireFixture)
/// per test. `manifest` is the required snapshot of the fixture-derived
/// capability set, so losing a fixture sample fails
/// `derived_capabilities_match_the_manifest`. `xfail` lists sanctioned known
/// failures as `"<scenario>: <reason>"`.
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

        /// The fixture-derived capability set equals the manifest, so losing
        /// a fixture sample cannot turn a scenario into a passing named skip.
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

        // The canonical scenario set, in canonical order.
        $crate::__streaming_conformance_scenarios! {
            ungated truncation_preserves_content_without_terminal;
            ungated transport_error_after_tool_call_yields_err_then_end;
            gated malformed_frame_ends_the_reply => malformed_frame;
            gated unknown_event_is_skipped => unknown_event_frame;
            gated defective_known_event_ends_the_reply => defective_known_frame;
            gated delta_less_choice_prelude_is_a_noop => delta_less_prelude;
            gated refusal_frames_deliver_text_without_error => refusal;
            gated bare_terminal_after_only_unparseable_frames_fabricates_nothing
                => bare_terminal && malformed_frame;
            ungated usage_variants_are_reported_or_absent;
            gated interleaved_constant_id_reasoning_preserves_order => interleaved_reasoning;
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
