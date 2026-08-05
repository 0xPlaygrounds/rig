//! The declarative per-wire-family conformance suite macro.
//!
//! [`streaming_conformance_suite!`](crate::streaming_conformance_suite)
//! expands the full canonical scenario set for one wire family — one named
//! `#[tokio::test]` per scenario — plus two anti-tamper tests (the langchain
//! `standard-tests` precedent: capability flags gate skips inside test
//! bodies, never test deletion, and an inherited completeness check proves no
//! scenario was dropped):
//!
//! - `suite_is_complete`: the expanded scenario list equals
//!   [`CANONICAL_SCENARIOS`](crate::test_utils::streaming_conformance::CANONICAL_SCENARIOS),
//!   and every `xfail` entry names a canonical scenario with a reason.
//! - `capabilities_match_fixture`: the declared capability flags agree with
//!   the optional shapes the fixture actually supplies, so a flag cannot
//!   drift from the wire.
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
///         capabilities: {
///             partial_tool_args: true,
///             zero_usage_terminal: false,
///             bare_terminal: true,
///             malformed_frame: true,
///             unknown_event_frame: true,
///             defective_known_frame: true,
///             delta_less_prelude: false,
///             refusal: false,
///         },
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
/// per test. The capability flags are all mandatory, in this order.
#[macro_export]
macro_rules! streaming_conformance_suite {
    (
        provider: $family:literal,
        fixture: $fixture:expr,
        capabilities: {
            partial_tool_args: $partial_tool_args:literal,
            zero_usage_terminal: $zero_usage_terminal:literal,
            bare_terminal: $bare_terminal:literal,
            malformed_frame: $malformed_frame:literal,
            unknown_event_frame: $unknown_event_frame:literal,
            defective_known_frame: $defective_known_frame:literal,
            delta_less_prelude: $delta_less_prelude:literal,
            refusal: $refusal:literal $(,)?
        }
        $(, xfail: [$($xfail:literal),* $(,)?])?
        $(,)?
    ) => {
        /// Wire family this suite covers; the workspace registry test keys
        /// invocations on the `provider:` field naming it.
        #[allow(dead_code)]
        pub const WIRE_FAMILY: &str = $family;

        const SUITE_CAPABILITIES:
            $crate::test_utils::streaming_conformance::SuiteCapabilities =
            $crate::test_utils::streaming_conformance::SuiteCapabilities {
                partial_tool_args: $partial_tool_args,
                zero_usage_terminal: $zero_usage_terminal,
                bare_terminal: $bare_terminal,
                malformed_frame: $malformed_frame,
                unknown_event_frame: $unknown_event_frame,
                defective_known_frame: $defective_known_frame,
                delta_less_prelude: $delta_less_prelude,
                refusal: $refusal,
            };

        const SUITE_XFAIL: &[&str] = &[$($($xfail),*)?];

        fn suite_fixture() -> $crate::test_utils::streaming_conformance::ProviderWireFixture {
            $fixture
        }

        #[tokio::test]
        async fn truncation_preserves_content_without_terminal() {
            let result = $crate::test_utils::streaming_conformance::truncation_preserves_content_without_terminal(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_ungated_outcome(
                "truncation_preserves_content_without_terminal",
                SUITE_XFAIL,
                result,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn transport_error_after_tool_call_yields_err_then_end() {
            let result = $crate::test_utils::streaming_conformance::transport_error_after_tool_call_yields_err_then_end(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_ungated_outcome(
                "transport_error_after_tool_call_yields_err_then_end",
                SUITE_XFAIL,
                result,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn malformed_frame_surfaces_err_and_terminal_still_completes() {
            let outcome = $crate::test_utils::streaming_conformance::malformed_frame_surfaces_err_and_terminal_still_completes(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "malformed_frame_surfaces_err_and_terminal_still_completes",
                SUITE_CAPABILITIES.malformed_frame,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn unknown_event_is_skipped() {
            let outcome = $crate::test_utils::streaming_conformance::unknown_event_is_skipped(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "unknown_event_is_skipped",
                SUITE_CAPABILITIES.unknown_event_frame,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn defective_known_event_surfaces_err() {
            let outcome = $crate::test_utils::streaming_conformance::defective_known_event_surfaces_err(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "defective_known_event_surfaces_err",
                SUITE_CAPABILITIES.defective_known_frame,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn delta_less_choice_prelude_is_a_noop() {
            let outcome = $crate::test_utils::streaming_conformance::delta_less_choice_prelude_is_a_noop(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "delta_less_choice_prelude_is_a_noop",
                SUITE_CAPABILITIES.delta_less_prelude,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn refusal_frames_deliver_text_without_error() {
            let outcome = $crate::test_utils::streaming_conformance::refusal_frames_deliver_text_without_error(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "refusal_frames_deliver_text_without_error",
                SUITE_CAPABILITIES.refusal,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn bare_terminal_after_only_unparseable_frames_fabricates_nothing() {
            let outcome = $crate::test_utils::streaming_conformance::bare_terminal_after_only_unparseable_frames_fabricates_nothing(&suite_fixture()).await;
            // Runnable only when the wire spells both a bare terminal and a
            // malformed frame; either gap is a visible skip.
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "bare_terminal_after_only_unparseable_frames_fabricates_nothing",
                SUITE_CAPABILITIES.bare_terminal && SUITE_CAPABILITIES.malformed_frame,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn usage_variants_are_reported_or_zero_sentinel() {
            let result = $crate::test_utils::streaming_conformance::usage_variants_are_reported_or_zero_sentinel(&suite_fixture()).await;
            let verdict = $crate::test_utils::streaming_conformance::check_ungated_outcome(
                "usage_variants_are_reported_or_zero_sentinel",
                SUITE_XFAIL,
                result,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        /// Anti-tamper: the expanded suite covers exactly the canonical
        /// scenario list, and every `xfail` entry is well-formed.
        #[test]
        fn suite_is_complete() {
            const EMITTED_SCENARIOS: &[&str] = &[
                "truncation_preserves_content_without_terminal",
                "transport_error_after_tool_call_yields_err_then_end",
                "malformed_frame_surfaces_err_and_terminal_still_completes",
                "unknown_event_is_skipped",
                "defective_known_event_surfaces_err",
                "delta_less_choice_prelude_is_a_noop",
                "refusal_frames_deliver_text_without_error",
                "bare_terminal_after_only_unparseable_frames_fabricates_nothing",
                "usage_variants_are_reported_or_zero_sentinel",
            ];
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

        /// Anti-drift: the declared capability flags agree with the optional
        /// shapes the fixture supplies.
        #[test]
        fn capabilities_match_fixture() {
            let mismatches = $crate::test_utils::streaming_conformance::capability_fixture_mismatches(
                &suite_fixture(),
                &SUITE_CAPABILITIES,
            );
            assert!(
                mismatches.is_empty(),
                "capability flags must match the fixture's optional shapes: {mismatches:?}",
            );
        }
    };
}
