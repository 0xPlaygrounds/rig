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
//!   and every `xfail` entry names a canonical scenario with a reason.
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
/// per test. There is no `capabilities` block: the flags derive from the
/// fixture's populated optional fields, so they cannot be written out of sync
/// with the frames the suite actually drives.
#[macro_export]
macro_rules! streaming_conformance_suite {
    (
        provider: $family:literal,
        fixture: $fixture:expr
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
            let fixture = suite_fixture();
            let capability = fixture.capabilities().malformed_frame;
            let outcome = $crate::test_utils::streaming_conformance::malformed_frame_surfaces_err_and_terminal_still_completes(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "malformed_frame_surfaces_err_and_terminal_still_completes",
                capability,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn unknown_event_is_skipped() {
            let fixture = suite_fixture();
            let capability = fixture.capabilities().unknown_event_frame;
            let outcome = $crate::test_utils::streaming_conformance::unknown_event_is_skipped(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "unknown_event_is_skipped",
                capability,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn defective_known_event_surfaces_err() {
            let fixture = suite_fixture();
            let capability = fixture.capabilities().defective_known_frame;
            let outcome = $crate::test_utils::streaming_conformance::defective_known_event_surfaces_err(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "defective_known_event_surfaces_err",
                capability,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn delta_less_choice_prelude_is_a_noop() {
            let fixture = suite_fixture();
            let capability = fixture.capabilities().delta_less_prelude;
            let outcome = $crate::test_utils::streaming_conformance::delta_less_choice_prelude_is_a_noop(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "delta_less_choice_prelude_is_a_noop",
                capability,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn refusal_frames_deliver_text_without_error() {
            let fixture = suite_fixture();
            let capability = fixture.capabilities().refusal;
            let outcome = $crate::test_utils::streaming_conformance::refusal_frames_deliver_text_without_error(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "refusal_frames_deliver_text_without_error",
                capability,
                SUITE_XFAIL,
                outcome,
            );
            assert!(verdict.is_ok(), "{}", verdict.err().unwrap_or_default());
        }

        #[tokio::test]
        async fn bare_terminal_after_only_unparseable_frames_fabricates_nothing() {
            let fixture = suite_fixture();
            // Runnable only when the wire spells both a bare terminal and a
            // malformed frame; either gap is a visible skip.
            let capabilities = fixture.capabilities();
            let capability = capabilities.bare_terminal && capabilities.malformed_frame;
            let outcome = $crate::test_utils::streaming_conformance::bare_terminal_after_only_unparseable_frames_fabricates_nothing(&fixture).await;
            let verdict = $crate::test_utils::streaming_conformance::check_gated_outcome(
                "bare_terminal_after_only_unparseable_frames_fabricates_nothing",
                capability,
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
    };
}
