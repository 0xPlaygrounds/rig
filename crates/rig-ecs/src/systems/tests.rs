use rig_core::completion::FinishReason;
use rig_core::error::{ProviderError, RigError};

use super::no_answer;

/// A truncated turn with no answer reports exactly what the provider error of
/// the same kind reports, as the classic agent's does.
#[test]
fn a_turn_with_no_answer_reports_as_the_classic_agent_does() {
    for reason in [FinishReason::Length, FinishReason::ContentFilter] {
        assert_eq!(
            no_answer(&reason),
            RigError::from(ProviderError::Response(reason.no_answer_message()))
        );
    }
}
