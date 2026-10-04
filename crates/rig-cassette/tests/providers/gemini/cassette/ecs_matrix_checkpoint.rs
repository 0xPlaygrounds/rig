//! Focused tool-turn checkpoint matrix on Gemini: gemini-2.5-flash-lite.
//! One producer recording is reused by every native cut with strict matching.

use crate::ecs_matrix::checkpoint;

/// Negative matcher evidence only; fresh-world continuation uses the native tests above.
#[tokio::test]
async fn large_result_last_byte_mismatch() {
    checkpoint::assert_large_request_rejected("gemini", "checkpoint_matrix/large_result").await;
}
