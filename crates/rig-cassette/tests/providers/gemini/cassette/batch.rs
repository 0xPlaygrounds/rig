//! Batch completions through Gemini's Batch API, recorded end to end: the
//! inline batch, its polls and the replies it holds once done.

use super::super::support::with_gemini_auto_caching_cassette;

#[tokio::test]
async fn batch_round_trip() {
    with_gemini_auto_caching_cassette("batch/round_trip", |client, clock| async move {
        let model = client.completion(rig::providers::gemini::completion::GEMINI_2_5_FLASH);
        crate::batch::round_trip(&model, &clock).await;
    })
    .await;
}
