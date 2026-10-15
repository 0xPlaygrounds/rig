//! Batch completions through OpenAI's Batch API, recorded end to end: the
//! input file upload, the batch, its polls and its output file.

use super::super::support::with_openai_long_run_cassette;

#[tokio::test]
async fn batch_round_trip() {
    with_openai_long_run_cassette("batch/round_trip", |client, clock| async move {
        let model = client
            .openai
            .completion(rig::providers::openai::GPT_4_1_NANO);
        crate::batch::round_trip(&model, &clock).await;
    })
    .await;
}
