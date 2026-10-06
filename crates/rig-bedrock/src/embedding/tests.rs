use super::*;

/// Answers each text with a two-element vector and a fixed token count,
/// standing in for one `InvokeModel` call per text.
#[derive(Clone)]
struct Replies {
    tokens: usize,
}

impl Transport<Embeddings> for Replies {
    fn send(&self, batch: EmbeddingBatch, _exchange: Exchange) -> Opening<EmbeddingFrame> {
        let frames: Vec<Result<EmbeddingFrame, ProviderError>> = batch
            .texts
            .into_iter()
            .map(|(document, _body)| {
                Ok(EmbeddingFrame::Embedded {
                    document,
                    response: EmbeddingResponse {
                        embedding: vec![0.5, -0.25],
                        input_text_token_count: self.tokens,
                    },
                })
            })
            .collect();
        Opening::ready(Opened::new(futures::stream::iter(frames)))
    }
}

#[tokio::test]
async fn a_batch_bills_input_only_and_its_total_is_input_plus_output() {
    let response = Model::new(
        Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0, Some(2)),
        Replies { tokens: 4 },
    )
    .call(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the replies decode");

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.usage.input_tokens, Some(8));
    assert_eq!(response.usage.output_tokens, Some(0));
    assert_eq!(response.usage.total_tokens, Some(8));
    assert_eq!(
        response.usage.total_tokens,
        response
            .usage
            .input_tokens
            .zip(response.usage.output_tokens)
            .map(|(input, output)| input + output)
    );
}
