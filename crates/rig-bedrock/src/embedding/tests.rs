use super::*;

/// A transport whose one text fails with its error.
#[derive(Clone)]
struct Failing(ProviderError);

impl Transport<Embeddings> for Failing {
    fn send(&self, _batch: EmbeddingBatch, _exchange: Exchange) -> Opening<EmbeddingFrame> {
        let frame = EmbeddingFrame::Failed(self.0.clone());
        Opening::new(async move { Ok(Opened::new(futures::stream::iter([Ok(frame)]))) })
    }
}

/// The batch fails with the failed text's own error, so a throttled call is
/// still retried and a refused one is still final.
#[tokio::test]
async fn a_failed_text_fails_the_batch_with_its_own_error() {
    let throttled = ProviderError::from_provider_body("Too many requests")
        .with_provider_code(Some("ThrottlingException".into()));
    let refused = ProviderError::from_http_response(
        rig_core::http_client::StatusCode::BAD_REQUEST,
        "Malformed input request",
    );
    for (failure, retryable) in [(throttled, true), (refused, false)] {
        let model = Model::new(
            Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0, None),
            Failing(failure.clone()),
        );
        let Err(error) = model.call(vec!["text".to_owned()]).await else {
            panic!("the failed text must fail the batch");
        };
        assert_eq!(error.is_retryable(), retryable, "{error}");
        assert_eq!(error.report(), failure.report());
    }
}
