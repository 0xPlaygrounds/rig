use super::*;
use rig_core::driver::Model;
use rig_core::error::ErrorKind;
use std::sync::{Arc, Mutex};

/// Records every request it is sent and answers each with a one-value vector.
#[derive(Clone, Default)]
struct Recording(Arc<Mutex<Vec<EmbedContentRequest>>>);

impl Recording {
    fn sent(&self) -> Vec<EmbedContentRequest> {
        self.0.lock().expect("record lock").clone()
    }
}

impl Transport<Embeddings> for Recording {
    fn send(
        &self,
        requests: Vec<(String, EmbedContentRequest)>,
        _exchange: Exchange,
    ) -> Opening<(String, proto::EmbedContentResponse)> {
        let mut sent = self.0.lock().expect("record lock");
        let frames = requests
            .into_iter()
            .map(|(text, request)| {
                sent.push(request);
                let response = proto::EmbedContentResponse {
                    embedding: Some(proto::ContentEmbedding { values: vec![1.0] }),
                };
                Ok((text, response))
            })
            .collect::<Vec<_>>();
        Opening::ready(Opened::new(futures::stream::iter(frames)))
    }
}

/// A width the request field cannot hold fails locally and sends nothing; a
/// width it can hold reaches the request unchanged.
#[test]
fn the_width_reaches_the_request_only_when_it_fits() {
    let transport = Recording::default();
    let too_wide = Model::new(
        Embeddings::new(EMBEDDING_001, Some(i32::MAX as usize + 1)),
        transport.clone(),
    );
    let error = futures::executor::block_on(too_wide.embed_text("hello"))
        .expect_err("the width does not fit an i32");
    assert_eq!(error.kind(), ErrorKind::Request);
    assert!(transport.sent().is_empty());

    let narrow = Model::new(Embeddings::new(EMBEDDING_001, Some(256)), transport.clone());
    futures::executor::block_on(narrow.embed_text("hello")).expect("the width fits");
    let widths = transport
        .sent()
        .iter()
        .map(|request| request.output_dimensionality)
        .collect::<Vec<_>>();
    assert_eq!(widths, vec![Some(256)]);
}
