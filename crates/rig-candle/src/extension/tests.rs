use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use rig_core::completion::{
    CompletionRequest, GenerationOptions, OnUnsupported, ProviderOptions, Reasoning,
};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::wire::{Mode, Wire};
use serde_json::json;

use super::*;
use crate::generation::effective_generation;
use crate::{
    CandleCompletionResponse, CandleFrame, CandleRequest, ConversationProtocol, Generation,
    GenerationConfig, GenerationEvent,
};

fn scripted() -> Generation {
    Generation {
        model: "qwen3-scripted".to_owned(),
        protocol: ConversationProtocol::Qwen3,
    }
}

fn with(options: &CandleOptions) -> CompletionRequest {
    CompletionRequest::new("hello").provider_options(
        ProviderOptions::new()
            .with::<CandleExt>(options)
            .expect("the options serialize"),
    )
}

/// The generation settings `request` gets over the default configuration.
fn settings(request: CompletionRequest) -> Result<GenerationConfig, crate::CandleError> {
    let encoded = scripted()
        .encode(request.clone(), Mode::Unary)
        .expect("the request encodes");
    effective_generation(&request, &encoded.params, &GenerationConfig::default(), 64)
}

#[test]
fn top_k_sets_the_sampler_top_k() {
    let generation = settings(with(&CandleOptions::default().top_k(4))).expect("valid settings");
    assert_eq!(generation.top_k, Some(4));
}

#[test]
fn repeat_penalty_is_applied() {
    let generation =
        settings(with(&CandleOptions::default().repeat_penalty(1.3))).expect("valid settings");
    assert_eq!(generation.repeat_penalty, 1.3);
}

#[test]
fn repeat_last_n_is_applied() {
    let generation =
        settings(with(&CandleOptions::default().repeat_last_n(9))).expect("valid settings");
    assert_eq!(generation.repeat_last_n, 9);
}

/// The options sit beside the mapped `top_p` and `seed`, and a raw key
/// replaces a typed one.
#[test]
fn the_options_join_the_mapped_ones_and_raw_beats_them() {
    let mut request = with(&CandleOptions::default().top_k(4).repeat_last_n(9))
        .options(GenerationOptions::default().top_p(0.7).seed(11));
    let generation = settings(request.clone()).expect("valid settings");
    assert_eq!(
        (generation.top_k, generation.top_p, generation.seed),
        (Some(4), Some(0.7), 11)
    );
    request.additional_params = Some(json!({"top_k": 2}));
    let generation = settings(request).expect("valid settings");
    assert_eq!(generation.top_k, Some(2));
    assert_eq!(generation.repeat_last_n, 9);
}

/// A `top_k` the vocabulary cannot hold is refused as a raw one is.
#[test]
fn an_invalid_top_k_is_refused() {
    assert!(settings(with(&CandleOptions::default().top_k(65))).is_err());
}

/// Every key of `value`, at any depth.
fn keys(value: &Value, out: &mut BTreeSet<String>) {
    if let Value::Object(fields) = value {
        for (key, field) in fields {
            if let Value::Object(_) = field {
                keys(field, out);
            }
            out.insert(key.clone());
        }
    }
}

/// No field writes a key that a mapped option writes.
#[test]
fn no_field_writes_a_reserved_key() {
    let options = ProviderOptions::new()
        .with::<CandleExt>(
            &CandleOptions::default()
                .top_k(4)
                .repeat_penalty(1.3)
                .repeat_last_n(9),
        )
        .expect("the options serialize");
    let mut provider = BTreeSet::new();
    for section in options.get::<CandleExt>().expect("an entry").values() {
        keys(section, &mut provider);
    }
    let request = CompletionRequest::new("hello").options(
        GenerationOptions::default()
            .reasoning(Reasoning::Off)
            .top_p(0.5)
            .seed(1)
            .stop(["END"])
            .on_unsupported(OnUnsupported::Ignore),
    );
    let encoded = scripted()
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let mut owned = BTreeSet::new();
    keys(
        &serde_json::to_value(&encoded.params).expect("the body serializes"),
        &mut owned,
    );
    assert_eq!(
        owned,
        BTreeSet::from(["seed".to_owned(), "top_p".to_owned()])
    );
    assert!(provider.is_disjoint(&owned), "{provider:?} meets {owned:?}");
}

/// Replays scripted generation events as a local generator would send them.
#[derive(Clone)]
struct Scripted(Arc<Mutex<Vec<GenerationEvent>>>);

impl Transport<Generation> for Scripted {
    fn send(&self, _request: CandleRequest, _exchange: Exchange) -> Opening<CandleFrame> {
        let events = match self.0.lock() {
            Ok(mut events) => std::mem::take(&mut *events),
            Err(_) => {
                return Opening::failed(ProviderError::Provider(
                    "the script lock was poisoned".to_owned(),
                ));
            }
        };
        Opening::ready(Opened::new(futures::stream::iter(
            events
                .into_iter()
                .map(|event| Ok(CandleFrame::Event(event))),
        )))
    }
}

/// Not a cassette test: a local generation has no recorded traffic, and
/// none can be made without model weights, so the generator's final record
/// is scripted here and read back off the reply.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn extras_read_the_local_generation_record() {
    use futures::StreamExt;

    let record = CandleCompletionResponse {
        text: "done".to_owned(),
        prompt_tokens: 5,
        generated_tokens: 2,
        requested_max_tokens: 4,
        effective_max_tokens: 3,
        finish_reason: FinishReason::MaxTokens,
        prefill_duration_ms: 8,
        time_to_first_token_ms: Some(10),
        generation_duration_ms: 20,
        tokens_per_second: Some(100.5),
    };
    let events = vec![
        GenerationEvent::Text("done".to_owned()),
        GenerationEvent::Final(record),
    ];
    let mut stream = rig_core::Model::new(scripted(), Scripted(Arc::new(Mutex::new(events))))
        .stream(CompletionRequest::new("hello"))
        .expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("a stream item");
    }
    let response = stream.finish().await.expect("the stream finishes");
    let extras = response
        .extras::<CandleExt>()
        .expect("a Candle reply")
        .expect("the extras read");
    assert_eq!(
        extras,
        CandleExtras {
            text: Some("done".to_owned()),
            prompt_tokens: Some(5),
            generated_tokens: Some(2),
            requested_max_tokens: Some(4),
            effective_max_tokens: Some(3),
            finish_reason: Some(FinishReason::MaxTokens),
            prefill_duration_ms: Some(8),
            time_to_first_token_ms: Some(10),
            generation_duration_ms: Some(20),
            tokens_per_second: Some(100.5),
        }
    );
}
