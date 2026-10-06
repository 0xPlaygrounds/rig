use super::join_logs;
use rig_cassette::effect_log::{EffectLog, RecordedStreamError};
use rig_core::{
    effect::{Delivery, DeliveryKind, EffectId},
    error::{ErrorKind, ErrorReport},
};

#[test]
fn cached_anthropic_wire_is_not_rebuilt_without_its_options() {
    use super::super::{Wire, cells::ThinkingWire};
    use rig_core::{providers::anthropic::wire::AnthropicConfig, test_utils::SequencedHttpClient};
    let model = AnthropicConfig::new("local-test-key")
        .connect(rig::http_client::DynHttpClient::new(
            SequencedHttpClient::new(vec![]),
        ))
        .completion("model");
    let wire = Wire {
        model,
        thinking: ThinkingWire::Anthropic,
        route: None,
        temperature: None,
        additional_params: None,
    };
    assert!(
        wire.binding().is_some(),
        "default model options fit the recipe"
    );
    let cached = Wire {
        model: rig::Model::new(wire.model.wire.with_prompt_caching(), wire.model.transport),
        ..wire
    };
    assert!(
        cached.binding().is_none(),
        "cache options require the intact model binding"
    );
}

#[test]
fn model_level_options_require_intact_host_bindings() {
    use super::super::{Wire, cells::ThinkingWire};
    use rig_core::{
        providers::{gemini::GeminiConfig, openai::OpenAIConfig},
        test_utils::SequencedHttpClient,
    };
    fn check<W, T>(model: rig::driver::Model<W, T>, thinking: ThinkingWire)
    where
        W: rig::wire::Wire<Op = rig::operation::Completion>,
        T: rig::driver::Transport<W>,
    {
        let wire = Wire {
            model,
            thinking,
            route: None,
            temperature: None,
            additional_params: None,
        };
        assert!(
            wire.binding().is_none(),
            "provider-only recipes cannot erase model options"
        );
    }
    let http = SequencedHttpClient::new(vec![]);
    let provider = OpenAIConfig::new("local-test-key").connect(http.clone());
    let mut chat = provider.chat("model");
    chat.wire = chat.wire.with_prompt_caching();
    check(chat, ThinkingWire::OpenAiChat);
    let mut responses = provider.responses("model");
    responses.wire = responses.wire.with_strict_tools();
    check(responses, ThinkingWire::OpenAiResponses);
    let mut gemini = GeminiConfig::new("local-test-key")
        .connect(http)
        .completion("model");
    gemini.wire = gemini.wire.with_cached_content("cachedContents/test");
    check(gemini, ThinkingWire::Gemini);
}

#[test]
fn joining_a_restored_tail_keeps_deliveries_and_stream_errors() {
    let mut head = EffectLog::default();
    head.header.deliveries = Some(vec![Delivery {
        batch: 7,
        id: EffectId::from_raw(0),
        kind: DeliveryKind::Outcome,
    }]);
    let mut tail = EffectLog::default();
    tail.header.deliveries = Some(vec![
        Delivery {
            batch: 2,
            id: EffectId::from_raw(1),
            kind: DeliveryKind::Stream { items: 1 },
        },
        Delivery {
            batch: 2,
            id: EffectId::from_raw(2),
            kind: DeliveryKind::Outcome,
        },
        Delivery {
            batch: 3,
            id: EffectId::from_raw(1),
            kind: DeliveryKind::Outcome,
        },
    ]);
    let error = RecordedStreamError {
        item: 0,
        error: ErrorReport::new(ErrorKind::ProviderResponse, "stream error"),
    };
    tail.header
        .stream_errors
        .insert(EffectId::from_raw(1), vec![error.clone()]);
    tail.header
        .delivery_limitations
        .push("unobserved delivery".into());
    let log = join_logs(head, tail);
    let deliveries = log.header.deliveries.expect("both delivery traces");
    assert_eq!(
        deliveries
            .iter()
            .map(|delivery| (delivery.batch, delivery.id.as_u64()))
            .collect::<Vec<_>>(),
        [(7, 0), (9, 1), (9, 2), (10, 1)]
    );
    assert_eq!(log.header.stream_errors[&EffectId::from_raw(1)], [error]);
    assert_eq!(log.header.delivery_limitations, ["unobserved delivery"]);
}
