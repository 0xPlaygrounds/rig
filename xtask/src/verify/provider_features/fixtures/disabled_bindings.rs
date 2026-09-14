use rig_ecs::bus::binding::{MaterializeError, Materializer, ProviderBinding, ProviderKind};

fn main() {
    let materializer = Materializer::new(
        |_| panic!("disabled providers must not resolve credentials"),
        || panic!("disabled providers must not construct transports"),
    )
    .serving(|_| panic!("disabled providers must not invoke the serving wrapper"));
    for (kind, feature) in [
        (ProviderKind::Anthropic, "anthropic"),
        (ProviderKind::OpenAiChat, "openai"),
        (ProviderKind::OpenAiResponses, "openai"),
        (ProviderKind::Gemini, "gemini"),
        (ProviderKind::DeepSeek, "deepseek"),
    ] {
        assert_eq!(kind.is_enabled(), SELECTED.contains(&feature));
        if kind.is_enabled() {
            continue;
        }
        let binding = ProviderBinding::new("disabled-provider", kind, "model", "secret-ref");
        match materializer.build(&binding) {
            Err(MaterializeError::ProviderDisabled { key, kind: actual }) => {
                assert_eq!(key, binding.key);
                assert_eq!(actual, kind);
            }
            _ => panic!("expected ProviderDisabled for {kind}"),
        }
    }
}
