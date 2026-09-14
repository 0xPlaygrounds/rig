use std::collections::HashSet;

use super::*;

#[test]
fn supported_profiles_centralize_backend_format_and_context() -> Result<(), CandleError> {
    let llama = definition_for(ConversationProtocol::Llama3, ArtifactFormat::Safetensors)?;
    assert_eq!(llama.loader, LoaderBackend::LlamaSafetensors);
    assert_eq!(llama.architecture, ModelArchitecture::Llama);
    assert_eq!(llama.quantization, None);
    assert_eq!(llama.context_limit_cap, None);
    assert_eq!(
        llama.tokenizer_vocabulary,
        TokenizerVocabulary::ModelCapacity
    );
    assert_eq!(llama.start_token, BEGIN_OF_TEXT);
    assert_eq!(llama.end_token, END_OF_TURN);
    assert!(llama.gguf.is_none());

    let smol = definition_for(ConversationProtocol::SmolLm2, ArtifactFormat::Gguf)?;
    assert_eq!(smol.loader, LoaderBackend::LlamaGguf);
    assert_eq!(smol.quantization, Some(Quantization::Q4K));
    let smol = ValidatedProfile::new(smol, 49_152, 8192, HashSet::from([1]))?;
    assert_eq!(smol.context_limit, 4096);
    assert_eq!(
        smol.definition.config_requirements.rope_theta,
        Some(100_000.0)
    );
    assert_eq!(smol.definition.end_token, IM_END);
    assert!(smol.definition.gguf.as_ref().is_some_and(|requirements| {
        requirements
            .allowed_tensor_dtypes
            .contains(&GgmlDType::Q5_0)
    }));

    let qwen = definition_for(ConversationProtocol::Qwen3, ArtifactFormat::Gguf)?;
    assert_eq!(qwen.loader, LoaderBackend::Qwen3Gguf);
    assert_eq!(qwen.architecture, ModelArchitecture::Qwen3);
    assert_eq!(qwen.quantization, Some(Quantization::Q4K));
    let qwen = ValidatedProfile::new(qwen, 151_936, 40_960, HashSet::from([151_645]))?;
    assert_eq!(qwen.context_limit, 4096);
    assert_eq!(
        qwen.definition.tokenizer_vocabulary,
        TokenizerVocabulary::Exact(151_669)
    );
    assert_eq!(
        qwen.definition.config_requirements.eos_token_id,
        Some(151_645)
    );
    assert!(
        qwen.definition
            .gguf
            .as_ref()
            .is_some_and(|requirements| requirements.tensors_per_layer == Some(11)
                && requirements.chat_template_markers.contains(&"<tool_call>"))
    );
    Ok(())
}

#[test]
fn profiles_reject_unsupported_artifact_combinations_and_empty_stops() {
    assert!(matches!(
        definition_for(ConversationProtocol::SmolLm2, ArtifactFormat::Safetensors),
        Err(CandleError::UnsupportedModelFamily(_))
    ));
    assert!(matches!(
        definition_for(ConversationProtocol::Qwen3, ArtifactFormat::Safetensors),
        Err(CandleError::UnsupportedModelFamily(_))
    ));
    assert!(matches!(
        definition_for(ConversationProtocol::Llama3, ArtifactFormat::Gguf),
        Err(CandleError::UnsupportedModelFamily(_))
    ));
    assert!(matches!(
        ValidatedProfile::new(&LLAMA3_PROFILE, 8, 16, HashSet::new()),
        Err(CandleError::MissingStopToken)
    ));
}
