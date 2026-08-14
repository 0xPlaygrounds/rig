use candle_core::quantized::{GgmlDType, gguf_file};
use candle_transformers::generation::Sampling;
use candle_transformers::models::llama::LlamaConfig;
#[cfg(not(target_family = "wasm"))]
use futures::StreamExt;
use rig_core::completion::{CompletionModel, Document, ToolDefinition};
use rig_core::message::{AudioMediaType, ImageDetail, ImageMediaType, ToolChoice};
#[cfg(not(target_family = "wasm"))]
use safetensors::tensor::{Dtype, View, serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};

use super::*;

#[cfg(not(target_family = "wasm"))]
type ControlledModel = (LlamaModel, Arc<TestControl>, Arc<tokio::sync::Semaphore>);

struct TestTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for TestTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

fn tiny_config() -> Vec<u8> {
    br#"{
        "hidden_size": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 0.00001,
        "max_position_embeddings": 128,
        "bos_token_id": 2,
        "eos_token_id": [1, 3],
        "tie_word_embeddings": false
    }"#
    .to_vec()
}

fn tiny_tokenizer() -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    tiny_tokenizer_with_end_header(END_HEADER, true)
}

fn tiny_tokenizer_with_end_header(
    end_header: &str,
    mark_end_header_special: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let vocab = [
        ("<unk>".to_string(), 0),
        (END_OF_TURN.to_string(), 1),
        (BEGIN_OF_TEXT.to_string(), 2),
        ("<eos>".to_string(), 3),
        (START_HEADER.to_string(), 4),
        (end_header.to_string(), 5),
        ("assistant".to_string(), 6),
        ("hello".to_string(), 7),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    let mut special_tokens = vec![
        AddedToken::from(END_OF_TURN, true),
        AddedToken::from(BEGIN_OF_TEXT, true),
        AddedToken::from(START_HEADER, true),
    ];
    if mark_end_header_special {
        special_tokens.push(AddedToken::from(end_header, true));
    }
    tokenizer.add_special_tokens(&special_tokens);
    Ok(tokenizer.to_string(false)?.into_bytes())
}

fn tiny_smollm2_tokenizer(
    include_end: bool,
    mark_end_special: bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let end = if include_end { IM_END } else { "<other>" };
    let vocab = [
        ("<unk>".to_string(), 0),
        (IM_START.to_string(), 1),
        (end.to_string(), 2),
        ("<eos>".to_string(), 3),
        ("system".to_string(), 4),
        ("user".to_string(), 5),
        ("assistant".to_string(), 6),
        ("hello".to_string(), 7),
    ]
    .into_iter()
    .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".to_string())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    let mut special = vec![AddedToken::from(IM_START, true)];
    if mark_end_special {
        special.push(AddedToken::from(end, true));
    }
    tokenizer.add_special_tokens(&special);
    Ok(tokenizer.to_string(false)?.into_bytes())
}

fn tensor(shape: &[usize]) -> TestTensor {
    tensor_with_dtype(shape, Dtype::F32)
}

fn tensor_with_dtype(shape: &[usize], dtype: Dtype) -> TestTensor {
    let elements = shape.iter().product::<usize>();
    let element_size = match dtype {
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
        _ => 1,
    };
    TestTensor {
        dtype,
        shape: shape.to_vec(),
        bytes: vec![0; elements * element_size],
    }
}

fn checkpoint(include_all: bool) -> Result<Vec<u8>, safetensors::SafeTensorError> {
    checkpoint_custom(include_all, tensor(&[8, 4]), true)
}

fn checkpoint_custom(
    include_all: bool,
    embedding: TestTensor,
    include_lm_head: bool,
) -> Result<Vec<u8>, safetensors::SafeTensorError> {
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), embedding),
        ("model.norm.weight".to_string(), tensor(&[4])),
        (
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            tensor(&[4, 4]),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            tensor(&[4, 4]),
        ),
        (
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            tensor(&[4, 4]),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            tensor(&[4, 4]),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            tensor(&[8, 4]),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".to_string(),
            tensor(&[8, 4]),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".to_string(),
            tensor(&[4, 8]),
        ),
        (
            "model.layers.0.input_layernorm.weight".to_string(),
            tensor(&[4]),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            tensor(&[4]),
        ),
    ];
    if include_lm_head {
        tensors.push(("lm_head.weight".to_string(), tensor(&[8, 4])));
    }
    if !include_all {
        tensors.retain(|(name, _)| name != "model.layers.0.self_attn.q_proj.weight");
    }
    serialize(tensors, None)
}

fn model_data() -> Result<ModelData, Box<dyn std::error::Error + Send + Sync>> {
    Ok(ModelData {
        config: tiny_config(),
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    })
}

fn config_with(
    field: &str,
    value: serde_json::Value,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
    config
        .as_object_mut()
        .ok_or("test config must be a JSON object")?
        .insert(field.to_string(), value);
    Ok(serde_json::to_vec(&config)?)
}

fn request(messages: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: if messages.is_empty() {
            vec![Message::user("hello")]
        } else {
            messages
        },
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[cfg(not(target_family = "wasm"))]
async fn collect_stream(
    model: &LlamaModel,
    request: CompletionRequest,
) -> Result<(String, CandleCompletionResponse), Box<dyn std::error::Error + Send + Sync>> {
    let mut response = model.raw_stream(request).await?;
    let mut text = String::new();
    let mut final_response = None;
    while let Some(item) = response.next().await {
        match item? {
            RawStreamingChoice::Message(fragment) => text.push_str(&fragment),
            RawStreamingChoice::FinalResponse(raw) => final_response = Some(raw),
            _ => {}
        }
    }
    let raw = final_response.ok_or("stream did not emit a final response")?;
    Ok((text, raw))
}

#[cfg(not(target_family = "wasm"))]
fn controlled_model(
    blocked: bool,
    panic_after_gate: bool,
    max_tokens: u64,
) -> Result<ControlledModel, Box<dyn std::error::Error + Send + Sync>> {
    let generation = GenerationConfig {
        temperature: 0.0,
        max_tokens,
        ..GenerationConfig::default()
    };
    let mut loaded = load_model(model_data()?, generation, 1)?;
    let control = Arc::new(TestControl::new(blocked, panic_after_gate));
    let concurrency = Arc::clone(&loaded.concurrency);
    loaded.test_control = Some(Arc::clone(&control));
    Ok((
        LlamaModel {
            state: Arc::new(loaded),
        },
        control,
        concurrency,
    ))
}

#[test]
fn rejects_empty_and_malformed_artifacts() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for (artifact, data) in [
        (
            "config",
            ModelData {
                config: Vec::new(),
                tokenizer: vec![1],
                weights: vec![1],
            },
        ),
        (
            "tokenizer",
            ModelData {
                config: tiny_config(),
                tokenizer: Vec::new(),
                weights: vec![1],
            },
        ),
        (
            "weights",
            ModelData {
                config: tiny_config(),
                tokenizer: tiny_tokenizer()?,
                weights: Vec::new(),
            },
        ),
    ] {
        let error = LlamaModel::from_safetensors(data)
            .err()
            .ok_or("expected empty-buffer error")?;
        assert!(
            matches!(error, CandleError::EmptyBuffer { artifact: actual } if actual == artifact)
        );
    }

    let mut data = model_data()?;
    data.config = b"not json".to_vec();
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::Configuration(_))
    ));
    let mut data = model_data()?;
    data.tokenizer = b"not json".to_vec();
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::TokenizerLoading(_))
    ));
    let mut data = model_data()?;
    data.weights = b"not safetensors".to_vec();
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::InvalidCheckpoint(_))
    ));
    Ok(())
}

#[test]
fn validates_checkpoint_metadata_before_model_loading()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
    let config = config.into_config(false);
    validate_checkpoint(&checkpoint(true)?, &config)?;
    let error = validate_checkpoint(&checkpoint(false)?, &config)
        .err()
        .ok_or("expected missing tensor")?;
    assert!(
        matches!(error, CandleError::MissingTensor(name) if name == "model.layers.0.self_attn.q_proj.weight")
    );
    Ok(())
}

#[test]
fn validates_tensor_shapes_dtypes_and_tied_embeddings()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
    let config = config.into_config(false);

    let shape_error =
        validate_checkpoint(&checkpoint_custom(true, tensor(&[7, 4]), true)?, &config)
            .err()
            .ok_or("expected shape error")?;
    assert!(matches!(
        shape_error,
        CandleError::TensorShapeMismatch { tensor, expected, actual }
            if tensor == "model.embed_tokens.weight"
                && expected == vec![8, 4]
                && actual == vec![7, 4]
    ));

    let dtype_error = validate_checkpoint(
        &checkpoint_custom(true, tensor_with_dtype(&[8, 4], Dtype::U8), true)?,
        &config,
    )
    .err()
    .ok_or("expected dtype error")?;
    assert!(matches!(
        dtype_error,
        CandleError::UnsupportedTensorDtype { tensor, dtype }
            if tensor == "model.embed_tokens.weight" && dtype == "U8"
    ));
    validate_checkpoint(
        &checkpoint_custom(true, tensor_with_dtype(&[8, 4], Dtype::F16), true)?,
        &config,
    )?;
    validate_checkpoint(
        &checkpoint_custom(true, tensor_with_dtype(&[8, 4], Dtype::BF16), true)?,
        &config,
    )?;
    assert!(matches!(
        validate_checkpoint(
            &checkpoint_custom(true, tensor(&[8, 4]), false)?,
            &config
        ),
        Err(CandleError::MissingTensor(name)) if name == "lm_head.weight"
    ));

    let tied_config: LlamaConfig =
        serde_json::from_slice(&config_with("tie_word_embeddings", true.into())?)?;
    let tied_config = tied_config.into_config(false);
    validate_checkpoint(
        &checkpoint_custom(true, tensor(&[8, 4]), false)?,
        &tied_config,
    )?;
    let model = LlamaModel::from_safetensors(ModelData {
        config: config_with("tie_word_embeddings", true.into())?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint_custom(true, tensor(&[8, 4]), false)?,
    })?;
    // The model is only constructible loaded, so assert the loaded profile
    // rather than a state discriminant that no longer exists.
    assert!(model.state.runtime.is_consistent_cpu());
    Ok(())
}

#[test]
fn validates_tokenizer_vocabulary_special_tokens_and_configured_ids()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data = ModelData {
        config: config_with("vocab_size", 9.into())?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    };
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::TokenizerVocabularyMismatch {
            expected: 9,
            actual: 8
        })
    ));

    let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
    let config = config.into_config(false);
    let llama = definition_for(ModelFamily::Llama3, ArtifactFormat::Safetensors)?;
    let smollm2 = definition_for(ModelFamily::SmolLm2, ArtifactFormat::Gguf)?;
    let tokenizer = Tokenizer::from_bytes(tiny_tokenizer_with_end_header("<other>", true)?)?;
    assert!(matches!(
        validate_tokenizer(&config, &tokenizer, llama),
        Err(CandleError::MissingSpecialToken { token: END_HEADER })
    ));
    let tokenizer = Tokenizer::from_bytes(tiny_tokenizer_with_end_header(END_HEADER, false)?)?;
    assert!(matches!(
        validate_tokenizer(&config, &tokenizer, llama),
        Err(CandleError::SpecialTokenNotMarked { token: END_HEADER })
    ));

    let tokenizer = Tokenizer::from_bytes(tiny_smollm2_tokenizer(false, true)?)?;
    assert!(matches!(
        validate_tokenizer(&config, &tokenizer, smollm2),
        Err(CandleError::MissingSpecialToken { token: IM_END })
    ));
    let tokenizer = Tokenizer::from_bytes(tiny_smollm2_tokenizer(true, false)?)?;
    assert!(matches!(
        validate_tokenizer(&config, &tokenizer, smollm2),
        Err(CandleError::SpecialTokenNotMarked { token: IM_END })
    ));

    let data = ModelData {
        config: config_with("bos_token_id", 8.into())?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    };
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::TokenIdOutOfRange { token, id: 8, .. }) if token == "bos_token_id"
    ));
    let data = ModelData {
        config: config_with("eos_token_id", serde_json::json!([1, 9]))?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    };
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::TokenIdOutOfRange { token, id: 9, .. }) if token == "eos_token_id"
    ));
    for (field, value) in [("bos_token_id", 7.into()), ("eos_token_id", 3.into())] {
        let data = ModelData {
            config: config_with(field, value)?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        };
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::ArtifactMismatch { artifact, .. }) if artifact == field
        ));
    }
    let data = ModelData {
        config: config_with("eos_token_id", serde_json::json!([]))?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    };
    assert!(matches!(
        LlamaModel::from_safetensors(data),
        Err(CandleError::InvalidConfigurationValue {
            field: "eos_token_id",
            ..
        })
    ));
    Ok(())
}

#[test]
fn validates_model_dimension_relationships() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
{
    for (field, value) in [
        ("hidden_size", 0),
        ("num_attention_heads", 0),
        ("num_key_value_heads", 0),
        ("max_position_embeddings", 0),
    ] {
        let config: LlamaConfig = serde_json::from_slice(&config_with(field, value.into())?)?;
        assert!(matches!(
            validate_model_config(&config.into_config(false)),
            Err(CandleError::InvalidConfigurationValue { field: actual, .. }) if actual == field
        ));
    }
    let config: LlamaConfig =
        serde_json::from_slice(&config_with("num_attention_heads", 3.into())?)?;
    assert!(matches!(
        validate_model_config(&config.into_config(false)),
        Err(CandleError::InvalidConfigurationValue {
            field: "hidden_size",
            ..
        })
    ));
    let mut odd_head_config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
    let object = odd_head_config
        .as_object_mut()
        .ok_or("test config must be a JSON object")?;
    object.insert("hidden_size".to_string(), 6.into());
    object.insert("num_attention_heads".to_string(), 2.into());
    let config: LlamaConfig = serde_json::from_value(odd_head_config)?;
    assert!(matches!(
        validate_model_config(&config.into_config(false)),
        Err(CandleError::InvalidConfigurationValue {
            field: "hidden_size",
            ..
        })
    ));

    #[cfg(target_pointer_width = "64")]
    {
        let oversized_context = u64::from(u32::MAX) + 1;
        let config: LlamaConfig = serde_json::from_slice(&config_with(
            "max_position_embeddings",
            oversized_context.into(),
        )?)?;
        assert!(matches!(
            validate_model_config(&config.into_config(false)),
            Err(CandleError::InvalidConfigurationValue {
                field: "max_position_embeddings",
                ..
            })
        ));
    }

    let mut rope_config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
    rope_config
        .as_object_mut()
        .ok_or("test config must be a JSON object")?
        .insert(
            "rope_scaling".to_string(),
            serde_json::json!({
                "factor": 0.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 128,
                "rope_type": "llama3"
            }),
        );
    let config: LlamaConfig = serde_json::from_value(rope_config)?;
    assert!(matches!(
        validate_model_config(&config.into_config(false)),
        Err(CandleError::InvalidConfigurationValue {
            field: "rope_scaling.factor",
            ..
        })
    ));

    let mut rope_config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
    rope_config
        .as_object_mut()
        .ok_or("test config must be a JSON object")?
        .insert(
            "rope_scaling".to_string(),
            serde_json::json!({
                "factor": 8.0,
                "low_freq_factor": 4.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 128,
                "rope_type": "llama3"
            }),
        );
    let config: LlamaConfig = serde_json::from_value(rope_config)?;
    assert!(matches!(
        validate_model_config(&config.into_config(false)),
        Err(CandleError::InvalidConfigurationValue {
            field: "rope_scaling.high_freq_factor",
            ..
        })
    ));
    Ok(())
}

#[test]
fn context_limit_boundaries_clamp_and_detect_conversion_overflow() {
    assert!(matches!(
        effective_output_limit(9, 1, 8),
        Err(CandleError::PromptTooLong {
            prompt_tokens: 9,
            context_limit: 8
        })
    ));
    assert!(matches!(
        effective_output_limit(8, 1, 8),
        Err(CandleError::NoGenerationCapacity {
            prompt_tokens: 8,
            context_limit: 8
        })
    ));
    assert!(matches!(effective_output_limit(6, 10, 8), Ok(2)));
    assert!(matches!(effective_output_limit(6, 1, 8), Ok(1)));
    assert!(matches!(
        max_tokens_to_usize(256, 255),
        Err(CandleError::NumericConversion {
            field: "max_tokens",
            value: 256
        })
    ));
}

#[test]
fn loads_entirely_from_owned_bytes() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = LlamaModel::from_safetensors(model_data()?)?;
    let loaded = &model.state;
    assert!(loaded.runtime.is_consistent_cpu());
    assert_eq!(
        loaded.profile.definition.loader,
        LoaderBackend::LlamaSafetensors
    );
    assert_eq!(
        loaded.profile.definition.artifact_format,
        ArtifactFormat::Safetensors
    );
    assert_eq!(model.conversation_protocol(), Some(ModelFamily::Llama3));
    assert_eq!(model.quantization(), None);
    Ok(())
}

#[test]
fn borrowed_gguf_builder_keeps_borrowed_artifacts_and_all_settings() {
    let data = GgufModelData {
        config: b"config",
        tokenizer: b"tokenizer",
        weights: b"weights",
    };
    let builder = CandleModel::builder_from_gguf_bytes(data)
        .conversation_protocol(ModelFamily::SmolLm2)
        .max_tokens(17)
        .temperature(0.25)
        .top_k(Some(4))
        .top_p(None)
        .seed(9)
        .repeat_penalty(1.2)
        .repeat_last_n(11)
        .max_concurrent_requests(3);

    assert!(
        matches!(builder.source, ModelSource::BorrowedGguf(actual) if actual.weights.as_ptr() == data.weights.as_ptr())
    );
    assert_eq!(builder.family, Some(ModelFamily::SmolLm2));
    assert_eq!(builder.generation.max_tokens, 17);
    assert_eq!(builder.generation.temperature, 0.25);
    assert_eq!(builder.generation.top_k, Some(4));
    assert_eq!(builder.generation.top_p, None);
    assert_eq!(builder.generation.seed, 9);
    assert_eq!(builder.generation.repeat_penalty, 1.2);
    assert_eq!(builder.generation.repeat_last_n, 11);
    assert_eq!(builder.max_concurrent_requests, 3);
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn async_loading_succeeds_and_preserves_builder_settings()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let direct = CandleModel::from_safetensors_async(model_data()?).await?;
    assert_eq!(direct.conversation_protocol(), Some(ModelFamily::Llama3));

    let configured = CandleModel::builder(model_data()?)
        .max_tokens(17)
        .temperature(0.25)
        .top_k(Some(4))
        .top_p(None)
        .seed(9)
        .repeat_penalty(1.2)
        .repeat_last_n(11)
        .max_concurrent_requests(3)
        .build_async()
        .await?;
    let loaded = &configured.state;
    assert_eq!(loaded.generation.max_tokens, 17);
    assert_eq!(loaded.generation.temperature, 0.25);
    assert_eq!(loaded.generation.top_k, Some(4));
    assert_eq!(loaded.generation.top_p, None);
    assert_eq!(loaded.generation.seed, 9);
    assert_eq!(loaded.generation.repeat_penalty, 1.2);
    assert_eq!(loaded.generation.repeat_last_n, 11);
    assert_eq!(loaded.concurrency.available_permits(), 3);
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn async_loading_preserves_typed_errors_and_converts_panics() {
    let invalid = CandleModel::from_safetensors_async(ModelData {
        config: b"not json".to_vec(),
        tokenizer: b"tokenizer".to_vec(),
        weights: b"weights".to_vec(),
    })
    .await;
    assert!(matches!(invalid, Err(CandleError::Configuration(_))));

    let panicked = join_model_load(tokio::task::spawn_blocking(|| {
        std::panic::resume_unwind(Box::new("intentional async-loading test panic"));
        // Never evaluated — the closure panics above. It exists only to pin the
        // return type that `join_model_load` expects.
        #[allow(unreachable_code)]
        Err::<CandleModel, CandleError>(CandleError::Configuration(
            "unreachable: the closure panics above".to_owned(),
        ))
    }))
    .await;
    assert!(matches!(panicked, Err(CandleError::BlockingTaskJoin(_))));
}

#[test]
fn typed_gguf_and_family_errors_preserve_the_failure_kind()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data = model_data()?;
    assert!(matches!(
        LlamaModel::builder(data)
            .conversation_protocol(ModelFamily::SmolLm2)
            .build(),
        Err(CandleError::ModelFamilyMismatch {
            selected: ModelFamily::SmolLm2,
            detected: ModelFamily::Llama3,
        })
    ));

    assert!(matches!(
        LlamaModel::from_gguf(model_data()?),
        Err(CandleError::UnsupportedModelFamily(_))
    ));
    Ok(())
}

#[test]
fn gguf_metadata_shapes_and_tensor_encodings_are_validated_before_loading()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
    let config = config.into_config(false);
    let mut content = gguf_file::Content {
        magic: gguf_file::VersionedMagic::GgufV3,
        metadata: HashMap::from([("llama.vocab_size".to_string(), gguf_file::Value::U32(9))]),
        tensor_infos: HashMap::new(),
        tensor_data_offset: 0,
    };
    let tokenizer = Tokenizer::from_bytes(tiny_tokenizer()?)?;
    let definition = definition_for(ModelFamily::SmolLm2, ArtifactFormat::Gguf)?;
    assert!(matches!(
        validate_gguf_metadata(&content, &config, &tokenizer, definition),
        Err(CandleError::ArtifactMismatch {
            artifact: "model.gguf",
            ..
        })
    ));

    content.tensor_infos.insert(
        "token_embd.weight".to_string(),
        gguf_file::TensorInfo {
            ggml_dtype: GgmlDType::Q4K,
            shape: candle_core::Shape::from(vec![7, 4]),
            offset: 0,
        },
    );
    assert!(matches!(
        validate_gguf_tensors(&content, &config, definition),
        Err(CandleError::InvalidQuantizedCheckpoint(message))
            if message.contains("token_embd.weight") && message.contains("expected")
    ));

    content
        .tensor_infos
        .get_mut("token_embd.weight")
        .ok_or("synthetic GGUF tensor disappeared")?
        .ggml_dtype = GgmlDType::Q2K;
    assert!(matches!(
        validate_gguf_tensors(&content, &config, definition),
        Err(CandleError::UnsupportedQuantization(message))
            if message.contains("token_embd.weight")
    ));
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn loaded_model_works_with_agent_builder() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use rig_agent::{agent::AgentBuilder, completion::Prompt};

    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        let model = LlamaModel::builder(model_data()?)
            .temperature(0.0)
            .max_tokens(1)
            .build()?;
        let agent = AgentBuilder::new(model).preamble("Be brief.").build();
        let _answer = agent.prompt("hello").await?;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    })?;
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn buffered_and_streaming_generation_are_equivalent()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = LlamaModel::builder(model_data()?)
        .temperature(0.0)
        .max_tokens(3)
        .build()?;
    let completion_request = request(vec![Message::user("hello")]);
    let buffered = model.raw_completion(completion_request.clone()).await?;
    let (streamed_text, streamed) = collect_stream(&model, completion_request).await?;

    assert_eq!(streamed_text, buffered.text);
    assert_eq!(streamed.text, buffered.text);
    assert_eq!(streamed.prompt_tokens, buffered.prompt_tokens);
    assert_eq!(streamed.generated_tokens, buffered.generated_tokens);
    assert_eq!(streamed.finish_reason, buffered.finish_reason);
    assert_eq!(streamed.requested_max_tokens, buffered.requested_max_tokens);
    assert_eq!(streamed.effective_max_tokens, buffered.effective_max_tokens);
    assert_eq!(
        rig_core::completion::Usage::from(&streamed),
        rig_core::completion::Usage::from(&buffered)
    );
    assert!(streamed.time_to_first_token_ms.is_some());
    assert!(streamed.prefill_duration_ms <= streamed.generation_duration_ms);
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn streaming_reports_eos_and_excludes_the_stop_token()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
    loaded.generation.temperature = 0.0;
    loaded.profile.stop_tokens.insert(0);
    let model = LlamaModel {
        state: Arc::new(loaded),
    };
    let (text, raw) = collect_stream(&model, request(vec![Message::user("hello")])).await?;
    assert!(text.is_empty());
    assert!(raw.text.is_empty());
    assert_eq!(raw.finish_reason, FinishReason::Eos);
    assert_eq!(raw.generated_tokens, 1);
    assert_eq!(rig_core::completion::Usage::from(&raw).output_tokens, 1);
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn streaming_clamps_context_and_rejects_bad_request_options()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
    let mut completion_request = request(vec![Message::user("hello")]);
    completion_request.max_tokens = Some(10);
    completion_request.temperature = Some(0.0);
    let prompt = render_prompt(&completion_request)?;
    let prompt_tokens = loaded.tokenizer.encode(prompt, false)?.len();
    loaded.profile.context_limit = prompt_tokens + 2;
    let model = LlamaModel {
        state: Arc::new(loaded),
    };
    let (_, raw) = collect_stream(&model, completion_request).await?;
    assert_eq!(raw.requested_max_tokens, 10);
    assert_eq!(raw.effective_max_tokens, 2);
    assert_eq!(raw.generated_tokens, 2);

    for additional_params in [
        serde_json::json!({"unknown": true}),
        serde_json::json!({"top_k": "four"}),
    ] {
        let mut bad_request = request(vec![Message::user("hello")]);
        bad_request.additional_params = Some(additional_params);
        let mut stream = model.raw_stream(bad_request).await?;
        let item = stream
            .next()
            .await
            .ok_or("bad streaming request produced no error item")?;
        assert!(item.is_err());
    }
    Ok(())
}

#[test]
fn incremental_decoder_preserves_token_boundaries() -> Result<(), CandleError> {
    let tokenizer = Tokenizer::from_bytes(
        tiny_tokenizer().map_err(|error| CandleError::TokenizerLoading(error.to_string()))?,
    )
    .map_err(|error| CandleError::TokenizerLoading(error.to_string()))?;
    let ids = [0, 0, 7];
    let independently_decoded = ids
        .iter()
        .map(|id| tokenizer.decode(&[*id], true))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?
        .join("");
    let complete = tokenizer
        .decode(&ids, true)
        .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
    assert_ne!(independently_decoded, complete);

    let mut decoder = IncrementalTextDecoder::new(&tokenizer);
    let mut streamed = String::new();
    for id in ids {
        if let Some(fragment) = decoder.push(id)? {
            streamed.push_str(&fragment);
        }
    }
    if let Some(fragment) = decoder.finish()? {
        streamed.push_str(&fragment);
    }
    assert_eq!(streamed, complete);
    assert_eq!(streamed, decoder.text());
    Ok(())
}

#[test]
fn incremental_decoder_waits_for_complete_unicode_bytes()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let vocab: Vocab = [
        ("<0x20>".to_string(), 0),
        ("<0xC3>".to_string(), 1),
        ("<0xA9>".to_string(), 2),
    ]
    .into_iter()
    .collect();
    let tokenizer: Tokenizer = TokenizerBuilder::default()
        .with_model(
            BPE::builder()
                .vocab_and_merges(vocab, Vec::new())
                .byte_fallback(true)
                .build()?,
        )
        .with_decoder(Some(ByteFallback::default()))
        .with_normalizer(Some(NFC))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .build()?
        .into();
    let mut decoder = IncrementalTextDecoder::new(&tokenizer);
    assert!(decoder.push(1)?.is_none());
    assert_eq!(decoder.push(2)?.as_deref(), Some("é"));
    assert!(decoder.finish()?.is_none());
    assert_eq!(decoder.text(), "é");
    Ok(())
}

#[test]
fn inference_clamps_context_and_uses_fresh_generation_state()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
    let mut completion_request = request(vec![Message::user("hello")]);
    completion_request.max_tokens = Some(10);
    completion_request.temperature = Some(0.0);
    let prompt = render_prompt(&completion_request)?;
    let prompt_tokens = loaded.tokenizer.encode(prompt, false)?.len();
    loaded.profile.context_limit = prompt_tokens + 2;

    let first = infer(
        &loaded,
        completion_request.clone(),
        &CancellationSignal::default(),
    )?;
    let second = infer(&loaded, completion_request, &CancellationSignal::default())?;
    let (first, second) = (first.response, second.response);
    assert_eq!(first.text, second.text);
    assert_eq!(first.generated_tokens, 2);
    assert_eq!(first.requested_max_tokens, 10);
    assert_eq!(first.effective_max_tokens, 2);
    assert_eq!(first.finish_reason, FinishReason::MaxTokens);
    assert_eq!(rig_core::completion::Usage::from(&first).output_tokens, 2);
    assert!(!first.text.contains("hello"));
    Ok(())
}

#[test]
fn eos_is_counted_but_excluded_from_decoded_text()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
    loaded.profile.stop_tokens.insert(0);
    let mut completion_request = request(vec![Message::user("hello")]);
    completion_request.temperature = Some(0.0);
    let response = infer(&loaded, completion_request, &CancellationSignal::default())?.response;
    assert_eq!(response.finish_reason, FinishReason::Eos);
    assert_eq!(response.generated_tokens, 1);
    assert_eq!(
        rig_core::completion::Usage::from(&response).output_tokens,
        1
    );
    assert!(response.text.is_empty());
    Ok(())
}

#[test]
fn sampling_modes_and_repeat_window_are_exact() {
    let mut config = GenerationConfig {
        temperature: 0.0,
        ..GenerationConfig::default()
    };
    assert!(matches!(sampling(&config), Sampling::ArgMax));
    config.temperature = 0.5;
    config.top_k = Some(3);
    config.top_p = None;
    assert!(matches!(sampling(&config), Sampling::TopK { k: 3, .. }));
    config.top_k = None;
    config.top_p = Some(0.8);
    assert!(matches!(sampling(&config), Sampling::TopP { p: 0.8, .. }));
    config.top_k = Some(2);
    assert!(matches!(
        sampling(&config),
        Sampling::TopKThenTopP { k: 2, p: 0.8, .. }
    ));
    assert_eq!(recent_tokens(&[1, 2, 3, 4], 2), &[3, 4]);
    assert_eq!(recent_tokens(&[1, 2], 8), &[1, 2]);
    assert_eq!(recent_tokens(&[1, 2], 0), &[] as &[u32]);
    assert!(matches!(next_cache_position(12, 0), Ok(12)));
    assert!(matches!(next_cache_position(12, 1), Ok(13)));
    assert!(next_cache_position(usize::MAX, 1).is_err());
}

#[test]
fn qwen3_4b_configuration_is_exactly_scoped() -> Result<(), CandleError> {
    let mut config: Qwen3Config = serde_json::from_str(
        r#"{
            "architectures":["Qwen3ForCausalLM"],
            "model_type":"qwen3",
            "hidden_size":2560,
            "intermediate_size":9728,
            "num_hidden_layers":36,
            "num_attention_heads":32,
            "num_key_value_heads":8,
            "head_dim":128,
            "max_position_embeddings":40960,
            "vocab_size":151936,
            "rms_norm_eps":0.000001,
            "rope_theta":1000000,
            "tie_word_embeddings":true,
            "bos_token_id":151643,
            "eos_token_id":151645,
            "hidden_act":"silu",
            "attention_bias":false
        }"#,
    )
    .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let definition = definition_for(ModelFamily::Qwen3, ArtifactFormat::Gguf)?;
    validate_qwen3_config(&config, definition)?;

    config.model_type = "qwen2".to_string();
    assert!(matches!(
        validate_qwen3_config(&config, definition),
        Err(CandleError::UnsupportedModelFamily(_))
    ));
    config.model_type = "qwen3".to_string();
    config.hidden_size = 4096;
    assert!(matches!(
        validate_qwen3_config(&config, definition),
        Err(CandleError::ArtifactMismatch {
            artifact: "config.json",
            ..
        })
    ));
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn concurrency_limit_and_cancellation_are_deterministic()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    assert!(matches!(
        LlamaModel::builder(model_data()?)
            .max_concurrent_requests(0)
            .build(),
        Err(CandleError::InvalidConcurrencyLimit)
    ));

    let loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
    let permit = Arc::clone(&loaded.concurrency).try_acquire_owned()?;
    assert!(Arc::clone(&loaded.concurrency).try_acquire_owned().is_err());
    drop(permit);
    assert!(Arc::clone(&loaded.concurrency).try_acquire_owned().is_ok());

    let signal = CancellationSignal::default();
    {
        let _guard = CancelOnDrop::new(signal.clone());
    }
    assert!(signal.is_cancelled());

    let signal = CancellationSignal::default();
    signal.cancel();
    assert!(matches!(
        infer(&loaded, request(vec![Message::user("hello")]), &signal),
        Err(CandleError::Cancelled)
    ));
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn concurrent_completions_have_independent_caches_and_samplers()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = LlamaModel::builder(model_data()?)
        .temperature(0.0)
        .max_tokens(2)
        .max_concurrent_requests(2)
        .build()?;
    let first = model.raw_completion(request(vec![Message::user("hello")]));
    let second = model.raw_completion(request(vec![Message::user("hello")]));
    let (first, second) = tokio::join!(first, second);
    let first = first?;
    let second = second?;
    assert_eq!(first.text, second.text);
    assert_eq!(first.generated_tokens, 2);
    assert_eq!(second.generated_tokens, 2);

    let first_stream = collect_stream(&model, request(vec![Message::user("hello")]));
    let second_stream = collect_stream(&model, request(vec![Message::user("hello")]));
    let (first_stream, second_stream) = tokio::join!(first_stream, second_stream);
    let (first_text, first_raw) = first_stream?;
    let (second_text, second_raw) = second_stream?;
    assert_eq!(first_text, second_text);
    assert_eq!(first_raw.text, second_raw.text);
    assert_eq!(first_raw.generated_tokens, 2);
    assert_eq!(second_raw.generated_tokens, 2);

    let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
    semaphore.close();
    assert!(matches!(
        acquire_concurrency(semaphore).await,
        Err(CandleError::ConcurrencyControllerClosed)
    ));
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn closed_admission_controller_fails_public_operations()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = LlamaModel::builder(model_data()?).build()?;
    let loaded = &model.state;
    loaded.concurrency.close();
    let completion_error = model
        .completion(request(vec![Message::user("hello")]))
        .await
        .err()
        .ok_or("closed completion admission unexpectedly succeeded")?;
    assert!(
        completion_error
            .to_string()
            .contains("concurrency controller is closed")
    );
    let stream_error = model
        .stream(request(vec![Message::user("hello")]))
        .await
        .err()
        .ok_or("closed stream admission unexpectedly succeeded")?;
    assert!(
        stream_error
            .to_string()
            .contains("concurrency controller is closed")
    );
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn dropping_buffered_completion_retains_permit_until_worker_exits()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (model, control, concurrency) = controlled_model(true, false, 2)?;
    let first_model = model.clone();
    let first = tokio::spawn(async move {
        first_model
            .completion(request(vec![Message::user("hello")]))
            .await
    });
    control.wait_until_entered().await;

    let second = model.raw_completion(request(vec![Message::user("hello")]));
    futures::pin_mut!(second);
    assert!(futures::poll!(&mut second).is_pending());

    first.abort();
    assert!(first.await.is_err());
    assert!(Arc::clone(&concurrency).try_acquire_owned().is_err());
    control.release()?;

    let second = second.await?;
    assert_eq!(second.generated_tokens, 2);
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn dropping_stream_cancels_worker_before_queued_request_runs()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (model, control, concurrency) = controlled_model(true, false, 2)?;
    let stream = model
        .raw_stream(request(vec![Message::user("hello")]))
        .await?;
    control.wait_until_entered().await;

    let queued = model.raw_completion(request(vec![Message::user("hello")]));
    futures::pin_mut!(queued);
    assert!(futures::poll!(&mut queued).is_pending());

    drop(stream);
    assert!(Arc::clone(&concurrency).try_acquire_owned().is_err());
    control.release()?;

    let queued = queued.await?;
    assert_eq!(queued.generated_tokens, 2);
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn public_stream_cancel_stops_worker_without_dropping_response()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (model, control, concurrency) = controlled_model(true, false, 2)?;
    let mut stream = model.stream(request(vec![Message::user("hello")])).await?;
    control.wait_until_entered().await;

    stream.cancel();
    assert!(stream.next().await.is_none());
    let queued = model.raw_completion(request(vec![Message::user("hello")]));
    futures::pin_mut!(queued);
    assert!(futures::poll!(&mut queued).is_pending());
    assert!(Arc::clone(&concurrency).try_acquire_owned().is_err());

    control.release()?;
    let queued = queued.await?;
    assert_eq!(queued.generated_tokens, 2);
    assert!(stream.next().await.is_none());
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn streaming_channel_applies_bounded_backpressure()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (model, control, concurrency) =
        controlled_model(false, false, (STREAM_CHANNEL_CAPACITY + 4) as u64)?;
    let stream = model
        .raw_stream(request(vec![Message::user("hello")]))
        .await?;
    control
        .wait_for_delivery_attempts(STREAM_CHANNEL_CAPACITY + 1)
        .await;
    assert_eq!(
        control.delivery_attempt_count(),
        STREAM_CHANNEL_CAPACITY + 1
    );

    drop(stream);
    let permit = Arc::clone(&concurrency).acquire_owned().await?;
    drop(permit);
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn blocking_task_panic_maps_to_typed_completion_error()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (model, _, _) = controlled_model(false, true, 1)?;
    let error = model
        .completion(request(vec![Message::user("hello")]))
        .await
        .err()
        .ok_or("blocking task panic unexpectedly succeeded")?;
    assert!(error.to_string().contains("Candle blocking task failed"));

    let (model, _, _) = controlled_model(false, true, 1)?;
    let mut stream = model
        .raw_stream(request(vec![Message::user("hello")]))
        .await?;
    let error = stream
        .next()
        .await
        .ok_or("panicked streaming task produced no error")?
        .err()
        .ok_or("panicked streaming task unexpectedly produced content")?;
    assert!(error.to_string().contains("Candle blocking task failed"));
    Ok(())
}

#[test]
fn builder_rejects_invalid_generation_defaults() {
    assert!(matches!(
        LlamaModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .max_tokens(0)
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
    assert!(matches!(
        LlamaModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .temperature(f64::INFINITY)
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
    assert!(matches!(
        LlamaModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .top_p(Some(0.0))
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
    assert!(matches!(
        LlamaModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .repeat_penalty(0.0)
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
}

#[test]
fn renders_llama3_history_and_documents() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let history_request = request(vec![
        Message::system("rules"),
        Message::user("question"),
        Message::assistant("answer"),
        Message::user("follow-up"),
    ]);
    assert_eq!(
        render_prompt(&history_request)?,
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nrules<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nquestion<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nanswer<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nfollow-up<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    let mut request = request(vec![Message::system("rules"), Message::user("question")]);
    request.documents.push(Document {
        id: "doc-1".to_string(),
        text: "context".to_string(),
        additional_props: HashMap::new(),
    });
    let rendered = render_prompt(&request)?;
    assert!(rendered.contains("<file id: doc-1>\ncontext\n</file>"));
    assert!(rendered.find("<file id: doc-1>") < rendered.find("question"));
    Ok(())
}

#[test]
fn renders_smollm2_history_default_system_and_generation_suffix()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let with_system = request(vec![
        Message::system("rules"),
        Message::user("question"),
        Message::assistant("answer"),
        Message::user("follow-up"),
    ]);
    assert_eq!(
        render_prompt_for(&with_system, ModelFamily::SmolLm2)?,
        "<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n<|im_start|>user\nfollow-up<|im_end|>\n<|im_start|>assistant\n"
    );

    let without_system = request(vec![Message::user("hello")]);
    assert_eq!(
        render_prompt_for(&without_system, ModelFamily::SmolLm2)?,
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
    Ok(())
}

#[test]
fn rejects_unsupported_request_features() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut tools = request(vec![Message::user("hello")]);
    tools.tools.push(ToolDefinition {
        name: "tool".to_string(),
        description: "tool".to_string(),
        parameters: serde_json::json!({}),
    });
    assert!(
        matches!(render_prompt(&tools), Err(CandleError::UnsupportedFeature(feature)) if feature.contains("Qwen3"))
    );

    let mut choice = request(vec![Message::user("hello")]);
    choice.tool_choice = Some(ToolChoice::Auto);
    assert!(render_prompt(&choice).is_err());

    let mut schema = request(vec![Message::user("hello")]);
    schema.output_schema = Some(serde_json::from_value(
        serde_json::json!({"type": "string"}),
    )?);
    assert!(render_prompt(&schema).is_err());

    let mut override_request = request(vec![Message::user("hello")]);
    override_request.model = Some("other".to_string());
    assert!(render_prompt(&override_request).is_err());

    let tool_result = request(vec![Message::tool_result("id", "tool", "result")]);
    assert!(render_prompt(&tool_result).is_err());

    let image = Message::User {
        content: vec![UserContent::image_base64(
            "data",
            Some(ImageMediaType::PNG),
            Some(ImageDetail::Auto),
        )],
    };
    assert!(render_prompt(&request(vec![image])).is_err());

    let audio = Message::User {
        content: vec![UserContent::audio("data", Some(AudioMediaType::WAV))],
    };
    assert!(render_prompt(&request(vec![audio])).is_err());
    Ok(())
}

#[test]
fn request_generation_overrides_defaults_and_validates() -> Result<(), CandleError> {
    let defaults = GenerationConfig::default();
    let mut request = request(vec![Message::user("hello")]);
    request.max_tokens = Some(12);
    request.temperature = Some(0.0);
    request.additional_params = Some(serde_json::json!({
        "top_k": 4,
        "top_p": 0.7,
        "seed": 7,
        "repeat_penalty": 1.2,
        "repeat_last_n": 9
    }));
    let effective = effective_generation(&request, &defaults, 8)?;
    assert_eq!(effective.max_tokens, 12);
    assert_eq!(effective.temperature, 0.0);
    assert_eq!(effective.top_k, Some(4));
    assert_eq!(effective.top_p, Some(0.7));
    assert_eq!(effective.seed, 7);

    request.additional_params = Some(serde_json::json!({
        "top_k": null,
        "top_p": null
    }));
    let effective = effective_generation(&request, &defaults, 8)?;
    assert_eq!(effective.top_k, None);
    assert_eq!(effective.top_p, None);

    let mut inherited_defaults = defaults.clone();
    inherited_defaults.top_k = Some(5);
    request.additional_params = Some(serde_json::json!({}));
    let effective = effective_generation(&request, &inherited_defaults, 8)?;
    assert_eq!(effective.top_k, Some(5));
    assert_eq!(effective.top_p, defaults.top_p);

    request.additional_params = Some(serde_json::json!({"unknown": true}));
    assert!(effective_generation(&request, &defaults, 8).is_err());
    request.additional_params = Some(serde_json::json!({"top_k": "four"}));
    assert!(effective_generation(&request, &defaults, 8).is_err());
    request.additional_params = None;
    request.max_tokens = Some(0);
    assert!(effective_generation(&request, &defaults, 8).is_err());
    request.max_tokens = Some(1);
    request.temperature = Some(f64::NAN);
    assert!(effective_generation(&request, &defaults, 8).is_err());
    Ok(())
}

#[test]
fn converts_finish_reason_and_usage() -> Result<(), CandleError> {
    let response = CandleCompletionResponse {
        text: "done".to_string(),
        prompt_tokens: 5,
        generated_tokens: 2,
        requested_max_tokens: 4,
        effective_max_tokens: 3,
        finish_reason: FinishReason::Eos,
        prefill_duration_ms: 8,
        time_to_first_token_ms: Some(10),
        generation_duration_ms: 20,
        tokens_per_second: Some(100.0),
    };
    let usage = rig_core::completion::Usage::from(&response);
    assert_eq!(usage.input_tokens, 5);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.total_tokens, 7);
    assert_eq!(response.finish_reason, FinishReason::Eos);
    assert_eq!(response.text, "done");
    assert_eq!(response.requested_max_tokens, 4);
    assert_eq!(response.effective_max_tokens, 3);
    assert_eq!(response.prefill_duration_ms, 8);
    assert_eq!(response.time_to_first_token_ms, Some(10));
    assert_eq!(response.generation_duration_ms, 20);
    assert_eq!(response.tokens_per_second, Some(100.0));
    Ok(())
}
