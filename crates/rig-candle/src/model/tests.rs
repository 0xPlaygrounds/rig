use candle_core::quantized::{GgmlDType, gguf_file};
use candle_transformers::generation::Sampling;
use candle_transformers::models::llama::LlamaConfig;
#[cfg(not(target_family = "wasm"))]
use futures::StreamExt;
use rig_core::completion::ToolDefinition;
use rig_core::message::{AudioMediaType, ImageDetail, ImageMediaType, ToolChoice};
#[cfg(not(target_family = "wasm"))]
use rig_core::streaming::{Item, StreamEvent};
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

/// The generation wire over `model`.
fn generation(model: &CandleModel) -> rig_core::Model<Generation, CandleModel> {
    model.completion()
}

/// A generation wire for a scripted Qwen3 runtime.
fn scripted() -> Generation {
    Generation {
        model: "qwen3-scripted".to_owned(),
        protocol: ConversationProtocol::Qwen3,
        facts: Default::default(),
    }
}

/// Replays scripted generation events as a local generator would send them.
#[derive(Clone)]
struct Scripted(Arc<std::sync::Mutex<Vec<GenerationEvent>>>);

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

/// The stream the generation wire yields for scripted `events`.
fn stream_from_events(
    events: Vec<GenerationEvent>,
) -> Result<rig_core::streaming::CompletionStream, ProviderError> {
    rig_core::Model::new(
        scripted(),
        Scripted(Arc::new(std::sync::Mutex::new(events))),
    )
    .stream(request(vec![Message::user("hello")]))
}

/// One unary completion's local response record, read off its `raw`.
async fn raw_completion(
    model: &CandleModel,
    request: CompletionRequest,
) -> Result<CandleCompletionResponse, ProviderError> {
    let response = generation(model).call(request).await?;
    Ok(serde_json::from_value(response.raw)?)
}

#[cfg(not(target_family = "wasm"))]
type ControlledModel = (CandleModel, Arc<TestControl>, Arc<tokio::sync::Semaphore>);

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
    CompletionRequest::from(if messages.is_empty() {
        vec![Message::user("hello")]
    } else {
        messages
    })
}

/// `request` as the generation wire encodes it for the loaded model.
fn payload(request: CompletionRequest) -> Result<CandleRequest, CandleError> {
    use rig_core::wire::Wire as _;
    scripted()
        .encode(request, Mode::Unary)
        .map_err(|error| CandleError::InvalidGeneration(error.to_string()))
}

/// The generation settings `request` asks of a model with `defaults`.
fn settings(
    request: &CompletionRequest,
    defaults: &GenerationConfig,
    vocab_size: usize,
) -> Result<GenerationConfig, CandleError> {
    effective_generation(
        request,
        &payload(request.clone())?.params,
        defaults,
        vocab_size,
    )
}

#[cfg(not(target_family = "wasm"))]
async fn collect_stream(
    model: &CandleModel,
    request: CompletionRequest,
) -> Result<(String, CandleCompletionResponse), Box<dyn std::error::Error + Send + Sync>> {
    let mut response = generation(model).stream(request)?;
    let mut text = String::new();
    while let Some(item) = response.next().await {
        if let Item::Event(StreamEvent::Text { text: fragment, .. }) = item? {
            text.push_str(&fragment);
        }
    }
    let terminal = response.finish().await?;
    // The local record rides the terminal's `raw`, typed back here.
    let raw: CandleCompletionResponse = serde_json::from_value(terminal.raw)?;
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
        CandleModel {
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
        let error = CandleModel::from_safetensors(data)
            .err()
            .ok_or("expected empty-buffer error")?;
        assert!(
            matches!(error, CandleError::EmptyBuffer { artifact: actual } if actual == artifact)
        );
    }

    let mut data = model_data()?;
    data.config = b"not json".to_vec();
    assert!(matches!(
        CandleModel::from_safetensors(data),
        Err(CandleError::Configuration(_))
    ));
    let mut data = model_data()?;
    data.tokenizer = b"not json".to_vec();
    assert!(matches!(
        CandleModel::from_safetensors(data),
        Err(CandleError::TokenizerLoading(_))
    ));
    let mut data = model_data()?;
    data.weights = b"not safetensors".to_vec();
    assert!(matches!(
        CandleModel::from_safetensors(data),
        Err(CandleError::InvalidCheckpoint(_))
    ));
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
    let model = CandleModel::from_safetensors(ModelData {
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
        CandleModel::from_safetensors(data),
        Err(CandleError::TokenizerVocabularyMismatch {
            expected: 9,
            actual: 8
        })
    ));

    let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
    let config = config.into_config(false);
    let llama = definition_for(ConversationProtocol::Llama3, ArtifactFormat::Safetensors)?;
    let smollm2 = definition_for(ConversationProtocol::SmolLm2, ArtifactFormat::Gguf)?;
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
        CandleModel::from_safetensors(data),
        Err(CandleError::TokenIdOutOfRange { token, id: 8, .. }) if token == "bos_token_id"
    ));
    let data = ModelData {
        config: config_with("eos_token_id", serde_json::json!([1, 9]))?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    };
    assert!(matches!(
        CandleModel::from_safetensors(data),
        Err(CandleError::TokenIdOutOfRange { token, id: 9, .. }) if token == "eos_token_id"
    ));
    for (field, value) in [("bos_token_id", 7.into()), ("eos_token_id", 3.into())] {
        let data = ModelData {
            config: config_with(field, value)?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        };
        assert!(matches!(
            CandleModel::from_safetensors(data),
            Err(CandleError::ArtifactMismatch { artifact, .. }) if artifact == field
        ));
    }
    let data = ModelData {
        config: config_with("eos_token_id", serde_json::json!([]))?,
        tokenizer: tiny_tokenizer()?,
        weights: checkpoint(true)?,
    };
    assert!(matches!(
        CandleModel::from_safetensors(data),
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
    let model = CandleModel::from_safetensors(model_data()?)?;
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
    assert_eq!(
        model.conversation_protocol(),
        Some(ConversationProtocol::Llama3)
    );
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
        .conversation_protocol(ConversationProtocol::SmolLm2)
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
    assert_eq!(builder.family, Some(ConversationProtocol::SmolLm2));
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
    assert_eq!(
        direct.conversation_protocol(),
        Some(ConversationProtocol::Llama3)
    );

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

#[test]
fn typed_gguf_and_family_errors_preserve_the_failure_kind()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data = model_data()?;
    assert!(matches!(
        CandleModel::builder(data)
            .conversation_protocol(ConversationProtocol::SmolLm2)
            .build(),
        Err(CandleError::ModelFamilyMismatch {
            selected: ConversationProtocol::SmolLm2,
            detected: ConversationProtocol::Llama3,
        })
    ));

    assert!(matches!(
        CandleModel::from_gguf(model_data()?),
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
    let definition = definition_for(ConversationProtocol::SmolLm2, ArtifactFormat::Gguf)?;
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

/// A turn another model produced replays from its canonical fields, so its
/// reasoning reaches the local prompt as plain text instead of vanishing.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn foreign_reasoning_in_history_reaches_the_prompt_as_text()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = CandleModel::builder(model_data()?).max_tokens(1).build()?;
    let history = vec![
        Message::user("hello"),
        Message::Assistant(rig_core::message::AssistantMessage::new(vec![
            rig_core::message::AssistantContent::Reasoning(rig_core::message::Reasoning::new(
                "elsewhere",
            )),
            rig_core::message::AssistantContent::text("hi"),
        ])),
        Message::user("again"),
    ];
    generation(&model).call(request(history)).await?;
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
    let definition = definition_for(ConversationProtocol::Qwen3, ArtifactFormat::Gguf)?;
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
        CandleModel::builder(model_data()?)
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
        infer(
            &loaded,
            &payload(request(vec![Message::user("hello")]))?,
            &signal
        ),
        Err(CandleError::Cancelled)
    ));
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn closed_admission_controller_fails_public_operations()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = CandleModel::builder(model_data()?).build()?;
    let loaded = &model.state;
    loaded.concurrency.close();
    let completion_error = generation(&model)
        .call(request(vec![Message::user("hello")]))
        .await
        .err()
        .ok_or("closed completion admission unexpectedly succeeded")?;
    assert!(
        completion_error
            .to_string()
            .contains("concurrency controller is closed")
    );
    // Admission is refused as the stream's first and only item.
    let stream_error = generation(&model)
        .stream(request(vec![Message::user("hello")]))?
        .next()
        .await
        .and_then(Result::err)
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
        generation(&first_model)
            .call(request(vec![Message::user("hello")]))
            .await
    });
    control.wait_until_entered().await;

    let second = raw_completion(&model, request(vec![Message::user("hello")]));
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
async fn streaming_channel_applies_bounded_backpressure()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (model, control, concurrency) =
        controlled_model(false, false, (STREAM_CHANNEL_CAPACITY + 4) as u64)?;
    let mut stream = generation(&model).stream(request(vec![Message::user("hello")]))?;
    // The worker starts when the stream is first polled.
    let _ = futures::poll!(stream.next());
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
    let error = generation(&model)
        .call(request(vec![Message::user("hello")]))
        .await
        .err()
        .ok_or("blocking task panic unexpectedly succeeded")?;
    assert!(error.to_string().contains("Candle blocking task failed"));

    let (model, _, _) = controlled_model(false, true, 1)?;
    let mut stream = generation(&model).stream(request(vec![Message::user("hello")]))?;
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
        CandleModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .max_tokens(0)
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
    assert!(matches!(
        CandleModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .temperature(f64::INFINITY)
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
    assert!(matches!(
        CandleModel::builder(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        })
        .top_p(Some(0.0))
        .build(),
        Err(CandleError::InvalidGeneration(_))
    ));
    assert!(matches!(
        CandleModel::builder(ModelData {
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
fn renders_smollm2_history_default_system_and_generation_suffix()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let with_system = request(vec![
        Message::system("rules"),
        Message::user("question"),
        Message::assistant("answer"),
        Message::user("follow-up"),
    ]);
    assert_eq!(
        render_prompt_for(&with_system, ConversationProtocol::SmolLm2)?,
        "<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n<|im_start|>user\nfollow-up<|im_end|>\n<|im_start|>assistant\n"
    );

    let without_system = request(vec![Message::user("hello")]);
    assert_eq!(
        render_prompt_for(&without_system, ConversationProtocol::SmolLm2)?,
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    );
    Ok(())
}

#[test]
fn rejects_unsupported_request_features() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut tools = request(vec![Message::user("hello")]);
    tools.tools.push(ToolDefinition {
        name: rig_core::message::ToolName::new("tool")?,
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

    let loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
    let mut override_request = request(vec![Message::user("hello")]);
    override_request.model = Some("other".to_string());
    assert!(matches!(
        infer(
            &loaded,
            &payload(override_request)?,
            &CancellationSignal::default()
        ),
        Err(CandleError::UnsupportedFeature(feature)) if feature.contains("model override")
    ));

    let tool_result = request(vec![Message::tool_result(
        rig_core::message::CallId::from_wire("id"),
        rig_core::message::ToolName::new("tool")?,
        "result",
    )]);
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
        content: vec![UserContent::audio_base64("data", Some(AudioMediaType::WAV))],
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
    let effective = settings(&request, &defaults, 8)?;
    assert_eq!(effective.max_tokens, 12);
    assert_eq!(effective.temperature, 0.0);
    assert_eq!(effective.top_k, Some(4));
    assert_eq!(effective.top_p, Some(0.7));
    assert_eq!(effective.seed, 7);

    request.additional_params = Some(serde_json::json!({
        "top_k": null,
        "top_p": null
    }));
    let effective = settings(&request, &defaults, 8)?;
    assert_eq!(effective.top_k, None);
    assert_eq!(effective.top_p, None);

    let mut inherited_defaults = defaults.clone();
    inherited_defaults.top_k = Some(5);
    request.additional_params = Some(serde_json::json!({}));
    let effective = settings(&request, &inherited_defaults, 8)?;
    assert_eq!(effective.top_k, Some(5));
    assert_eq!(effective.top_p, defaults.top_p);

    request.additional_params = Some(serde_json::json!({"unknown": true}));
    assert!(settings(&request, &defaults, 8).is_err());
    request.additional_params = Some(serde_json::json!({"top_k": "four"}));
    assert!(settings(&request, &defaults, 8).is_err());
    request.additional_params = None;
    request.max_tokens = Some(0);
    assert!(settings(&request, &defaults, 8).is_err());
    request.max_tokens = Some(1);
    request.temperature = Some(f64::NAN);
    assert!(settings(&request, &defaults, 8).is_err());
    Ok(())
}

/// The load-bearing property behind `CompletionResponse::raw` and
/// `CompletionResponse::raw` for this crate: the captured value is
/// `serde_json::to_value(&CandleCompletionResponse)` — the local record
/// `raw_completion` returns — and a consumer must be able to read it back as
/// the same type and get the same JSON, with the local generation metrics rig
/// never normalizes (timings, tokens/second, the local finish reason) intact.
/// There is no cassette harness for a local model, so this is the unit-form
/// pin, independent of any model load.
#[test]
fn candle_completion_response_round_trips_through_serde_json_value()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let raw = CandleCompletionResponse {
        text: "done".to_string(),
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

    let value = serde_json::to_value(&raw)?;
    assert_eq!(
        value.pointer("/tokens_per_second"),
        Some(&serde_json::json!(100.5))
    );
    assert_eq!(
        value.pointer("/finish_reason"),
        Some(&serde_json::json!("max_tokens"))
    );

    let back: CandleCompletionResponse = serde_json::from_value(value.clone())?;
    assert_eq!(
        serde_json::to_value(&back)?,
        value,
        "the capture must read back into CandleCompletionResponse and re-serialize identically"
    );
    assert_eq!(back, raw);
    Ok(())
}

/// Raw capture through the real `Model::call` path on the
/// tiny in-crate model (greedy, so two runs generate the same tokens): `raw`
/// deserializes back into `CandleCompletionResponse`, re-serializes
/// identically, and reports the same text and token counts `raw_completion`
/// returns for the same request — and the normalized usage agrees with it.
/// Timings are wall-clock and so are compared only through the round-trip,
/// never across runs.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn completion_raw_round_trips_into_the_local_record()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = CandleModel::builder(model_data()?)
        .temperature(0.0)
        .max_tokens(2)
        .build()?;

    let response = generation(&model)
        .call(request(vec![Message::user("hello")]))
        .await?;
    let escape_hatch = raw_completion(&model, request(vec![Message::user("hello")])).await?;

    assert!(
        !response.raw.is_null(),
        "a provider-backed completion always carries raw"
    );
    let raw = &response.raw;
    let typed: CandleCompletionResponse = serde_json::from_value(raw.clone())?;
    assert_eq!(
        serde_json::to_value(&typed)?,
        *raw,
        "the capture must be exactly what the local record serializes to"
    );
    assert_eq!(typed.text, escape_hatch.text);
    assert_eq!(typed.prompt_tokens, escape_hatch.prompt_tokens);
    assert_eq!(typed.generated_tokens, escape_hatch.generated_tokens);
    assert_eq!(typed.finish_reason, escape_hatch.finish_reason);
    assert_eq!(typed.generated_tokens, 2);

    assert_eq!(response.usage.input_tokens, Some(typed.prompt_tokens));
    assert_eq!(response.usage.output_tokens, Some(typed.generated_tokens));
    assert_eq!(response.usage.output_tokens, Some(2));
    Ok(())
}

/// The streaming twin through the real `Model::stream` path: the
/// terminal `CompletionResponse::raw` is the local record the generator's `Final`
/// event carries — it round-trips into `CandleCompletionResponse`, agrees
/// with a second stream's terminal on text and token counts, and
/// re-normalizing it through the events-first seam reproduces every
/// normalized field.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn stream_terminal_raw_round_trips_into_the_local_record()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = CandleModel::builder(model_data()?)
        .temperature(0.0)
        .max_tokens(2)
        .build()?;

    let mut stream = generation(&model).stream(request(vec![Message::user("hello")]))?;
    while let Some(item) = stream.next().await {
        item?;
    }
    let terminal = stream.finish().await?;
    let (_, streamed) = collect_stream(&model, request(vec![Message::user("hello")])).await?;

    assert!(
        !terminal.raw.is_null(),
        "a provider-backed terminal always carries raw"
    );
    let raw = &terminal.raw;
    let typed: CandleCompletionResponse = serde_json::from_value(raw.clone())?;
    assert_eq!(
        serde_json::to_value(&typed)?,
        *raw,
        "the capture must be exactly what the terminal record serializes to"
    );
    assert_eq!(typed.text, streamed.text);
    assert_eq!(typed.prompt_tokens, streamed.prompt_tokens);
    assert_eq!(typed.generated_tokens, streamed.generated_tokens);
    assert_eq!(typed.finish_reason, streamed.finish_reason);

    let mut renormalized = stream_from_events(vec![GenerationEvent::Final(typed)])?;
    while let Some(item) = renormalized.next().await {
        item?;
    }
    let renormalized = renormalized.finish().await?;
    assert_eq!(terminal.identity(), renormalized.identity());
    assert_eq!(terminal.finish_reason(), renormalized.finish_reason());
    assert_eq!(terminal.model(), renormalized.model());
    assert_eq!(terminal.usage, renormalized.usage);
    assert_eq!(terminal.usage.output_tokens, Some(2));
    Ok(())
}

/// The wire a loaded model builds names that checkpoint, by its protocol
/// and a digest of its config and tokenizer, so two loaded models are two
/// models to replay.
#[test]
fn the_wire_names_the_loaded_checkpoint() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = CandleModel::builder(model_data()?).build()?;
    let wire = model.completion().wire;
    assert_eq!(wire.model, model.model_id());
    assert_eq!(wire.protocol, ConversationProtocol::Llama3);
    let id = crate::loader::model_id(ConversationProtocol::Qwen3, b"{}", b"{}");
    assert_eq!(
        id,
        crate::loader::model_id(ConversationProtocol::Qwen3, b"{}", b"{}")
    );
    assert!(
        id.starts_with("qwen3-") && id.len() == "qwen3-".len() + 16,
        "{id}"
    );
    assert_ne!(
        id,
        crate::loader::model_id(ConversationProtocol::Qwen3, b"{}", b"{ }")
    );
    Ok(())
}

/// The generation wire for a loaded model of `protocol`.
fn local(protocol: ConversationProtocol) -> Generation {
    Generation {
        model: crate::loader::model_id(protocol, b"config", b"tokenizer"),
        protocol,
        facts: Default::default(),
    }
}

/// `history` and a next prompt, prepared for `wire` as the driver prepares
/// it, with `tools` declared.
fn prepared_for(
    wire: &Generation,
    history: Vec<Message>,
    tools: Vec<ToolDefinition>,
) -> Result<CompletionRequest, ProviderError> {
    let mut request = CompletionRequest::new("next").tools(tools);
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    <rig_core::operation::Completion as rig_core::wire::Operation>::prepare(
        request,
        &rig_core::wire::Wire::describe(wire),
    )
}

/// NEW-candle-tools (round 4): another model's tool exchange reaches a
/// Llama 3 or SmolLM2 model, whose prompts carry no tools, as text: the
/// adapter keeps no call or result even when the request declares the
/// tool, and the prompt renders.
#[test]
fn another_models_tool_history_renders_for_a_plain_protocol()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use rig_core::message::{
        AssistantContent, AssistantMessage, Origin, StopReason, ToolCall, ToolFunction,
        ToolResultContent,
    };
    let call = ToolCall::new(
        rig_core::message::CallId::from_wire("toolu_1"),
        ToolFunction::new(
            rig_core::message::ToolName::new("add")?,
            serde_json::json!({"x": 1}),
        ),
    );
    let history = vec![
        Message::user("add"),
        Message::Assistant(
            AssistantMessage::new(vec![
                AssistantContent::text("adding"),
                AssistantContent::ToolCall(call.clone()),
            ])
            .with_origin(Origin::new("anthropic.messages", "anthropic", "claude"))
            .with_stop(StopReason::ToolUse),
        ),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("2")]),
            )],
        },
        Message::assistant("it is 2"),
    ];
    let add = ToolDefinition {
        name: rig_core::message::ToolName::new("add")?,
        description: "add".to_owned(),
        parameters: serde_json::json!({"type": "object"}),
    };
    for protocol in [ConversationProtocol::Llama3, ConversationProtocol::SmolLm2] {
        let wire = local(protocol);
        let declared = prepared_for(&wire, history.clone(), vec![add.clone()])?;
        let structured = declared.chat_history.iter().any(|message| match message {
            Message::Assistant(turn) => turn
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_))),
            Message::User { content } => content
                .iter()
                .any(|part| matches!(part, UserContent::ToolResult(_))),
            Message::System { .. } => false,
        });
        assert!(!structured, "{protocol:?}: {:?}", declared.chat_history);
        let request = prepared_for(&wire, history.clone(), Vec::new())?;
        let prompt = wire.prompt(&request)?;
        assert!(prompt.contains("2"), "{protocol:?}: {prompt}");
    }
    Ok(())
}

/// `top_p` and `seed` become generation overrides above a raw `top_k`;
/// thinking off is the protocols' default; the rest is refused.
#[test]
fn options_become_generation_overrides() -> Result<(), CandleError> {
    use rig_core::completion::ReplayTarget as _;
    use rig_core::completion::{
        CacheRetention, GenerationOptions, Reasoning, ServiceTier, options::Mapping,
    };

    let mut options_request = request(vec![Message::user("hello")])
        .top_p(0.7)
        .seed(11)
        .reasoning(Reasoning::Off);
    options_request.additional_params = Some(serde_json::json!({"top_k": 4, "seed": 3}));
    let generation = settings(&options_request, &GenerationConfig::default(), 8)?;
    assert_eq!(generation.top_p, Some(0.7));
    assert_eq!(generation.top_k, Some(4));
    // `additional_params` is above the mapped options.
    assert_eq!(generation.seed, 3);

    let answers = |options: GenerationOptions| {
        let request = request(vec![Message::user("hello")]).options(options);
        scripted().map_options(&request, request.options.fields())
    };
    assert!(matches!(
        answers(GenerationOptions::default().reasoning(Reasoning::Off)).reasoning,
        Mapping::Omit(_)
    ));
    assert!(matches!(
        answers(GenerationOptions::default().cache(CacheRetention::Short)).cache,
        Mapping::Unsupported(_)
    ));
    assert!(matches!(
        answers(GenerationOptions::default().service_tier(ServiceTier::Auto)).service_tier,
        Mapping::Unsupported(_)
    ));
    assert!(matches!(
        answers(GenerationOptions::default().stop(["END"])).stop,
        Mapping::Unsupported(_)
    ));
    assert!(matches!(
        answers(GenerationOptions::default().top_p(1.5)).top_p,
        Mapping::Unsupported(_)
    ));
    Ok(())
}

/// Guarantee 5 for Candle: a stream's `raw` is the document a unary call of
/// the same greedy generation records, but for its wall-clock timings.
#[cfg(not(target_family = "wasm"))]
#[tokio::test(flavor = "current_thread")]
async fn a_streams_raw_is_the_unary_document()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = CandleModel::builder(model_data()?)
        .temperature(0.0)
        .max_tokens(2)
        .build()?;
    let unary = generation(&model)
        .call(request(vec![Message::user("hello")]))
        .await?
        .raw;
    let mut stream = generation(&model).stream(request(vec![Message::user("hello")]))?;
    while let Some(item) = stream.next().await {
        item?;
    }
    let streamed = stream.finish().await?.raw;
    let timings = [
        "/prefill_duration_ms",
        "/time_to_first_token_ms",
        "/generation_duration_ms",
        "/tokens_per_second",
    ];
    assert!(!unary.is_null());
    assert_eq!(
        rig_core::test_utils::raw_parity::comparable(&streamed, &timings),
        rig_core::test_utils::raw_parity::comparable(&unary, &timings),
    );
    Ok(())
}

/// The reassembler keeps the local response record a whole turn or a
/// stream's final event carries, and nothing else.
#[test]
fn the_document_is_the_final_response_record() -> Result<(), Box<dyn std::error::Error>> {
    use rig_core::wire::document::Reassemble;

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
        tokens_per_second: None,
    };
    let record = serde_json::to_value(&response)?;

    assert!(CandleDocument::default().finish().is_null());

    let mut streamed = CandleDocument::default();
    streamed.absorb(&CandleFrame::Event(GenerationEvent::Text(
        "done".to_owned(),
    )));
    streamed.absorb(&CandleFrame::Event(GenerationEvent::Final(
        response.clone(),
    )));
    assert_eq!(streamed.finish(), record);

    let mut whole = CandleDocument::default();
    whole.absorb(&CandleFrame::Whole(crate::generation::InferredCompletion {
        response,
        choice: Vec::new(),
    }));
    assert_eq!(whole.finish(), record);
    Ok(())
}

/// The facts a local generation is given are the facts it answers with: its
/// `ReplayTarget::facts`, which the shared option rules and the cost read,
/// and its descriptor's spec, which `DynModel::spec` returns.
#[test]
fn the_wire_answers_from_the_facts_it_is_given() -> Result<(), Box<dyn std::error::Error>> {
    use rig_core::catalog::{ModelFacts, ModelSpec};
    use rig_core::completion::ReplayTarget as _;
    use rig_core::wire::Wire as _;

    let vendor =
        rig_core::providers::registry::ProviderId::catalog("candle").ok_or("a catalog vendor")?;
    let spec = ModelSpec::new(vendor, "qwen3-scripted").with_max_output_tokens(1_234);
    let wire = scripted().with_facts(ModelFacts::new(spec));
    let bound = wire.facts().and_then(ModelFacts::spec);
    assert_eq!(bound.and_then(|spec| spec.max_output_tokens), Some(1_234));
    assert_eq!(
        wire.describe()
            .spec()
            .and_then(|spec| spec.max_output_tokens),
        Some(1_234)
    );
    Ok(())
}
