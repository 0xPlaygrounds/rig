//! Configuration, tokenizer, safetensors, and GGUF validation.

use std::collections::HashSet;

use candle_core::quantized::{GgmlDType, gguf_file};
use candle_transformers::models::llama::{Config, Llama3RopeType, LlamaEosToks};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::CandleError;
use crate::profile::{
    BEGIN_OF_TEXT, ConfigValues, END_HEADER, END_OF_TURN, IM_END, IM_START, ModelFamily,
    ProfileDefinition, START_HEADER, validate_config_requirements, validate_dimensions,
    validate_tokenizer_requirements,
};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Qwen3Config {
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    pub(crate) model_type: String,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) bos_token_id: u32,
    pub(crate) eos_token_id: u32,
    pub(crate) hidden_act: String,
    pub(crate) attention_bias: bool,
}

pub(crate) fn validate_model_config(config: &Config) -> Result<(), CandleError> {
    for (field, value) in [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("max_position_embeddings", config.max_position_embeddings),
    ] {
        if value == 0 {
            return Err(CandleError::InvalidConfigurationValue {
                field,
                reason: "value 0 must be greater than zero".to_string(),
            });
        }
    }
    if !config
        .hidden_size
        .is_multiple_of(config.num_attention_heads)
    {
        return Err(CandleError::InvalidConfigurationValue {
            field: "hidden_size",
            reason: format!(
                "value {} must be divisible by num_attention_heads {}",
                config.hidden_size, config.num_attention_heads
            ),
        });
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
    {
        return Err(CandleError::InvalidConfigurationValue {
            field: "num_attention_heads",
            reason: format!(
                "value {} must be divisible by num_key_value_heads {}",
                config.num_attention_heads, config.num_key_value_heads
            ),
        });
    }
    let head_dim = config.hidden_size / config.num_attention_heads;
    if !head_dim.is_multiple_of(2) {
        return Err(CandleError::InvalidConfigurationValue {
            field: "hidden_size",
            reason: format!(
                "attention head dimension {head_dim} must be even for rotary embeddings"
            ),
        });
    }
    if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field: "rms_norm_eps",
            reason: format!(
                "value {} must be finite and greater than zero",
                config.rms_norm_eps
            ),
        });
    }
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field: "rope_theta",
            reason: format!(
                "value {} must be finite and greater than zero",
                config.rope_theta
            ),
        });
    }
    if config.max_position_embeddings > u32::MAX as usize {
        return Err(CandleError::InvalidConfigurationValue {
            field: "max_position_embeddings",
            reason: format!(
                "value {} exceeds Candle's maximum representable value {}",
                config.max_position_embeddings,
                u32::MAX
            ),
        });
    }
    if let Some(rope_scaling) = &config.rope_scaling
        && matches!(rope_scaling.rope_type, Llama3RopeType::Llama3)
    {
        validate_positive_finite("rope_scaling.factor", rope_scaling.factor)?;
        validate_positive_finite("rope_scaling.low_freq_factor", rope_scaling.low_freq_factor)?;
        validate_positive_finite(
            "rope_scaling.high_freq_factor",
            rope_scaling.high_freq_factor,
        )?;
        if rope_scaling.high_freq_factor <= rope_scaling.low_freq_factor {
            return Err(CandleError::InvalidConfigurationValue {
                field: "rope_scaling.high_freq_factor",
                reason: format!(
                    "value {} must be greater than low_freq_factor {}",
                    rope_scaling.high_freq_factor, rope_scaling.low_freq_factor
                ),
            });
        }
        if rope_scaling.original_max_position_embeddings == 0 {
            return Err(CandleError::InvalidConfigurationValue {
                field: "rope_scaling.original_max_position_embeddings",
                reason: "value 0 must be greater than zero".to_string(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelIdentity {
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    pub(crate) model_type: Option<String>,
    pub(crate) hidden_act: Option<String>,
    pub(crate) attention_bias: Option<bool>,
    pub(crate) mlp_bias: Option<bool>,
    pub(crate) rope_interleaved: Option<bool>,
}

pub(crate) fn validate_family_config(
    config_bytes: &[u8],
    config: &Config,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    let identity: ModelIdentity = serde_json::from_slice(config_bytes)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let actual = [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("max_position_embeddings", config.max_position_embeddings),
    ];
    validate_dimensions(definition, &actual)?;
    validate_config_requirements(
        definition,
        &ConfigValues {
            hidden_act: identity.hidden_act.as_deref(),
            attention_bias: identity.attention_bias,
            mlp_bias: identity.mlp_bias,
            rope_interleaved: identity.rope_interleaved,
            tie_word_embeddings: config.tie_word_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f64,
            bos_token_id: config.bos_token_id,
            eos_token_id: None,
        },
    )
}

pub(crate) fn validate_qwen3_config(
    config: &Qwen3Config,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    if config.model_type != "qwen3"
        || !config
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3ForCausalLM")
    {
        return Err(CandleError::UnsupportedModelFamily(
            "configuration must declare Qwen3ForCausalLM with model_type `qwen3`".to_string(),
        ));
    }
    let actual = [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("head_dim", config.head_dim),
        ("max_position_embeddings", config.max_position_embeddings),
        ("vocab_size", config.vocab_size),
    ];
    validate_dimensions(definition, &actual)?;
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(CandleError::InvalidConfigurationValue {
            field: "attention dimensions",
            reason: "Qwen3 attention heads must divide KV heads and head_dim must be even"
                .to_string(),
        });
    }
    validate_config_requirements(
        definition,
        &ConfigValues {
            hidden_act: Some(config.hidden_act.as_str()),
            attention_bias: Some(config.attention_bias),
            mlp_bias: None,
            rope_interleaved: None,
            tie_word_embeddings: config.tie_word_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            bos_token_id: Some(config.bos_token_id),
            eos_token_id: Some(config.eos_token_id),
        },
    )?;
    validate_token_id("bos_token_id", config.bos_token_id, config.vocab_size)?;
    validate_token_id("eos_token_id", config.eos_token_id, config.vocab_size)?;
    Ok(())
}

pub(crate) fn validate_positive_finite(field: &'static str, value: f32) -> Result<(), CandleError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field,
            reason: format!("value {value} must be finite and greater than zero"),
        });
    }
    Ok(())
}

pub(crate) fn detect_model_family(tokenizer: &Tokenizer) -> Result<ModelFamily, CandleError> {
    let llama3 = [BEGIN_OF_TEXT, START_HEADER, END_HEADER, END_OF_TURN]
        .iter()
        .all(|token| tokenizer.token_to_id(token).is_some());
    let smollm2 = [IM_START, IM_END]
        .iter()
        .all(|token| tokenizer.token_to_id(token).is_some());
    match (llama3, smollm2) {
        (true, false) => Ok(ModelFamily::Llama3),
        (false, true) => Ok(ModelFamily::SmolLm2),
        (true, true) => Err(CandleError::UnsupportedModelFamily(
            "tokenizer ambiguously contains both Llama 3 and SmolLM2 control tokens".to_string(),
        )),
        (false, false) => Err(CandleError::UnsupportedModelFamily(
            "tokenizer contains neither the Llama 3 nor SmolLM2 control-token set".to_string(),
        )),
    }
}

pub(crate) fn validate_tokenizer(
    config: &Config,
    tokenizer: &Tokenizer,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    if let Some(token) = config.bos_token_id {
        validate_token_id("bos_token_id", token, config.vocab_size)?;
    }
    let configured_eos = match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => {
            validate_token_id("eos_token_id", *token, config.vocab_size)?;
            vec![*token]
        }
        Some(LlamaEosToks::Multiple(tokens)) => {
            if tokens.is_empty() {
                return Err(CandleError::InvalidConfigurationValue {
                    field: "eos_token_id",
                    reason: "must contain at least one token ID when present".to_string(),
                });
            }
            for token in tokens {
                validate_token_id("eos_token_id", *token, config.vocab_size)?;
            }
            tokens.clone()
        }
        None => Vec::new(),
    };
    validate_tokenizer_requirements(
        definition,
        tokenizer,
        config.vocab_size,
        config.bos_token_id,
        &configured_eos,
    )
}

pub(crate) fn validate_token_id(
    token: &str,
    id: u32,
    vocab_size: usize,
) -> Result<(), CandleError> {
    if (id as usize) >= vocab_size {
        return Err(CandleError::TokenIdOutOfRange {
            token: token.to_string(),
            id,
            vocab_size,
        });
    }
    Ok(())
}

pub(crate) fn resolve_stop_tokens(
    config: &Config,
    tokenizer: &Tokenizer,
    definition: &ProfileDefinition,
) -> Result<HashSet<u32>, CandleError> {
    let mut tokens = HashSet::new();
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => {
            tokens.insert(*token);
        }
        Some(LlamaEosToks::Multiple(items)) => {
            tokens.extend(items.iter().copied());
        }
        None => {}
    }
    if let Some(token) = tokenizer.token_to_id(definition.end_token) {
        tokens.insert(token);
    }
    if tokens.is_empty() {
        Err(CandleError::MissingStopToken)
    } else {
        Ok(tokens)
    }
}

pub(crate) fn validate_gguf_metadata(
    content: &gguf_file::Content,
    config: &Config,
    tokenizer: &Tokenizer,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    let dims = [
        ("llama.vocab_size", config.vocab_size),
        ("llama.embedding_length", config.hidden_size),
        ("llama.feed_forward_length", config.intermediate_size),
        ("llama.block_count", config.num_hidden_layers),
        ("llama.attention.head_count", config.num_attention_heads),
        ("llama.attention.head_count_kv", config.num_key_value_heads),
        ("llama.context_length", config.max_position_embeddings),
        (
            "llama.rope.dimension_count",
            config.hidden_size / config.num_attention_heads,
        ),
    ];
    let floats = [
        ("llama.rope.freq_base", config.rope_theta as f64),
        (
            "llama.attention.layer_norm_rms_epsilon",
            config.rms_norm_eps,
        ),
    ];
    validate_gguf_common(
        content,
        &dims,
        &floats,
        definition,
        tokenizer,
        config.vocab_size,
        false,
    )
}

fn validate_gguf_common(
    content: &gguf_file::Content,
    dims: &[(&str, usize)],
    floats: &[(&str, f64)],
    definition: &ProfileDefinition,
    tokenizer: &Tokenizer,
    vocab_size: usize,
    allow_padding: bool,
) -> Result<(), CandleError> {
    for (key, expected) in dims {
        let actual = metadata_usize(content, key)?;
        if actual != *expected {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but config requires {expected}"),
            });
        }
    }
    for (key, expected) in floats {
        let actual = metadata_f64(content, key)?;
        if (actual - expected).abs() > 1e-5 * expected.abs().max(f64::MIN_POSITIVE) {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but config requires {expected}"),
            });
        }
    }
    let requirements = gguf_requirements(definition)?;
    for requirement in requirements.metadata_strings {
        require_metadata_string(content, requirement.key, requirement.value)?;
    }
    validate_gguf_tokenizer_vocabulary(content, vocab_size, tokenizer, allow_padding)?;
    for (key, token) in [
        ("tokenizer.ggml.bos_token_id", definition.start_token),
        ("tokenizer.ggml.eos_token_id", definition.end_token),
    ] {
        let actual = metadata_usize(content, key)?;
        let expected = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })? as usize;
        if actual != expected {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but tokenizer requires {expected}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_qwen3_gguf_metadata(
    content: &gguf_file::Content,
    config: &Qwen3Config,
    tokenizer: &Tokenizer,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    let dims = [
        ("qwen3.embedding_length", config.hidden_size),
        ("qwen3.feed_forward_length", config.intermediate_size),
        ("qwen3.block_count", config.num_hidden_layers),
        ("qwen3.attention.head_count", config.num_attention_heads),
        ("qwen3.attention.head_count_kv", config.num_key_value_heads),
        ("qwen3.attention.key_length", config.head_dim),
        ("qwen3.attention.value_length", config.head_dim),
        ("qwen3.context_length", config.max_position_embeddings),
    ];
    let floats = [
        ("qwen3.rope.freq_base", config.rope_theta),
        (
            "qwen3.attention.layer_norm_rms_epsilon",
            config.rms_norm_eps,
        ),
    ];
    validate_gguf_common(
        content,
        &dims,
        &floats,
        definition,
        tokenizer,
        config.vocab_size,
        true,
    )?;
    let requirements = gguf_requirements(definition)?;
    match content.metadata.get("tokenizer.chat_template") {
        Some(gguf_file::Value::String(template))
            if requirements
                .chat_template_markers
                .iter()
                .all(|required| template.contains(required)) => {}
        Some(_) => {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: "Qwen3 tokenizer.chat_template does not contain the official Hermes tool and no-thinking protocol markers".to_string(),
            });
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing `tokenizer.chat_template` metadata".to_string(),
            ));
        }
    }
    Ok(())
}

fn gguf_requirements(
    definition: &ProfileDefinition,
) -> Result<&crate::profile::GgufRequirements, CandleError> {
    definition.gguf.as_ref().ok_or_else(|| {
        CandleError::UnsupportedModelFamily(format!(
            "{} does not support GGUF artifacts",
            definition.name
        ))
    })
}

pub(crate) fn validate_gguf_tokenizer_vocabulary(
    content: &gguf_file::Content,
    vocab_size: usize,
    tokenizer: &Tokenizer,
    allow_padding: bool,
) -> Result<(), CandleError> {
    let tokens = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(gguf_file::Value::Array(tokens)) => tokens,
        Some(value) => {
            return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                "metadata `tokenizer.ggml.tokens` must be an array, found {value:?}"
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing `tokenizer.ggml.tokens` metadata".to_string(),
            ));
        }
    };
    if tokens.len() != vocab_size {
        return Err(CandleError::ArtifactMismatch {
            artifact: "model.gguf",
            reason: format!(
                "GGUF tokenizer has {} tokens, but config requires {vocab_size}",
                tokens.len()
            ),
        });
    }
    for (id, value) in tokens.iter().enumerate() {
        let gguf_token = match value {
            gguf_file::Value::String(token) => token,
            value => {
                return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                    "tokenizer.ggml.tokens[{id}] must be a string, found {value:?}"
                )));
            }
        };
        let id = u32::try_from(id).map_err(|_| {
            CandleError::InvalidQuantizedCheckpoint(
                "GGUF tokenizer index does not fit in u32".to_string(),
            )
        })?;
        let matches = tokenizer.id_to_token(id).map_or_else(
            || allow_padding && *gguf_token == format!("[PAD{id}]"),
            |token| token == gguf_token.as_str(),
        );
        if !matches {
            let reason = if allow_padding {
                format!(
                    "token ID {id} does not match the GGUF tokenizer vocabulary or its required padding token"
                )
            } else {
                format!("token ID {id} does not match the GGUF tokenizer vocabulary")
            };
            return Err(CandleError::ArtifactMismatch {
                artifact: "tokenizer.json",
                reason,
            });
        }
    }
    Ok(())
}

pub(crate) fn metadata_usize(
    content: &gguf_file::Content,
    key: &str,
) -> Result<usize, CandleError> {
    let value = content.metadata.get(key).ok_or_else(|| {
        CandleError::InvalidQuantizedCheckpoint(format!("missing `{key}` metadata"))
    })?;
    match value {
        gguf_file::Value::U32(value) => Ok(*value as usize),
        gguf_file::Value::U64(value) => usize::try_from(*value).map_err(|_| {
            CandleError::InvalidQuantizedCheckpoint(format!(
                "metadata `{key}` does not fit in usize"
            ))
        }),
        value => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "metadata `{key}` must be an unsigned integer, found {value:?}"
        ))),
    }
}

pub(crate) fn metadata_f64(content: &gguf_file::Content, key: &str) -> Result<f64, CandleError> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::F32(value)) => Ok(f64::from(*value)),
        Some(gguf_file::Value::F64(value)) => Ok(*value),
        Some(value) => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "metadata `{key}` must be floating point, found {value:?}"
        ))),
        None => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "missing `{key}` metadata"
        ))),
    }
}

pub(crate) fn require_metadata_string(
    content: &gguf_file::Content,
    key: &str,
    expected: &str,
) -> Result<(), CandleError> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::String(actual)) if actual.eq_ignore_ascii_case(expected) => Ok(()),
        Some(value) => Err(CandleError::ArtifactMismatch {
            artifact: "model.gguf",
            reason: format!("metadata `{key}` must be `{expected}`, found {value:?}"),
        }),
        None => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "missing `{key}` metadata"
        ))),
    }
}

/// Per-layer weight shapes shared by the Llama-architecture GGUF and
/// safetensors layouts: (GGUF suffix, safetensors suffix, shape).
fn llama_layer_shapes(config: &Config) -> [(&'static str, &'static str, Vec<usize>); 9] {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_size = (hidden / config.num_attention_heads) * config.num_key_value_heads;
    [
        (
            "attn_q.weight",
            "self_attn.q_proj.weight",
            vec![hidden, hidden],
        ),
        (
            "attn_k.weight",
            "self_attn.k_proj.weight",
            vec![kv_size, hidden],
        ),
        (
            "attn_v.weight",
            "self_attn.v_proj.weight",
            vec![kv_size, hidden],
        ),
        (
            "attn_output.weight",
            "self_attn.o_proj.weight",
            vec![hidden, hidden],
        ),
        (
            "ffn_gate.weight",
            "mlp.gate_proj.weight",
            vec![intermediate, hidden],
        ),
        (
            "ffn_down.weight",
            "mlp.down_proj.weight",
            vec![hidden, intermediate],
        ),
        (
            "ffn_up.weight",
            "mlp.up_proj.weight",
            vec![intermediate, hidden],
        ),
        ("attn_norm.weight", "input_layernorm.weight", vec![hidden]),
        (
            "ffn_norm.weight",
            "post_attention_layernorm.weight",
            vec![hidden],
        ),
    ]
}

fn reject_disallowed_dtypes(
    content: &gguf_file::Content,
    requirements: &crate::profile::GgufRequirements,
    prefix: &'static str,
    suffix: &'static str,
) -> Result<(), CandleError> {
    for (name, tensor) in &content.tensor_infos {
        if !requirements
            .allowed_tensor_dtypes
            .contains(&tensor.ggml_dtype)
        {
            return Err(CandleError::UnsupportedQuantization(format!(
                "{prefix}tensor `{name}` uses unsupported {:?}{suffix}",
                tensor.ggml_dtype
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_gguf_tensors(
    content: &gguf_file::Content,
    config: &Config,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    let requirements = gguf_requirements(definition)?;
    reject_disallowed_dtypes(content, requirements, "", " in a Q4_K_M checkpoint")?;
    if !content
        .tensor_infos
        .values()
        .any(|tensor| tensor.ggml_dtype == GgmlDType::Q4K)
    {
        return Err(CandleError::UnsupportedQuantization(
            "checkpoint contains no Q4_K tensors; rig-candle supports the Q4_K_M tensor mix"
                .to_string(),
        ));
    }
    validate_gguf_tensor(
        content,
        "token_embd.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    validate_gguf_tensor(content, "output_norm.weight", &[config.hidden_size])?;
    if !config.tie_word_embeddings || content.tensor_infos.contains_key("output.weight") {
        validate_gguf_tensor(
            content,
            "output.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    let layer_shapes = llama_layer_shapes(config);
    for layer in 0..config.num_hidden_layers {
        for (suffix, _, shape) in &layer_shapes {
            validate_gguf_tensor(content, &format!("blk.{layer}.{suffix}"), shape)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_qwen3_gguf_tensors(
    content: &gguf_file::Content,
    config: &Qwen3Config,
    definition: &ProfileDefinition,
) -> Result<(), CandleError> {
    let requirements = gguf_requirements(definition)?;
    reject_disallowed_dtypes(
        content,
        requirements,
        "Qwen3 ",
        "; pinned Q4_K_M permits F32, Q4_K, and Q6_K",
    )?;
    let tensors_per_layer = requirements.tensors_per_layer.ok_or_else(|| {
        CandleError::Configuration(format!(
            "{} does not define an exact GGUF tensor layout",
            definition.name
        ))
    })?;
    let expected_count = 2usize
        .checked_add(
            config
                .num_hidden_layers
                .checked_mul(tensors_per_layer)
                .ok_or_else(|| {
                    CandleError::InvalidQuantizedCheckpoint(
                        "Qwen3 expected tensor count overflowed usize".to_string(),
                    )
                })?,
        )
        .ok_or_else(|| {
            CandleError::InvalidQuantizedCheckpoint(
                "Qwen3 expected tensor count overflowed usize".to_string(),
            )
        })?;
    if content.tensor_infos.len() != expected_count {
        return Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "Qwen3-4B Q4_K_M contains {} tensors, expected exactly {expected_count}",
            content.tensor_infos.len()
        )));
    }
    validate_gguf_tensor_dtype(
        content,
        "token_embd.weight",
        &[config.vocab_size, config.hidden_size],
        requirements.token_embedding_dtypes,
    )?;
    validate_gguf_tensor_dtype(
        content,
        "output_norm.weight",
        &[config.hidden_size],
        requirements.norm_dtypes,
    )?;
    let (hidden, intermediate) = (config.hidden_size, config.intermediate_size);
    let query_size = config.num_attention_heads * config.head_dim;
    let kv_size = config.num_key_value_heads * config.head_dim;
    let (matrix, mixed, norm) = (
        requirements.matrix_dtypes,
        requirements.mixed_matrix_dtypes,
        requirements.norm_dtypes,
    );
    for layer in 0..config.num_hidden_layers {
        for (suffix, shape, dtypes) in [
            ("attn_q.weight", vec![query_size, hidden], matrix),
            ("attn_k.weight", vec![kv_size, hidden], matrix),
            ("attn_v.weight", vec![kv_size, hidden], mixed),
            ("attn_output.weight", vec![hidden, query_size], matrix),
            ("attn_q_norm.weight", vec![config.head_dim], norm),
            ("attn_k_norm.weight", vec![config.head_dim], norm),
            ("ffn_gate.weight", vec![intermediate, hidden], matrix),
            ("ffn_down.weight", vec![hidden, intermediate], mixed),
            ("ffn_up.weight", vec![intermediate, hidden], matrix),
            ("attn_norm.weight", vec![hidden], norm),
            ("ffn_norm.weight", vec![hidden], norm),
        ] {
            validate_gguf_tensor_dtype(content, &format!("blk.{layer}.{suffix}"), &shape, dtypes)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_gguf_tensor_dtype(
    content: &gguf_file::Content,
    name: &str,
    expected_shape: &[usize],
    allowed_dtypes: &[GgmlDType],
) -> Result<(), CandleError> {
    validate_gguf_tensor(content, name, expected_shape)?;
    let tensor = content.tensor_infos.get(name).ok_or_else(|| {
        CandleError::InvalidQuantizedCheckpoint(format!("missing expected tensor `{name}`"))
    })?;
    if !allowed_dtypes.contains(&tensor.ggml_dtype) {
        return Err(CandleError::UnsupportedQuantization(format!(
            "tensor `{name}` uses {:?}, expected one of {allowed_dtypes:?}",
            tensor.ggml_dtype
        )));
    }
    Ok(())
}

pub(crate) fn validate_gguf_tensor(
    content: &gguf_file::Content,
    name: &str,
    expected: &[usize],
) -> Result<(), CandleError> {
    let tensor = content.tensor_infos.get(name).ok_or_else(|| {
        CandleError::InvalidQuantizedCheckpoint(format!("missing expected tensor `{name}`"))
    })?;
    if tensor.shape.dims() != expected {
        return Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "tensor `{name}` has shape {:?}, expected {expected:?}",
            tensor.shape.dims()
        )));
    }
    Ok(())
}

pub(crate) fn validate_checkpoint(weights: &[u8], config: &Config) -> Result<(), CandleError> {
    let tensors = SafeTensors::deserialize(weights)
        .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
    validate_tensor(
        &tensors,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    validate_tensor(&tensors, "model.norm.weight", &[config.hidden_size])?;
    if !config.tie_word_embeddings {
        validate_tensor(
            &tensors,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    let layer_shapes = llama_layer_shapes(config);
    for layer in 0..config.num_hidden_layers {
        for (_, suffix, shape) in &layer_shapes {
            validate_tensor(&tensors, &format!("model.layers.{layer}.{suffix}"), shape)?;
        }
    }
    Ok(())
}

pub(crate) fn validate_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<(), CandleError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| CandleError::MissingTensor(name.to_string()))?;
    if tensor.shape() != expected_shape {
        return Err(CandleError::TensorShapeMismatch {
            tensor: name.to_string(),
            expected: expected_shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    if !matches!(
        tensor.dtype(),
        SafeDtype::F32 | SafeDtype::F16 | SafeDtype::BF16
    ) {
        return Err(CandleError::UnsupportedTensorDtype {
            tensor: name.to_string(),
            dtype: format!("{:?}", tensor.dtype()),
        });
    }
    Ok(())
}
