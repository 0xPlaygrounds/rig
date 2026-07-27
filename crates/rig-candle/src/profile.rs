//! Central definitions for the model/checkpoint combinations Rig can execute.

use std::collections::HashSet;

use candle_core::quantized::GgmlDType;
use serde::{Deserialize, Serialize};

use crate::CandleError;

pub const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
pub const START_HEADER: &str = "<|start_header_id|>";
pub const END_HEADER: &str = "<|end_header_id|>";
pub const END_OF_TURN: &str = "<|eot_id|>";
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub(crate) const END_OF_TEXT: &str = "<|endoftext|>";
pub const SMOLLM2_DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/// Explicit conversation and generated-output protocol selected from validated artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationProtocol {
    /// Meta Llama 3 instruct control-token format.
    Llama3,
    /// Hugging Face SmolLM2 instruct (ChatML-style) format.
    SmolLm2,
    /// Qwen3 ChatML/Hermes tool-calling format.
    Qwen3,
}

/// Backwards-compatible name for [`ConversationProtocol`].
pub type ModelFamily = ConversationProtocol;

/// Transformer architecture used to execute a loaded checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    /// Candle's Llama implementation, including compatible SmolLM2 checkpoints.
    Llama,
    /// Candle's Qwen3 implementation with per-head query/key normalization.
    Qwen3,
}

/// Quantized tensor encoding detected in a GGUF checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    /// Mixed GGUF tensors whose primary matrix encoding is Q4_K.
    Q4K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactFormat {
    Safetensors,
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoaderBackend {
    LlamaSafetensors,
    LlamaGguf,
    Qwen3Gguf,
}

#[derive(Debug)]
pub(crate) struct ConfigIdentity {
    pub(crate) model_type: &'static str,
    pub(crate) architecture: &'static str,
    pub(crate) required: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DimensionRequirement {
    pub(crate) field: &'static str,
    pub(crate) value: usize,
}

#[derive(Debug)]
pub(crate) struct ConfigRequirements {
    pub(crate) hidden_act: Option<&'static str>,
    pub(crate) attention_bias: Option<bool>,
    pub(crate) mlp_bias: Option<bool>,
    pub(crate) rope_interleaved: Option<bool>,
    pub(crate) tie_word_embeddings: Option<bool>,
    pub(crate) rms_norm_eps: Option<f64>,
    pub(crate) rope_theta: Option<f64>,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenizerVocabulary {
    ModelCapacity,
    Exact(usize),
}

#[derive(Debug)]
pub(crate) struct MetadataStringRequirement {
    pub(crate) key: &'static str,
    pub(crate) value: &'static str,
}

#[derive(Debug)]
pub(crate) struct GgufRequirements {
    pub(crate) file_type: u32,
    pub(crate) quantization_version: usize,
    pub(crate) metadata_strings: &'static [MetadataStringRequirement],
    pub(crate) chat_template_markers: &'static [&'static str],
    pub(crate) allowed_tensor_dtypes: &'static [GgmlDType],
    pub(crate) token_embedding_dtypes: &'static [GgmlDType],
    pub(crate) norm_dtypes: &'static [GgmlDType],
    pub(crate) matrix_dtypes: &'static [GgmlDType],
    pub(crate) mixed_matrix_dtypes: &'static [GgmlDType],
    pub(crate) tensors_per_layer: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct ProfileDefinition {
    pub(crate) name: &'static str,
    pub(crate) architecture: ModelArchitecture,
    pub(crate) protocol: ConversationProtocol,
    pub(crate) artifact_format: ArtifactFormat,
    pub(crate) loader: LoaderBackend,
    pub(crate) quantization: Option<Quantization>,
    pub(crate) config_identity: ConfigIdentity,
    pub(crate) config_dimensions: &'static [DimensionRequirement],
    pub(crate) config_requirements: ConfigRequirements,
    pub(crate) tokenizer_tokens: &'static [&'static str],
    pub(crate) tokenizer_vocabulary: TokenizerVocabulary,
    pub(crate) start_token: &'static str,
    pub(crate) end_token: &'static str,
    pub(crate) context_limit_cap: Option<usize>,
    pub(crate) gguf: Option<GgufRequirements>,
}

const LLAMA3_PROFILE: ProfileDefinition = ProfileDefinition {
    name: "Llama 3 safetensors",
    architecture: ModelArchitecture::Llama,
    protocol: ConversationProtocol::Llama3,
    artifact_format: ArtifactFormat::Safetensors,
    loader: LoaderBackend::LlamaSafetensors,
    quantization: None,
    config_identity: ConfigIdentity {
        model_type: "llama",
        architecture: "LlamaForCausalLM",
        required: false,
    },
    config_dimensions: &[],
    config_requirements: ConfigRequirements {
        hidden_act: None,
        attention_bias: None,
        mlp_bias: None,
        rope_interleaved: None,
        tie_word_embeddings: None,
        rms_norm_eps: None,
        rope_theta: None,
        bos_token_id: None,
        eos_token_id: None,
    },
    tokenizer_tokens: &[
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ],
    tokenizer_vocabulary: TokenizerVocabulary::ModelCapacity,
    start_token: BEGIN_OF_TEXT,
    end_token: END_OF_TURN,
    context_limit_cap: None,
    gguf: None,
};

const SMOLLM2_PROFILE: ProfileDefinition = ProfileDefinition {
    name: "SmolLM2-360M-Instruct Q4_K_M GGUF",
    architecture: ModelArchitecture::Llama,
    protocol: ConversationProtocol::SmolLm2,
    artifact_format: ArtifactFormat::Gguf,
    loader: LoaderBackend::LlamaGguf,
    quantization: Some(Quantization::Q4K),
    config_identity: ConfigIdentity {
        model_type: "llama",
        architecture: "LlamaForCausalLM",
        required: true,
    },
    config_dimensions: &[
        DimensionRequirement {
            field: "hidden_size",
            value: 960,
        },
        DimensionRequirement {
            field: "intermediate_size",
            value: 2560,
        },
        DimensionRequirement {
            field: "vocab_size",
            value: 49_152,
        },
        DimensionRequirement {
            field: "num_hidden_layers",
            value: 32,
        },
        DimensionRequirement {
            field: "num_attention_heads",
            value: 15,
        },
        DimensionRequirement {
            field: "num_key_value_heads",
            value: 5,
        },
        DimensionRequirement {
            field: "max_position_embeddings",
            value: 8192,
        },
    ],
    config_requirements: ConfigRequirements {
        hidden_act: Some("silu"),
        attention_bias: Some(false),
        mlp_bias: Some(false),
        rope_interleaved: Some(false),
        tie_word_embeddings: Some(true),
        rms_norm_eps: Some(1e-5),
        rope_theta: Some(100_000.0),
        bos_token_id: None,
        eos_token_id: None,
    },
    tokenizer_tokens: &["<|im_start|>", "<|im_end|>"],
    tokenizer_vocabulary: TokenizerVocabulary::ModelCapacity,
    start_token: IM_START,
    end_token: IM_END,
    context_limit_cap: Some(4096),
    gguf: Some(GgufRequirements {
        file_type: 15,
        quantization_version: 2,
        metadata_strings: &[
            MetadataStringRequirement {
                key: "general.basename",
                value: "smollm2",
            },
            MetadataStringRequirement {
                key: "tokenizer.ggml.model",
                value: "gpt2",
            },
            MetadataStringRequirement {
                key: "tokenizer.ggml.pre",
                value: "smollm",
            },
        ],
        chat_template_markers: &[],
        allowed_tensor_dtypes: &[
            GgmlDType::F32,
            GgmlDType::Q4K,
            GgmlDType::Q5_0,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
        ],
        token_embedding_dtypes: &[],
        norm_dtypes: &[],
        matrix_dtypes: &[],
        mixed_matrix_dtypes: &[],
        tensors_per_layer: None,
    }),
};

const QWEN3_PROFILE: ProfileDefinition = ProfileDefinition {
    name: "Qwen3-4B Q4_K_M GGUF",
    architecture: ModelArchitecture::Qwen3,
    protocol: ConversationProtocol::Qwen3,
    artifact_format: ArtifactFormat::Gguf,
    loader: LoaderBackend::Qwen3Gguf,
    quantization: Some(Quantization::Q4K),
    config_identity: ConfigIdentity {
        model_type: "qwen3",
        architecture: "Qwen3ForCausalLM",
        required: true,
    },
    config_dimensions: &[
        DimensionRequirement {
            field: "hidden_size",
            value: 2560,
        },
        DimensionRequirement {
            field: "intermediate_size",
            value: 9728,
        },
        DimensionRequirement {
            field: "num_hidden_layers",
            value: 36,
        },
        DimensionRequirement {
            field: "num_attention_heads",
            value: 32,
        },
        DimensionRequirement {
            field: "num_key_value_heads",
            value: 8,
        },
        DimensionRequirement {
            field: "head_dim",
            value: 128,
        },
        DimensionRequirement {
            field: "max_position_embeddings",
            value: 40_960,
        },
        DimensionRequirement {
            field: "vocab_size",
            value: 151_936,
        },
    ],
    config_requirements: ConfigRequirements {
        hidden_act: Some("silu"),
        attention_bias: Some(false),
        mlp_bias: None,
        rope_interleaved: None,
        tie_word_embeddings: Some(true),
        rms_norm_eps: Some(1e-6),
        rope_theta: Some(1_000_000.0),
        bos_token_id: Some(151_643),
        eos_token_id: Some(151_645),
    },
    tokenizer_tokens: &["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    tokenizer_vocabulary: TokenizerVocabulary::Exact(151_669),
    start_token: END_OF_TEXT,
    end_token: IM_END,
    context_limit_cap: Some(4096),
    gguf: Some(GgufRequirements {
        file_type: 15,
        quantization_version: 2,
        metadata_strings: &[
            MetadataStringRequirement {
                key: "general.basename",
                value: "qwen3",
            },
            MetadataStringRequirement {
                key: "general.size_label",
                value: "4b",
            },
            MetadataStringRequirement {
                key: "general.finetune",
                value: "instruct-awq",
            },
            MetadataStringRequirement {
                key: "tokenizer.ggml.model",
                value: "gpt2",
            },
            MetadataStringRequirement {
                key: "tokenizer.ggml.pre",
                value: "qwen2",
            },
        ],
        chat_template_markers: &[
            "# Tools",
            "<tools></tools>",
            "<tool_call>",
            "<tool_response>",
            "enable_thinking",
        ],
        allowed_tensor_dtypes: &[GgmlDType::F32, GgmlDType::Q4K, GgmlDType::Q6K],
        token_embedding_dtypes: &[GgmlDType::Q6K],
        norm_dtypes: &[GgmlDType::F32],
        matrix_dtypes: &[GgmlDType::Q4K],
        mixed_matrix_dtypes: &[GgmlDType::Q4K, GgmlDType::Q6K],
        tensors_per_layer: Some(11),
    }),
};

#[derive(Debug, Clone)]
pub(crate) struct ValidatedProfile {
    pub(crate) definition: &'static ProfileDefinition,
    pub(crate) vocab_size: usize,
    pub(crate) context_limit: usize,
    pub(crate) stop_tokens: HashSet<u32>,
}

impl ValidatedProfile {
    pub(crate) fn new(
        definition: &'static ProfileDefinition,
        vocab_size: usize,
        configured_context_limit: usize,
        stop_tokens: HashSet<u32>,
    ) -> Result<Self, CandleError> {
        if stop_tokens.is_empty() {
            return Err(CandleError::MissingStopToken);
        }
        let context_limit = definition
            .context_limit_cap
            .map_or(configured_context_limit, |cap| {
                configured_context_limit.min(cap)
            });
        Ok(Self {
            definition,
            vocab_size,
            context_limit,
            stop_tokens,
        })
    }
}

pub(crate) fn definition_for(
    protocol: ConversationProtocol,
    artifact_format: ArtifactFormat,
) -> Result<&'static ProfileDefinition, CandleError> {
    let definition = match protocol {
        ConversationProtocol::Llama3 => &LLAMA3_PROFILE,
        ConversationProtocol::SmolLm2 => &SMOLLM2_PROFILE,
        ConversationProtocol::Qwen3 => &QWEN3_PROFILE,
    };
    if definition.artifact_format != artifact_format {
        return Err(CandleError::UnsupportedModelFamily(format!(
            "{} requires {:?} artifacts",
            definition.name, definition.artifact_format
        )));
    }
    Ok(definition)
}

pub(crate) fn validate_identity(
    definition: &ProfileDefinition,
    model_type: Option<&str>,
    architectures: &[String],
) -> Result<(), CandleError> {
    let expected = &definition.config_identity;
    let model_type_mismatch = model_type
        .is_some_and(|model_type| model_type != expected.model_type)
        || (expected.required && model_type.is_none());
    let architecture_mismatch = (!architectures.is_empty()
        && !architectures
            .iter()
            .any(|architecture| architecture == expected.architecture))
        || (expected.required && architectures.is_empty());
    if model_type_mismatch || architecture_mismatch {
        return Err(CandleError::UnsupportedModelFamily(format!(
            "{} requires model_type `{}` and architecture `{}`",
            definition.name, expected.model_type, expected.architecture
        )));
    }
    Ok(())
}

pub(crate) fn validate_dimensions(
    definition: &ProfileDefinition,
    actual: &[(&'static str, usize)],
) -> Result<(), CandleError> {
    for requirement in definition.config_dimensions {
        let actual = actual
            .iter()
            .find_map(|(field, value)| (*field == requirement.field).then_some(*value))
            .ok_or_else(|| {
                CandleError::Configuration(format!(
                    "internal profile validation omitted `{}`",
                    requirement.field
                ))
            })?;
        if actual != requirement.value {
            return Err(CandleError::ArtifactMismatch {
                artifact: "config.json",
                reason: format!(
                    "{} requires {}={}, found {actual}",
                    definition.name, requirement.field, requirement.value
                ),
            });
        }
    }
    Ok(())
}

pub(crate) struct ConfigValues<'a> {
    pub(crate) hidden_act: Option<&'a str>,
    pub(crate) attention_bias: Option<bool>,
    pub(crate) mlp_bias: Option<bool>,
    pub(crate) rope_interleaved: Option<bool>,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_id: Option<u32>,
}

pub(crate) fn validate_config_requirements(
    definition: &ProfileDefinition,
    actual: &ConfigValues<'_>,
) -> Result<(), CandleError> {
    let expected = &definition.config_requirements;
    let mismatch = expected
        .hidden_act
        .is_some_and(|value| actual.hidden_act != Some(value))
        || expected
            .attention_bias
            .is_some_and(|value| actual.attention_bias != Some(value))
        || expected
            .mlp_bias
            .is_some_and(|value| actual.mlp_bias != Some(value))
        || expected
            .rope_interleaved
            .is_some_and(|value| actual.rope_interleaved != Some(value))
        || expected
            .tie_word_embeddings
            .is_some_and(|value| actual.tie_word_embeddings != value)
        || expected
            .rms_norm_eps
            .is_some_and(|value| (actual.rms_norm_eps - value).abs() > f64::EPSILON)
        || expected
            .rope_theta
            .is_some_and(|value| (actual.rope_theta - value).abs() > f64::EPSILON)
        || expected
            .bos_token_id
            .is_some_and(|value| actual.bos_token_id != Some(value))
        || expected
            .eos_token_id
            .is_some_and(|value| actual.eos_token_id != Some(value));
    if mismatch {
        return Err(CandleError::ArtifactMismatch {
            artifact: "config.json",
            reason: format!(
                "{} configuration invariants do not match its validated profile",
                definition.name
            ),
        });
    }
    Ok(())
}

pub(crate) fn validate_tokenizer_requirements(
    definition: &ProfileDefinition,
    tokenizer: &tokenizers::Tokenizer,
    vocab_size: usize,
    configured_bos: Option<u32>,
    configured_eos: &[u32],
) -> Result<(), CandleError> {
    let actual_vocabulary = tokenizer.get_vocab_size(true);
    match definition.tokenizer_vocabulary {
        TokenizerVocabulary::ModelCapacity if actual_vocabulary != vocab_size => {
            return Err(CandleError::TokenizerVocabularyMismatch {
                expected: vocab_size,
                actual: actual_vocabulary,
            });
        }
        TokenizerVocabulary::Exact(expected) if actual_vocabulary != expected => {
            return Err(CandleError::ArtifactMismatch {
                artifact: "tokenizer.json",
                reason: format!(
                    "{} requires {expected} defined tokenizer IDs within model capacity {vocab_size}, found {actual_vocabulary}",
                    definition.name
                ),
            });
        }
        _ if actual_vocabulary > vocab_size => {
            return Err(CandleError::TokenizerVocabularyMismatch {
                expected: vocab_size,
                actual: actual_vocabulary,
            });
        }
        _ => {}
    }
    for &token in definition.tokenizer_tokens {
        let id = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })?;
        if id as usize >= vocab_size {
            return Err(CandleError::TokenIdOutOfRange {
                token: token.to_string(),
                id,
                vocab_size,
            });
        }
        if !tokenizer.get_added_vocabulary().is_special_token(token) {
            return Err(CandleError::SpecialTokenNotMarked { token });
        }
    }
    let start_id =
        tokenizer
            .token_to_id(definition.start_token)
            .ok_or(CandleError::MissingSpecialToken {
                token: definition.start_token,
            })?;
    if configured_bos.is_some_and(|configured| configured != start_id) {
        return Err(CandleError::ArtifactMismatch {
            artifact: "bos_token_id",
            reason: format!(
                "configured BOS ID does not match '{}' ID {start_id}",
                definition.start_token
            ),
        });
    }
    let end_id =
        tokenizer
            .token_to_id(definition.end_token)
            .ok_or(CandleError::MissingSpecialToken {
                token: definition.end_token,
            })?;
    if !configured_eos.is_empty() && !configured_eos.contains(&end_id) {
        return Err(CandleError::ArtifactMismatch {
            artifact: "eos_token_id",
            reason: format!(
                "configured EOS IDs do not contain '{}' ID {end_id}",
                definition.end_token
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests {
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
}
