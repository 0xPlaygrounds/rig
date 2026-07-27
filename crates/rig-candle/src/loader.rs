//! Artifact parsing and Candle weight loading.

use std::collections::HashSet;
#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;

use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama, LlamaConfig};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use tokenizers::Tokenizer;

use crate::CandleError;
#[cfg(test)]
use crate::artifacts::ModelData;
use crate::artifacts::{GgufModelData, ModelArtifacts, require_nonempty};
use crate::generation::GenerationConfig;
#[cfg(target_family = "wasm")]
use crate::profile::ModelArchitecture;
use crate::profile::{
    ArtifactFormat, LoaderBackend, ModelFamily, ValidatedProfile, definition_for,
    validate_identity, validate_tokenizer_requirements,
};
use crate::runtime::RuntimeDevice;
#[cfg(all(test, not(target_family = "wasm")))]
use crate::runtime::TestControl;
use crate::validation::{
    ModelIdentity, Qwen3Config, detect_model_family, metadata_usize, resolve_stop_tokens,
    validate_checkpoint, validate_family_config, validate_gguf_metadata, validate_gguf_tensors,
    validate_model_config, validate_qwen3_config, validate_qwen3_gguf_metadata,
    validate_qwen3_gguf_tensors, validate_tokenizer,
};

pub(crate) struct LoadedModel {
    pub(crate) model: LoadedWeights,
    pub(crate) runtime: RuntimeDevice,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) profile: ValidatedProfile,
    pub(crate) generation: GenerationConfig,
    #[cfg(not(target_family = "wasm"))]
    pub(crate) concurrency: Arc<tokio::sync::Semaphore>,
    #[cfg(all(test, not(target_family = "wasm")))]
    pub(crate) test_control: Option<Arc<TestControl>>,
}

pub(crate) enum LoadedWeights {
    Safetensors { model: Llama, config: Config },
    QuantizedLlama(QuantizedLlama),
    QuantizedQwen3(QuantizedQwen3),
}

struct PreparedModel {
    profile: ValidatedProfile,
    tokenizer: Tokenizer,
    llama_config: Option<Config>,
    qwen3_config: Option<Qwen3Config>,
}

pub(crate) fn load_model_with_family(
    artifacts: ModelArtifacts,
    selected_family: Option<ModelFamily>,
    generation: GenerationConfig,
    _max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    let data = match artifacts {
        ModelArtifacts::Safetensors(data) => data,
        ModelArtifacts::Gguf(data) => {
            return load_gguf_model(
                GgufModelData {
                    config: &data.config,
                    tokenizer: &data.tokenizer,
                    weights: &data.weights,
                },
                selected_family,
                generation,
                _max_concurrent_requests,
            );
        }
    };
    require_nonempty(&data.config, "config")?;
    require_nonempty(&data.tokenizer, "tokenizer")?;
    require_nonempty(&data.weights, "weights")?;

    let mut prepared = prepare_model(
        &data.config,
        &data.tokenizer,
        selected_family,
        ArtifactFormat::Safetensors,
    )?;
    let config = prepared.llama_config.take().ok_or_else(|| {
        CandleError::Configuration("prepared Llama model omitted its configuration".to_string())
    })?;

    let runtime = RuntimeDevice::cpu();
    validate_checkpoint(&data.weights, &config)?;
    let load = || {
        let builder = VarBuilder::from_buffered_safetensors(
            data.weights,
            runtime.cache_dtype(),
            runtime.device(),
        )
        .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
        Llama::load(builder, &config).map_err(|error| CandleError::ModelLoading(error.to_string()))
    };
    #[cfg(not(target_family = "wasm"))]
    let model = runtime.device().with_context(load)?;
    #[cfg(target_family = "wasm")]
    let model = load()?;

    Ok(LoadedModel {
        model: LoadedWeights::Safetensors { model, config },
        runtime,
        tokenizer: prepared.tokenizer,
        profile: prepared.profile,
        generation,
        #[cfg(not(target_family = "wasm"))]
        concurrency: Arc::new(tokio::sync::Semaphore::new(_max_concurrent_requests)),
        #[cfg(all(test, not(target_family = "wasm")))]
        test_control: None,
    })
}

pub(crate) fn load_gguf_model(
    data: GgufModelData<'_>,
    selected_family: Option<ModelFamily>,
    generation: GenerationConfig,
    _max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    require_nonempty(data.config, "config")?;
    require_nonempty(data.tokenizer, "tokenizer")?;
    require_nonempty(data.weights, "weights")?;
    let prepared = prepare_model(
        data.config,
        data.tokenizer,
        selected_family,
        ArtifactFormat::Gguf,
    )?;
    #[cfg(target_family = "wasm")]
    if prepared.profile.definition.architecture == ModelArchitecture::Qwen3 {
        return Err(CandleError::UnsupportedModelFamily(
            "the validated Qwen3-4B profile is native-only because its runtime memory exceeds wasm32 linear-memory capacity; use SmolLM2 for WASM"
                .to_string(),
        ));
    }
    let runtime = RuntimeDevice::cpu();
    let load = || load_gguf(data.weights, &prepared, runtime.device());
    #[cfg(not(target_family = "wasm"))]
    let model = runtime.device().with_context(load)?;
    #[cfg(target_family = "wasm")]
    let model = load()?;
    Ok(LoadedModel {
        model,
        runtime,
        tokenizer: prepared.tokenizer,
        profile: prepared.profile,
        generation,
        #[cfg(not(target_family = "wasm"))]
        concurrency: Arc::new(tokio::sync::Semaphore::new(_max_concurrent_requests)),
        #[cfg(all(test, not(target_family = "wasm")))]
        test_control: None,
    })
}

fn prepare_model(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    selected_family: Option<ModelFamily>,
    artifact_format: ArtifactFormat,
) -> Result<PreparedModel, CandleError> {
    let identity: ModelIdentity = serde_json::from_slice(config_bytes)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| CandleError::TokenizerLoading(error.to_string()))?;
    let is_qwen3 = identity.model_type.as_deref() == Some("qwen3")
        || identity
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3ForCausalLM");
    if is_qwen3 {
        let config: Qwen3Config = serde_json::from_slice(config_bytes)
            .map_err(|error| CandleError::Configuration(error.to_string()))?;
        let detected_family = ModelFamily::Qwen3;
        if let Some(selected) = selected_family
            && selected != detected_family
        {
            return Err(CandleError::ModelFamilyMismatch {
                selected,
                detected: detected_family,
            });
        }
        let definition = definition_for(detected_family, artifact_format)?;
        validate_qwen3_config(&config, definition)?;
        validate_identity(
            definition,
            Some(config.model_type.as_str()),
            &config.architectures,
        )?;
        validate_tokenizer_requirements(
            definition,
            &tokenizer,
            config.vocab_size,
            Some(config.bos_token_id),
            &[config.eos_token_id],
        )?;
        let mut stop_tokens = HashSet::new();
        stop_tokens.insert(config.eos_token_id);
        return Ok(PreparedModel {
            profile: ValidatedProfile::new(
                definition,
                config.vocab_size,
                config.max_position_embeddings,
                stop_tokens,
            )?,
            tokenizer,
            llama_config: None,
            qwen3_config: Some(config),
        });
    }

    let llama_config: LlamaConfig = serde_json::from_slice(config_bytes)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let config = llama_config.into_config(false);
    validate_model_config(&config)?;
    let detected_family = detect_model_family(&tokenizer)?;
    if let Some(selected) = selected_family
        && selected != detected_family
    {
        return Err(CandleError::ModelFamilyMismatch {
            selected,
            detected: detected_family,
        });
    }
    let definition = definition_for(detected_family, artifact_format)?;
    validate_identity(
        definition,
        identity.model_type.as_deref(),
        &identity.architectures,
    )?;
    validate_family_config(config_bytes, &config, definition)?;
    validate_tokenizer(&config, &tokenizer, definition)?;
    let stop_tokens = resolve_stop_tokens(&config, &tokenizer, definition)?;
    Ok(PreparedModel {
        profile: ValidatedProfile::new(
            definition,
            config.vocab_size,
            config.max_position_embeddings,
            stop_tokens,
        )?,
        tokenizer,
        llama_config: Some(config),
        qwen3_config: None,
    })
}

#[cfg(test)]
pub(crate) fn load_model(
    data: ModelData,
    generation: GenerationConfig,
    max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    load_model_with_family(
        ModelArtifacts::Safetensors(data),
        None,
        generation,
        max_concurrent_requests,
    )
}

fn load_gguf(
    weights: &[u8],
    prepared: &PreparedModel,
    device: &Device,
) -> Result<LoadedWeights, CandleError> {
    let mut reader = std::io::Cursor::new(weights);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(|error| CandleError::InvalidQuantizedCheckpoint(error.to_string()))?;
    let definition = prepared.profile.definition;
    let expected_architecture = definition.config_identity.model_type;
    match content.metadata.get("general.architecture") {
        Some(gguf_file::Value::String(architecture)) if architecture == expected_architecture => {}
        Some(value) => {
            return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                "general.architecture must be `{expected_architecture}`, found {value:?}"
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing general.architecture metadata".to_string(),
            ));
        }
    }
    let requirements = definition.gguf.as_ref().ok_or_else(|| {
        CandleError::UnsupportedModelFamily(format!(
            "{} does not support GGUF artifacts",
            definition.name
        ))
    })?;
    match content.metadata.get("general.file_type") {
        // Q4_K_M legitimately uses auxiliary F32/Q5/Q6/Q8 encodings for
        // selected tensors; per-tensor validation below enforces that mix.
        Some(gguf_file::Value::U32(actual)) if *actual == requirements.file_type => {}
        Some(value) => {
            return Err(CandleError::UnsupportedQuantization(format!(
                "general.file_type must identify Q4_K_M ({}), found {value:?}",
                requirements.file_type
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing general.file_type metadata".to_string(),
            ));
        }
    }
    if metadata_usize(&content, "general.quantization_version")?
        != requirements.quantization_version
    {
        return Err(CandleError::UnsupportedQuantization(format!(
            "Q4_K_M checkpoint must use GGML quantization version {}",
            requirements.quantization_version
        )));
    }
    match prepared.profile.definition.loader {
        LoaderBackend::LlamaGguf => {
            let config = prepared.llama_config.as_ref().ok_or_else(|| {
                CandleError::Configuration(
                    "prepared Llama GGUF omitted its Llama configuration".to_string(),
                )
            })?;
            validate_gguf_metadata(&content, config, &prepared.tokenizer, definition)?;
            validate_gguf_tensors(&content, config, definition)?;
            QuantizedLlama::from_gguf(content, &mut reader, device)
                .map(LoadedWeights::QuantizedLlama)
                .map_err(|error| CandleError::ModelLoading(error.to_string()))
        }
        LoaderBackend::Qwen3Gguf => {
            let config = prepared.qwen3_config.as_ref().ok_or_else(|| {
                CandleError::Configuration(
                    "prepared Qwen3 GGUF omitted its Qwen3 configuration".to_string(),
                )
            })?;
            validate_qwen3_gguf_metadata(&content, config, &prepared.tokenizer, definition)?;
            validate_qwen3_gguf_tensors(&content, config, definition)?;
            QuantizedQwen3::from_gguf(content, &mut reader, device)
                .map(LoadedWeights::QuantizedQwen3)
                .map_err(|error| CandleError::ModelLoading(error.to_string()))
        }
        LoaderBackend::LlamaSafetensors => Err(CandleError::UnsupportedModelFamily(
            "a safetensors profile cannot be loaded from GGUF artifacts".to_string(),
        )),
    }
}
