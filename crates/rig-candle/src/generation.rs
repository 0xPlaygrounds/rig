//! Generation configuration, sampling, incremental decoding, and inference sessions.

use candle_transformers::generation::Sampling;
use rig_core::completion::CompletionRequest;
use serde::Deserialize;

use crate::CandleError;

/// Sampling and length defaults used when a completion request does not override them.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    /// Maximum number of tokens generated for a request.
    pub max_tokens: u64,
    /// Sampling temperature. Zero selects greedy decoding.
    pub temperature: f64,
    /// Optional number of highest-probability tokens retained during sampling.
    pub top_k: Option<usize>,
    /// Optional nucleus-sampling probability threshold.
    pub top_p: Option<f64>,
    /// Deterministic random seed used by the sampler.
    pub seed: u64,
    /// Penalty applied to tokens repeated in the recent context. `1.0` disables it.
    pub repeat_penalty: f32,
    /// Number of recent tokens considered by the repeat penalty.
    pub repeat_last_n: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_k: None,
            top_p: Some(0.95),
            seed: 299_792_458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
enum OptionalGenerationOverride<T> {
    #[default]
    Inherit,
    Set(T),
    Disable,
}

impl<'de, T> Deserialize<'de> for OptionalGenerationOverride<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|value| match value {
            Some(value) => Self::Set(value),
            None => Self::Disable,
        })
    }
}

impl<T> OptionalGenerationOverride<T> {
    fn resolve(self, default: Option<T>) -> Option<T> {
        match self {
            Self::Inherit => default,
            Self::Set(value) => Some(value),
            Self::Disable => None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RequestGenerationOverrides {
    top_k: OptionalGenerationOverride<usize>,
    top_p: OptionalGenerationOverride<f64>,
    seed: Option<u64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
}

fn override_or<T>(value: Option<T>, default: T) -> T {
    match value {
        Some(value) => value,
        None => default,
    }
}

pub(crate) fn effective_generation(
    request: &CompletionRequest,
    defaults: &GenerationConfig,
    vocab_size: usize,
) -> Result<GenerationConfig, CandleError> {
    let overrides = match &request.additional_params {
        Some(value) => serde_json::from_value::<RequestGenerationOverrides>(value.clone())
            .map_err(|error| CandleError::InvalidGeneration(error.to_string()))?,
        None => RequestGenerationOverrides::default(),
    };
    let generation = GenerationConfig {
        max_tokens: override_or(request.max_tokens, defaults.max_tokens),
        temperature: override_or(request.temperature, defaults.temperature),
        top_k: overrides.top_k.resolve(defaults.top_k),
        top_p: overrides.top_p.resolve(defaults.top_p),
        seed: override_or(overrides.seed, defaults.seed),
        repeat_penalty: override_or(overrides.repeat_penalty, defaults.repeat_penalty),
        repeat_last_n: override_or(overrides.repeat_last_n, defaults.repeat_last_n),
    };
    validate_generation(&generation, Some(vocab_size))?;
    Ok(generation)
}

pub(crate) fn validate_generation(
    generation: &GenerationConfig,
    vocab_size: Option<usize>,
) -> Result<(), CandleError> {
    if generation.max_tokens == 0 {
        return Err(CandleError::InvalidGeneration(
            "max_tokens must be greater than zero".to_string(),
        ));
    }
    if !generation.temperature.is_finite() || generation.temperature < 0.0 {
        return Err(CandleError::InvalidGeneration(
            "temperature must be finite and non-negative".to_string(),
        ));
    }
    if let Some(top_k) = generation.top_k
        && (top_k == 0 || vocab_size.is_some_and(|size| top_k > size))
    {
        return Err(CandleError::InvalidGeneration(
            "top_k must be greater than zero and no larger than the vocabulary".to_string(),
        ));
    }
    if let Some(top_p) = generation.top_p
        && !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0)
    {
        return Err(CandleError::InvalidGeneration(
            "top_p must be finite and in (0, 1]".to_string(),
        ));
    }
    if !generation.repeat_penalty.is_finite() || generation.repeat_penalty <= 0.0 {
        return Err(CandleError::InvalidGeneration(
            "repeat_penalty must be finite and greater than zero".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn sampling(config: &GenerationConfig) -> Sampling {
    if config.temperature == 0.0 {
        Sampling::ArgMax
    } else {
        match (config.top_k, config.top_p) {
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: config.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: config.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: config.temperature,
            },
            (None, None) => Sampling::All {
                temperature: config.temperature,
            },
        }
    }
}

pub(crate) fn effective_output_limit(
    prompt_tokens: usize,
    requested_max_tokens: u64,
    context_limit: usize,
) -> Result<usize, CandleError> {
    if prompt_tokens > context_limit {
        return Err(CandleError::PromptTooLong {
            prompt_tokens,
            context_limit,
        });
    }
    let remaining = context_limit - prompt_tokens;
    if remaining == 0 {
        return Err(CandleError::NoGenerationCapacity {
            prompt_tokens,
            context_limit,
        });
    }
    let remaining = u64::try_from(remaining).map_err(|_| CandleError::NumericConversion {
        field: "remaining_context_tokens",
        value: u64::MAX,
    })?;
    max_tokens_to_usize(requested_max_tokens.min(remaining), usize::MAX as u64)
}

pub(crate) fn max_tokens_to_usize(value: u64, platform_max: u64) -> Result<usize, CandleError> {
    if value > platform_max {
        return Err(CandleError::NumericConversion {
            field: "max_tokens",
            value,
        });
    }
    usize::try_from(value).map_err(|_| CandleError::NumericConversion {
        field: "max_tokens",
        value,
    })
}

pub(crate) fn recent_tokens(tokens: &[u32], repeat_last_n: usize) -> &[u32] {
    tokens
        .get(tokens.len().saturating_sub(repeat_last_n)..)
        .map_or(&[], |recent| recent)
}

pub(crate) fn next_cache_position(
    prompt_tokens: usize,
    generated_index: usize,
) -> Result<usize, CandleError> {
    prompt_tokens
        .checked_add(generated_index)
        .ok_or_else(|| CandleError::Inference("KV-cache position overflowed usize".to_string()))
}

use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Llama};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use candle_transformers::utils::apply_repeat_penalty;
use rig_core::completion::{AssistantContent, CompletionResponse};
use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall};
use tokenizers::tokenizer::DecodeStream;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer,
};
use web_time::{Duration, Instant};

use crate::loader::{LoadedModel, LoadedWeights};
use crate::profile::ModelFamily;
use crate::runtime::{CancellationSignal, check_cancellation};
use crate::types::{CandleCompletionResponse, FinishReason};

type TokenDecodeStream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

enum GenerationStep {
    /// A token was sampled. Some token sequences need more IDs before they decode to valid UTF-8.
    Token(Option<String>),
    /// Generation and incremental decoding are complete.
    Finished(CandleCompletionResponse),
}

pub(crate) struct IncrementalTextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    stream: TokenDecodeStream<'a>,
    token_ids: Vec<u32>,
    text: String,
    flushed: bool,
}

impl<'a> IncrementalTextDecoder<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            stream: tokenizer.decode_stream(true),
            token_ids: Vec::new(),
            text: String::new(),
            flushed: false,
        }
    }

    pub(crate) fn push(&mut self, token: u32) -> Result<Option<String>, CandleError> {
        self.token_ids.push(token);
        let fragment = self
            .stream
            .step(token)
            .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
        if let Some(fragment) = &fragment {
            self.text.push_str(fragment);
        }
        Ok(fragment)
    }

    pub(crate) fn finish(&mut self) -> Result<Option<String>, CandleError> {
        if self.flushed {
            return Ok(None);
        }
        self.flushed = true;
        let fully_decoded = self
            .tokenizer
            .decode(&self.token_ids, true)
            .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
        let suffix = fully_decoded.strip_prefix(&self.text).ok_or_else(|| {
            CandleError::TokenizerDecoding(
                "incremental decoding did not match complete decoding".to_string(),
            )
        })?;
        if suffix.is_empty() {
            Ok(None)
        } else {
            let suffix = suffix.to_string();
            self.text.push_str(&suffix);
            Ok(Some(suffix))
        }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }
}

struct GenerationSession<'a> {
    loaded: &'a LoadedModel,
    generation: GenerationConfig,
    cancellation: &'a CancellationSignal,
    decoder: IncrementalTextDecoder<'a>,
    weights: SessionWeights<'a>,
    logits: Tensor,
    processor: LogitsProcessor,
    prompt_tokens: usize,
    max_tokens: usize,
    effective_max_tokens: u64,
    all_tokens: Vec<u32>,
    generated_tokens: usize,
    finish_reason: Option<FinishReason>,
    started: Instant,
    prefill_duration: Duration,
    time_to_first_token: Option<Duration>,
    delivery_duration: Duration,
}

enum SessionWeights<'a> {
    Safetensors { model: &'a Llama, cache: Cache },
    QuantizedLlama(QuantizedLlama),
    QuantizedQwen3(QuantizedQwen3),
}

impl SessionWeights<'_> {
    fn forward(&mut self, input: &Tensor, position: usize) -> Result<Tensor, CandleError> {
        match self {
            Self::Safetensors { model, cache } => model
                .forward(input, position, cache)
                .and_then(|tensor| tensor.squeeze(0)),
            Self::QuantizedLlama(model) => model
                .forward(input, position)
                .and_then(|tensor| tensor.squeeze(0)),
            Self::QuantizedQwen3(model) => model
                .forward(input, position)
                .and_then(|tensor| tensor.squeeze(0)),
        }
        .map_err(|error| CandleError::Inference(error.to_string()))
    }
}

impl<'a> GenerationSession<'a> {
    pub(crate) fn new(
        loaded: &'a LoadedModel,
        request: CompletionRequest,
        cancellation: &'a CancellationSignal,
    ) -> Result<Self, CandleError> {
        let prompt = crate::protocol::render_prompt(&request, loaded.profile.definition.protocol)?;
        let generation =
            effective_generation(&request, &loaded.generation, loaded.profile.vocab_size)?;
        let encoding = loaded
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| CandleError::TokenizerEncoding(error.to_string()))?;
        let prompt_ids = encoding.get_ids();
        if prompt_ids.is_empty() {
            return Err(CandleError::TokenizerEncoding(
                "the rendered prompt produced no tokens".to_string(),
            ));
        }
        let max_tokens = effective_output_limit(
            prompt_ids.len(),
            generation.max_tokens,
            loaded.profile.context_limit,
        )?;
        let effective_max_tokens =
            u64::try_from(max_tokens).map_err(|_| CandleError::NumericConversion {
                field: "effective_max_tokens",
                value: u64::MAX,
            })?;

        check_cancellation(cancellation)?;
        let started = Instant::now();
        let device = loaded.runtime.device();
        let input = Tensor::new(prompt_ids, device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| CandleError::Inference(error.to_string()))?;
        let mut weights = match &loaded.model {
            LoadedWeights::Safetensors { model, config } => SessionWeights::Safetensors {
                model,
                cache: Cache::new(true, loaded.runtime.cache_dtype(), config, device)
                    .map_err(|error| CandleError::Inference(error.to_string()))?,
            },
            LoadedWeights::QuantizedLlama(model) => SessionWeights::QuantizedLlama(model.clone()),
            LoadedWeights::QuantizedQwen3(model) => SessionWeights::QuantizedQwen3(model.clone()),
        };
        check_cancellation(cancellation)?;
        let logits = weights.forward(&input, 0)?;
        let prefill_duration = started.elapsed();
        let processor = LogitsProcessor::from_sampling(generation.seed, sampling(&generation));

        Ok(Self {
            loaded,
            processor,
            decoder: IncrementalTextDecoder::new(&loaded.tokenizer),
            weights,
            logits,
            prompt_tokens: prompt_ids.len(),
            max_tokens,
            effective_max_tokens,
            all_tokens: prompt_ids.to_vec(),
            generated_tokens: 0,
            finish_reason: None,
            started,
            prefill_duration,
            time_to_first_token: None,
            delivery_duration: Duration::ZERO,
            generation,
            cancellation,
        })
    }

    fn next_token(&mut self) -> Result<GenerationStep, CandleError> {
        check_cancellation(self.cancellation)?;

        if self.finish_reason.is_some() {
            return self.finish();
        }

        if self.generation.repeat_penalty != 1.0 && self.generation.repeat_last_n > 0 {
            let recent = recent_tokens(&self.all_tokens, self.generation.repeat_last_n);
            self.logits =
                apply_repeat_penalty(&self.logits, self.generation.repeat_penalty, recent)
                    .map_err(|error| CandleError::Inference(error.to_string()))?;
        }
        let token = self
            .processor
            .sample(&self.logits)
            .map_err(|error| CandleError::Inference(error.to_string()))?;
        self.generated_tokens = self.generated_tokens.checked_add(1).ok_or_else(|| {
            CandleError::Inference("generated token count overflowed usize".to_string())
        })?;
        if self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(self.started.elapsed());
        }

        if self.loaded.profile.stop_tokens.contains(&token) {
            self.finish_reason = Some(FinishReason::Eos);
            return Ok(GenerationStep::Token(None));
        }

        self.all_tokens.push(token);
        let fragment = self.decoder.push(token)?;

        if self.generated_tokens >= self.max_tokens {
            self.finish_reason = Some(FinishReason::MaxTokens);
        } else {
            check_cancellation(self.cancellation)?;
            let generated_index = self.generated_tokens.saturating_sub(1);
            let position = next_cache_position(self.prompt_tokens, generated_index)?;
            let next = Tensor::new(&[token], self.loaded.runtime.device())
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|error| CandleError::Inference(error.to_string()))?;
            self.logits = self.weights.forward(&next, position)?;
        }

        Ok(GenerationStep::Token(fragment))
    }

    pub(crate) fn finish(&mut self) -> Result<GenerationStep, CandleError> {
        if let Some(suffix) = self.decoder.finish()? {
            return Ok(GenerationStep::Token(Some(suffix)));
        }

        let finish_reason = self.finish_reason.ok_or_else(|| {
            CandleError::Inference("generation finished without a finish reason".to_string())
        })?;
        let prompt_tokens = u64::try_from(self.prompt_tokens).map_err(|_| {
            CandleError::Inference("prompt token count does not fit in u64".to_string())
        })?;
        let generated_tokens = u64::try_from(self.generated_tokens).map_err(|_| {
            CandleError::Inference("generated token count does not fit in u64".to_string())
        })?;
        let generation_duration = self
            .started
            .elapsed()
            .saturating_sub(self.delivery_duration);
        let tokens_per_second = if generation_duration.is_zero() {
            None
        } else {
            Some(generated_tokens as f64 / generation_duration.as_secs_f64())
        };
        Ok(GenerationStep::Finished(CandleCompletionResponse {
            text: self.decoder.text().to_string(),
            prompt_tokens,
            generated_tokens,
            requested_max_tokens: self.generation.max_tokens,
            effective_max_tokens: self.effective_max_tokens,
            finish_reason,
            prefill_duration_ms: duration_millis(self.prefill_duration),
            time_to_first_token_ms: self.time_to_first_token.map(duration_millis),
            generation_duration_ms: duration_millis(generation_duration),
            tokens_per_second,
        }))
    }

    fn record_delivery_duration(&mut self, duration: Duration) {
        self.delivery_duration = self.delivery_duration.saturating_add(duration);
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).map_or(u64::MAX, |value| value)
}

pub(crate) fn generate(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
    mut emit: impl FnMut(String) -> Result<(), CandleError>,
) -> Result<CandleCompletionResponse, CandleError> {
    #[cfg(all(test, not(target_family = "wasm")))]
    if let Some(control) = &loaded.test_control {
        control.enter_generation()?;
    }
    let mut session = GenerationSession::new(loaded, request, cancellation)?;
    loop {
        match session.next_token()? {
            GenerationStep::Token(Some(fragment)) if !fragment.is_empty() => {
                let delivery_started = Instant::now();
                let result = emit(fragment);
                session.record_delivery_duration(delivery_started.elapsed());
                result?;
            }
            GenerationStep::Token(_) => {}
            GenerationStep::Finished(response) => return Ok(response),
        }
    }
}

pub(crate) fn infer(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
) -> Result<InferredCompletion, CandleError> {
    let parse_request = request.clone();
    let mut raw_response = generate(loaded, request, cancellation, |_| Ok(()))?;
    let parsed = crate::protocol::parse_assistant(
        &raw_response.text,
        &parse_request,
        loaded.profile.definition.protocol,
    )?;
    raw_response.text = parsed.visible_text;
    // No emptiness guard here on purpose. One existed, but it was unreachable:
    // the parser fabricated an empty-text part whenever it had nothing, so
    // `items` was never empty. Made reachable, it would reject a real outcome —
    // a model that emits EOS immediately, or only whitespace, which `push_text`
    // trims away — and turn a degenerate-but-successful local generation into a
    // hard error. An empty assistant turn is legal everywhere else now; it is
    // legal here too.
    Ok(InferredCompletion {
        response: raw_response,
        choice: parsed.items,
    })
}

/// One local generation, kept in both the shapes callers need.
///
/// The assistant protocol is parsed exactly once: `response.text` is the
/// visible text with protocol markup stripped, and `choice` holds the items
/// that parse produced. Re-parsing `response.text` would find no tool calls,
/// since stripping already removed them — so both the raw and the normalized
/// path read from this single result.
pub(crate) struct InferredCompletion {
    /// The local model's own response record.
    pub(crate) response: CandleCompletionResponse,
    /// Assistant content parsed out of the generated text.
    pub(crate) choice: Vec<AssistantContent>,
}

impl InferredCompletion {
    /// Normalize into rig's completion response.
    pub(crate) fn into_normalized(self) -> CompletionResponse {
        let usage = (&self.response).into();
        let finish_reason = self.response.finish_reason.into();
        CompletionResponse::new(self.choice, usage, crate::types::PROVIDER_NAME)
            .with_finish_reason(finish_reason)
    }
}

pub(crate) fn stream_generate(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
    mut emit: impl FnMut(RawStreamingChoice<CandleCompletionResponse>) -> Result<(), CandleError>,
) -> Result<CandleCompletionResponse, CandleError> {
    if loaded.profile.definition.protocol != ModelFamily::Qwen3 {
        return generate(loaded, request, cancellation, |fragment| {
            emit(RawStreamingChoice::Message(fragment))
        });
    }

    let parse_request = request.clone();
    // Qwen tool syntax can straddle arbitrary token boundaries. Buffer one
    // model turn so control markup is never leaked as assistant text; complete
    // tool calls are still delivered through Rig's streaming agent driver.
    let mut response = generate(loaded, request, cancellation, |_| Ok(()))?;
    let parsed = crate::protocol::parse_assistant(
        &response.text,
        &parse_request,
        loaded.profile.definition.protocol,
    )?;
    response.text = parsed.visible_text;
    for item in parsed.items {
        match item {
            AssistantContent::Text(text) => emit(RawStreamingChoice::Message(text.text))?,
            AssistantContent::ToolCall(call) => {
                // The envelope-parsed call always carries a wire-issued id
                // (`from_wire` adopted it), so key the stream by it.
                let mut raw = RawStreamingToolCall::new(
                    call.id.as_str().to_owned(),
                    call.function.name,
                    call.function.arguments,
                );
                raw.call_id = None;
                raw.signature = call.signature;
                raw.additional_params = call.additional_params;
                emit(RawStreamingChoice::ToolCall(raw))?;
            }
            AssistantContent::Reasoning(reasoning) => {
                // Same constant-id full-block shape as the gemini adapter's
                // signed-thinking chunk (#2258 F1), but benign here: candle
                // never emits `ReasoningDelta`, so a full block under the
                // shared id has no delta buffer to replace-and-discard.
                for content in reasoning.content {
                    emit(RawStreamingChoice::Reasoning {
                        // Local generation has no wire id; fall back to a
                        // per-stream constant minted identity.
                        id: reasoning
                            .id
                            .clone()
                            .map(rig_core::streaming::StreamPartId::wire)
                            .unwrap_or(rig_core::streaming::StreamPartId::minted(
                                rig_core::streaming::MintKind::Reasoning,
                                0,
                            )),
                        provider_id: reasoning
                            .id
                            .clone()
                            .and_then(rig_core::streaming::WireId::new),
                        content,
                    })?;
                }
            }
            AssistantContent::Image(_) => {
                return Err(CandleError::Inference(
                    "text-only Qwen output parser produced image content".to_string(),
                ));
            }
        }
    }
    Ok(response)
}
