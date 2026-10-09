//! What the catalog knows about one model, and the checks it makes against
//! [`GenerationOptions`].

use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};

use crate::completion::options::CatalogRefusal;
use crate::completion::{
    CacheRetention, CompletionRequest, Cost, Effort, GenerationOptions, Reasoning,
    UnsupportedOption, Usage,
};
use crate::providers::registry::{Format, ProviderId};

/// One model's facts: limits, input modalities, the reasoning and caching it
/// takes, and its prices. [`Catalog`](super::Catalog) builds one per row of
/// its data; [`ModelSpec::new`] and the `with_*` setters build one in code,
/// for a model the catalog does not list, and
/// [`Catalog::insert`](super::Catalog::insert) adds it.
///
/// It serializes field by field and reads back the same. For the override
/// file's row shape, use [`Self::to_row_json`].
///
/// ```
/// use rig_core::catalog::{Catalog, ModelSpec, Pricing};
/// use rig_core::providers::registry::ProviderId;
///
/// let ollama = ProviderId::catalog("ollama").ok_or("a known vendor")?;
/// let mut catalog = Catalog::builtin().clone();
/// catalog.insert(
///     ModelSpec::new(ollama, "qwen3:4b")
///         .with_context_window(32_768)
///         .with_tools(true)
///         .with_pricing(Pricing::new(0.0, 0.0)),
/// );
/// let qwen = catalog.get(ollama, "qwen3:4b").ok_or("inserted")?.spec;
/// assert_eq!(qwen.context_window, Some(32_768));
/// # Ok::<(), &str>(())
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    /// The provider's own model id.
    pub id: String,
    /// The provider serving the model under [`Self::id`], in the protocol
    /// family the model is reached by: its vendor's own
    /// ([`ProviderId::catalog`]) unless the catalog row names another
    /// under `rig.format`. [`Self::format`] reads it.
    pub provider: ProviderId,
    /// A human-readable name.
    pub display_name: String,
    /// The most tokens the model reads and writes in one request, if known.
    pub context_window: Option<u32>,
    /// The most tokens the model writes in one reply, if known.
    pub max_output_tokens: Option<u32>,
    /// What the model reads.
    pub input: Modalities,
    /// The reasoning the model takes.
    pub reasoning: ReasoningSupport,
    /// The cache retentions the model honours.
    pub caching: CacheSupport,
    /// Whether the model calls tools.
    pub tools: bool,
    /// Whether the model constrains its output to a JSON schema.
    pub structured_output: bool,
    /// Prices in USD per million tokens, if known.
    pub pricing: Option<Pricing>,
    /// Whether the provider has deprecated the model.
    pub deprecated: bool,
    /// When the model takes sampling parameters (`temperature`, `top_p`,
    /// `top_logprobs`, `logprobs`), or `None` when unknown.
    pub sampling: Option<Sampling>,
    /// Wire facts the encoders read that no portable field holds.
    pub compat: Compat,
}

/// What a model reads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Modalities {
    /// Text.
    pub text: bool,
    /// Images.
    pub image: bool,
    /// Audio.
    pub audio: bool,
    /// Video.
    pub video: bool,
    /// PDF documents.
    pub pdf: bool,
}

/// The reasoning a model takes, in one of three states: it does not reason,
/// it reasons but the catalog does not say which controls it takes, or it
/// reasons and takes exactly the listed controls.
///
/// A catalog row that marks a model as reasoning but lists no reasoning
/// options (most of them gateway rows on HuggingFace, OpenRouter, Venice and
/// Bedrock) is [`Self::Unknown`], which refuses nothing, as an empty
/// [`CacheSupport`] does. Only a row whose hand-reviewed `rig` facts say
/// `"reasoning_control": "none"` (a model whose API rejects every reasoning
/// control) is [`Self::Listed`] with nothing listed.
///
/// ```
/// use rig_core::catalog::ReasoningSupport;
/// use rig_core::completion::{Effort, Reasoning};
///
/// let unknown = ReasoningSupport::Unknown { default: None };
/// assert!(unknown.supported() && unknown.levels().is_none());
/// assert_eq!(unknown.refusal(&Reasoning::Effort(Effort::High)), None);
///
/// let listed = ReasoningSupport::Listed {
///     levels: vec![Effort::Low, Effort::High],
///     budget: None,
///     can_disable: false,
///     default: Some(Effort::High),
/// };
/// assert!(listed.refusal(&Reasoning::Off).is_some());
/// assert_eq!(listed.refusal(&Reasoning::Effort(Effort::Low)), None);
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSupport {
    /// The model does not reason.
    #[default]
    None,
    /// The model reasons; the catalog does not say which controls it takes.
    /// Nothing is refused.
    Unknown {
        /// The effort it uses when a request names none, if documented.
        default: Option<Effort>,
    },
    /// The model reasons and takes exactly these controls.
    Listed {
        /// The effort levels it takes, empty when it takes none.
        levels: Vec<Effort>,
        /// The reasoning-token budgets it takes, if it takes one.
        budget: Option<RangeInclusive<u32>>,
        /// Whether a request can turn its reasoning off.
        can_disable: bool,
        /// The effort it uses when a request names none, if documented.
        default: Option<Effort>,
    },
}

/// The cache retentions a model honours.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheSupport {
    /// The [`CacheRetention`] values the model honours. Empty when the
    /// catalog does not know, in which case nothing is refused.
    pub retention: Vec<CacheRetention>,
}

/// Prices in USD per million tokens.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
    /// Uncached input.
    pub input: f64,
    /// Output, reasoning included.
    pub output: f64,
    /// Input read from the cache, or `None` when unknown.
    pub cache_read: Option<f64>,
    /// Input written to the cache, or `None` when unknown.
    pub cache_write: Option<f64>,
}

impl Pricing {
    /// Prices for uncached input and for output, with no cache prices.
    pub fn new(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            cache_read: None,
            cache_write: None,
        }
    }

    /// These prices with `price` for input read from the cache.
    pub fn with_cache_read(mut self, price: f64) -> Self {
        self.cache_read = Some(price);
        self
    }

    /// These prices with `price` for input written to the cache.
    pub fn with_cache_write(mut self, price: f64) -> Self {
        self.cache_write = Some(price);
        self
    }

    /// What `usage` costs at these prices, or `None` unless it reports both
    /// its input and output tokens. Uncached input is the input tokens less
    /// those read from and written to the cache. A cache part with tokens
    /// but no listed price is `None`, and the cost is then not
    /// [complete](Cost::is_complete): its `total` leaves that part out. Every
    /// cache write is charged at one price, so a provider that bills longer
    /// retention higher costs more than this says. It is the standard-tier
    /// list price of the tokens: the service tier, long-context price tiers
    /// and hosted-tool fees (web search, code execution) are not in it.
    pub fn cost(&self, usage: &Usage) -> Option<Cost> {
        let input = usage.input_tokens?;
        let output = usage.output_tokens?;
        let read = usage.cached_input_tokens.unwrap_or(0);
        let written = usage.cache_creation_input_tokens.unwrap_or(0);
        let uncached = input.saturating_sub(read).saturating_sub(written);
        // Token counts stay far below 2^53, so the conversion is exact.
        let price = |tokens: u64, per_million: f64| tokens as f64 * per_million / 1_000_000.0;
        // No tokens cost nothing, whether or not the rate is listed.
        let cache = |tokens: u64, rate: Option<f64>| match (tokens, rate) {
            (0, _) => Some(0.0),
            (tokens, rate) => rate.map(|rate| price(tokens, rate)),
        };
        Some(Cost::priced(
            price(uncached, self.input),
            price(output, self.output),
            cache(read, self.cache_read),
            cache(written, self.cache_write),
        ))
    }
}

/// When a model takes sampling parameters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    /// Always.
    Any,
    /// Only while its reasoning is off.
    ReasoningOff,
    /// Never.
    Never,
}

/// Wire facts the encoders read that no portable field holds. Each defaults
/// to `false` or `None`, which is what a model the field does not concern
/// has. An override file sets each one by its field name under a row's
/// `rig` object, and a spec built in code sets it with the `with_*` setter
/// of the same name.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Compat {
    /// The field an OpenAI Chat Completions assistant message must carry
    /// its reasoning under, such as `reasoning_content`.
    pub reasoning_field: Option<String>,
    /// Anthropic: an effort level goes with `thinking: {"type": "adaptive"}`.
    pub adaptive_thinking: bool,
    /// Anthropic: the `thinking.type` that turns reasoning off, where it is
    /// not `disabled`.
    pub thinking_off: Option<String>,
    /// Anthropic: the model takes `role: "system"` inside `messages`.
    pub mid_conversation_system: bool,
    /// Anthropic: the model answers a forced `tool_choice` with an error.
    pub rejects_forced_tool_choice: bool,
    /// Anthropic: the model binds its thinking to the request's tools and
    /// system prompt.
    pub binds_context: bool,
    /// OpenAI: the prompt cache takes `prompt_cache_options`, not
    /// `prompt_cache_retention`.
    pub prompt_cache_options: bool,
    /// OpenAI: Chat Completions takes the model's tools only while its
    /// reasoning is off.
    pub chat_tools_need_reasoning_off: bool,
}

impl Compat {
    /// These facts with `field` as the Chat Completions assistant message's
    /// reasoning field.
    pub fn with_reasoning_field(mut self, field: impl Into<String>) -> Self {
        self.reasoning_field = Some(field.into());
        self
    }

    /// These facts with adaptive thinking set to `on`.
    pub fn with_adaptive_thinking(mut self, on: bool) -> Self {
        self.adaptive_thinking = on;
        self
    }

    /// These facts with `kind` as the `thinking.type` that turns reasoning off.
    pub fn with_thinking_off(mut self, kind: impl Into<String>) -> Self {
        self.thinking_off = Some(kind.into());
        self
    }

    /// These facts with mid-conversation system messages set to `on`.
    pub fn with_mid_conversation_system(mut self, on: bool) -> Self {
        self.mid_conversation_system = on;
        self
    }

    /// These facts with forced `tool_choice` refusal set to `on`.
    pub fn with_rejects_forced_tool_choice(mut self, on: bool) -> Self {
        self.rejects_forced_tool_choice = on;
        self
    }

    /// These facts with context-bound thinking set to `on`.
    pub fn with_binds_context(mut self, on: bool) -> Self {
        self.binds_context = on;
        self
    }

    /// These facts with `prompt_cache_options` set to `on`.
    pub fn with_prompt_cache_options(mut self, on: bool) -> Self {
        self.prompt_cache_options = on;
        self
    }

    /// These facts with Chat tools needing reasoning off set to `on`.
    pub fn with_chat_tools_need_reasoning_off(mut self, on: bool) -> Self {
        self.chat_tools_need_reasoning_off = on;
        self
    }
}

impl CacheSupport {
    /// Support for exactly the retentions in `retention`. An empty list means
    /// the catalog does not know, and refuses nothing.
    pub fn new(retention: impl IntoIterator<Item = CacheRetention>) -> Self {
        Self {
            retention: retention.into_iter().collect(),
        }
    }
}

impl ModelSpec {
    /// A spec for `provider`'s model `id` with nothing else known: its id as
    /// its name, no limits, text input only, no reasoning, unknown caching,
    /// no tools or structured output, no prices (`None`, never zero), not
    /// deprecated, unknown sampling and default [`Compat`].
    pub fn new(provider: ProviderId, id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            display_name: id.clone(),
            id,
            provider,
            context_window: None,
            max_output_tokens: None,
            input: Modalities {
                text: true,
                ..Modalities::default()
            },
            reasoning: ReasoningSupport::default(),
            caching: CacheSupport::default(),
            tools: false,
            structured_output: false,
            pricing: None,
            deprecated: false,
            sampling: None,
            compat: Compat::default(),
        }
    }

    /// The protocol family a connection to this model speaks unless told
    /// otherwise, or `None` for a provider the registry cannot configure.
    /// To reach the model in another family its vendor is registered for,
    /// build the spec with that [`ProviderId`] or pass the family to
    /// [`ConnectOptions::format`](crate::providers::registry::ConnectOptions::format).
    pub fn format(&self) -> Option<Format> {
        self.provider.format()
    }

    /// The `vendor/model` reference [`Catalog::resolve`](super::Catalog::resolve)
    /// reads back to this model.
    pub fn reference(&self) -> String {
        format!("{}/{}", self.provider.vendor(), self.id)
    }

    /// The options of a request to this model with `reasoning` (`None`
    /// for the provider's default): that reasoning, and the short prompt
    /// cache when [`Self::caching`] lists it. A model whose caching the
    /// catalog does not know gets no cache option, which it could refuse.
    pub fn default_options(&self, reasoning: Option<Reasoning>) -> GenerationOptions {
        let options = GenerationOptions {
            reasoning,
            ..GenerationOptions::default()
        };
        if self.caching.retention.contains(&CacheRetention::Short) {
            options.cache(CacheRetention::Short)
        } else {
            options
        }
    }

    /// This spec named `name`.
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    /// This spec with a context window of `tokens`.
    pub fn with_context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// This spec writing at most `tokens` in one reply.
    pub fn with_max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// This spec reading `input`.
    pub fn with_input(mut self, input: Modalities) -> Self {
        self.input = input;
        self
    }

    /// This spec taking `reasoning`.
    pub fn with_reasoning(mut self, reasoning: ReasoningSupport) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// This spec honouring `caching`.
    pub fn with_caching(mut self, caching: CacheSupport) -> Self {
        self.caching = caching;
        self
    }

    /// This spec with tool calling set to `tools`.
    pub fn with_tools(mut self, tools: bool) -> Self {
        self.tools = tools;
        self
    }

    /// This spec with structured output set to `structured_output`.
    pub fn with_structured_output(mut self, structured_output: bool) -> Self {
        self.structured_output = structured_output;
        self
    }

    /// This spec priced at `pricing`.
    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// This spec marked deprecated or not.
    pub fn with_deprecated(mut self, deprecated: bool) -> Self {
        self.deprecated = deprecated;
        self
    }

    /// This spec taking sampling parameters as `sampling` says.
    pub fn with_sampling(mut self, sampling: Sampling) -> Self {
        self.sampling = Some(sampling);
        self
    }

    /// This spec with the wire facts `compat`.
    pub fn with_compat(mut self, compat: Compat) -> Self {
        self.compat = compat;
        self
    }

    /// This spec as one models.dev-shaped row of an override file, with
    /// rig's facts under `rig`. Read back under its vendor key, the row
    /// builds this spec again. A fact the spec does not know (a `None`
    /// limit, price, sampling rule or reasoning default, or empty caching)
    /// is left out, so laid over another row it keeps that row's value.
    pub fn to_row_json(&self) -> serde_json::Value {
        super::row::Row::from_spec(self).to_json()
    }

    /// Checks `options` against the rules [`Self::refusals`] applies,
    /// and returns the first refusal. It checks only what `options` holds,
    /// so it never refuses `temperature`, which a request carries outside
    /// its options; [`Self::refusals`] checks the whole request.
    ///
    /// # Errors
    ///
    /// The first option the model cannot take. The error names the option,
    /// the provider's vendor and this model.
    pub fn validate(&self, options: &GenerationOptions) -> Result<(), UnsupportedOption> {
        match self.rules(options, None).into_iter().next() {
            Some(refusal) => Err(self.unsupported(refusal)),
            None => Ok(()),
        }
    }

    /// Every option of `request` the model cannot take, by the rules every
    /// completion wire also applies before it encodes, so a request this
    /// refuses is refused by the model's wire too:
    ///
    /// - `reasoning` against [`Self::reasoning`] ([`ReasoningSupport::refusal`]);
    /// - `cache` against [`Self::caching`] ([`CacheSupport::refusal`]);
    /// - `top_p` and `temperature` against [`Self::sampling`]:
    ///   [`Sampling::Never`] refuses both, and [`Sampling::ReasoningOff`]
    ///   refuses both while the model reasons. A model that cannot turn
    ///   reasoning off always reasons; one that can reasons unless
    ///   `reasoning` is `Off`, and with no `reasoning` set, when it names a
    ///   default effort.
    ///
    /// A request whose [`GenerationOptions`] are default is not checked, as
    /// [`GenerationOptions::is_default`] says, so this is empty for it. A
    /// wire refuses more than this: what its route or API cannot carry
    /// (`DynModel::check` reports both).
    pub fn refusals(&self, request: &CompletionRequest) -> Vec<UnsupportedOption> {
        self.request_rules(request)
            .into_iter()
            .map(|refusal| self.unsupported(refusal))
            .collect()
    }

    /// [`Self::refusals`] as the options and reasons the wires report under
    /// their own provider name.
    pub(crate) fn request_rules(&self, request: &CompletionRequest) -> Vec<CatalogRefusal> {
        if request.options.is_default() {
            return Vec::new();
        }
        self.rules(&request.options, request.temperature)
    }

    /// The refusals of `options` and of a request's `temperature`.
    fn rules(&self, options: &GenerationOptions, temperature: Option<f64>) -> Vec<CatalogRefusal> {
        let mut refusals = Vec::new();
        if let Some(reason) = options
            .reasoning
            .as_ref()
            .and_then(|reasoning| self.reasoning.refusal(reasoning))
        {
            refusals.push(CatalogRefusal {
                field: "reasoning",
                reason,
            });
        }
        if let Some(reason) = options
            .cache
            .as_ref()
            .and_then(|cache| self.caching.refusal(cache))
        {
            refusals.push(CatalogRefusal {
                field: "cache",
                reason,
            });
        }
        let reasons = self.reasons_with(options.reasoning.as_ref());
        let sampled = [
            ("temperature", temperature.is_some()),
            ("top_p", options.top_p.is_some()),
        ];
        for (field, set) in sampled {
            if let Some(reason) = set.then(|| self.sampling_refusal(field, reasons)).flatten() {
                refusals.push(CatalogRefusal { field, reason });
            }
        }
        refusals
    }

    /// Why the model takes no `field`, a sampling parameter, when it reasons
    /// as `reasons` says, by [`Self::sampling`].
    pub(crate) fn sampling_refusal(&self, field: &str, reasons: bool) -> Option<String> {
        match self.sampling? {
            Sampling::Never => Some(format!("the model takes no `{field}`")),
            Sampling::ReasoningOff if reasons => {
                Some(match self.reasoning.can_disable() == Some(true) {
                    true => format!(
                        "the model rejects `{field}` while it reasons; remove `{field}` or set \
                         `reasoning` to `Off`"
                    ),
                    false => format!("the model rejects `{field}` while it reasons"),
                })
            }
            Sampling::ReasoningOff | Sampling::Any => None,
        }
    }

    /// Whether the model reasons when a request asks for `reasoning`: at
    /// `Off`, only when it cannot turn reasoning off; with nothing asked, as
    /// [`Self::reasons_by_default`] says.
    pub(crate) fn reasons_with(&self, reasoning: Option<&Reasoning>) -> bool {
        match reasoning {
            Some(Reasoning::Off) => {
                self.reasoning.supported() && self.reasoning.can_disable() == Some(false)
            }
            Some(_) => self.reasoning.supported(),
            None => self.reasons_by_default(),
        }
    }

    /// Whether the model reasons when a request asks for nothing: when it
    /// names a default effort or cannot turn reasoning off.
    pub(crate) fn reasons_by_default(&self) -> bool {
        self.reasoning.supported()
            && (self.reasoning.default_effort().is_some()
                || self.reasoning.can_disable() == Some(false))
    }

    /// `refusal` named for this model.
    fn unsupported(&self, refusal: CatalogRefusal) -> UnsupportedOption {
        UnsupportedOption::new(
            refusal.field,
            self.provider.vendor(),
            &self.id,
            refusal.reason,
        )
    }
}

/// A reasoning setting a model takes, by the name a picker shows: the
/// provider default (`default`, no setting), `off`, an effort level, or a
/// named budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReasoningChoice {
    /// `default`, `off`, the level's word, or the budget's name.
    pub name: &'static str,
    /// The setting a request carries; `None` for the provider default.
    pub reasoning: Option<Reasoning>,
}

impl ReasoningChoice {
    /// The name, with a budget's tokens.
    pub fn label(&self) -> String {
        match self.reasoning {
            Some(Reasoning::Budget { tokens }) => format!("{} ({tokens} tokens)", self.name),
            _ => self.name.to_owned(),
        }
    }
}

/// Token budgets for the named levels on a model that takes a budget
/// instead of levels, clamped into its range.
const NAMED_BUDGETS: [(&str, u32); 3] = [("low", 2048), ("medium", 8192), ("high", 16384)];

impl ReasoningSupport {
    /// The reasoning settings the model takes: the provider default first,
    /// then `off` when reasoning can be turned off, then each effort level,
    /// or `low`, `medium` and `high` budgets on a model that takes a budget
    /// and no levels. A model whose controls the catalog does not list
    /// offers only the default.
    pub fn choices(&self) -> Vec<ReasoningChoice> {
        let mut choices = vec![ReasoningChoice {
            name: "default",
            reasoning: None,
        }];
        let Self::Listed {
            levels,
            budget,
            can_disable,
            ..
        } = self
        else {
            return choices;
        };
        if *can_disable {
            choices.push(ReasoningChoice {
                name: "off",
                reasoning: Some(Reasoning::Off),
            });
        }
        choices.extend(levels.iter().map(|level| ReasoningChoice {
            name: level.as_str(),
            reasoning: Some(Reasoning::Effort(*level)),
        }));
        if levels.is_empty()
            && let Some(range) = budget
        {
            choices.extend(NAMED_BUDGETS.iter().map(|&(name, tokens)| ReasoningChoice {
                name,
                // Not `clamp`, which panics on an inverted range.
                reasoning: Some(Reasoning::Budget {
                    tokens: tokens.max(*range.start()).min(*range.end()),
                }),
            }));
        }
        choices
    }

    /// Whether the model reasons at all.
    pub fn supported(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// The effort levels the model takes: empty when it takes none (or does
    /// not reason), `None` when the catalog does not know, in which case a
    /// picker shows every level.
    pub fn levels(&self) -> Option<&[Effort]> {
        match self {
            Self::None => Some(&[]),
            Self::Unknown { .. } => None,
            Self::Listed { levels, .. } => Some(levels),
        }
    }

    /// The reasoning-token budgets the model takes, if the catalog lists
    /// one. `None` both when it takes no budget and when the catalog does
    /// not know; [`Self::levels`] tells the two apart.
    pub fn budget(&self) -> Option<&RangeInclusive<u32>> {
        match self {
            Self::Listed { budget, .. } => budget.as_ref(),
            _ => None,
        }
    }

    /// Whether a request can turn the model's reasoning off: `Some(false)`
    /// for a model that does not reason, `None` when the catalog does not
    /// know.
    pub fn can_disable(&self) -> Option<bool> {
        match self {
            Self::None => Some(false),
            Self::Unknown { .. } => None,
            Self::Listed { can_disable, .. } => Some(*can_disable),
        }
    }

    /// The effort the model uses when a request names none, if documented.
    pub fn default_effort(&self) -> Option<Effort> {
        match self {
            Self::None => None,
            Self::Unknown { default } | Self::Listed { default, .. } => *default,
        }
    }

    /// Why the model cannot take `reasoning`, or `None` when it can or the
    /// catalog does not know ([`Self::Unknown`]).
    pub fn refusal(&self, reasoning: &Reasoning) -> Option<String> {
        let (levels, budget, can_disable) = match self {
            Self::None => {
                return (!matches!(reasoning, Reasoning::Off))
                    .then(|| "the model does not reason".to_owned());
            }
            Self::Unknown { .. } => return None,
            Self::Listed {
                levels,
                budget,
                can_disable,
                ..
            } => (levels, budget, *can_disable),
        };
        match reasoning {
            Reasoning::Off if can_disable => None,
            Reasoning::Off => Some("reasoning cannot be turned off on this model".to_owned()),
            Reasoning::Effort(effort) if levels.contains(effort) => None,
            Reasoning::Effort(_) if levels.is_empty() && budget.is_some() => {
                Some("the model takes a reasoning budget, not an effort level".to_owned())
            }
            Reasoning::Effort(_) if levels.is_empty() => {
                Some("the model takes no effort level".to_owned())
            }
            Reasoning::Effort(effort) => Some(format!(
                "the model has no `{}` effort level",
                effort.as_str()
            )),
            Reasoning::Budget { tokens } => match budget {
                Some(range) if range.contains(tokens) => None,
                Some(range) => Some(format!(
                    "the model takes a reasoning budget from {} to {} tokens",
                    range.start(),
                    range.end()
                )),
                None if levels.is_empty() => Some("the model takes no reasoning budget".to_owned()),
                None => Some("the model takes an effort level, not a reasoning budget".to_owned()),
            },
        }
    }
}

impl CacheSupport {
    /// Why the model cannot honour `cache`, or `None` when it can or the
    /// catalog does not know.
    pub fn refusal(&self, cache: &CacheRetention) -> Option<String> {
        (!self.retention.is_empty() && !self.retention.contains(cache)).then(|| {
            format!(
                "the model does not honour `{}` cache retention",
                retention_word(cache)
            )
        })
    }
}

/// The lower-case word `cache` serializes as.
pub(super) fn retention_word(cache: &CacheRetention) -> &'static str {
    match cache {
        CacheRetention::None => "none",
        CacheRetention::Short => "short",
        CacheRetention::Long => "long",
    }
}
