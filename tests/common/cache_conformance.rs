//! Cross-provider prompt-cache conformance harness.
//!
//! Rig normalizes `Usage::cached_input_tokens` and
//! `Usage::cache_creation_input_tokens` for a dozen providers, but until this
//! harness existed exactly one of them — Anthropic — had ever been *observed*
//! caching. Every other provider shipped cache accounting no test had seen work.
//!
//! What this harness adds over "assert `cached_input_tokens > 0`" is the
//! distinction that costs real money. Prompt caching is a prefix match, so the
//! interesting failure is not "caching is off", it is "caching silently
//! degraded": a serializer that reorders a map, a loop that rewrites an earlier
//! turn, a tool set re-advertised in a different order. Any of those moves the
//! wire prefix, and a bare `> 0` assertion passes happily while 39,800 of 40,000
//! prefix tokens go uncached. [`assert_hit_ratio`] is the assertion that
//! actually answers the question.
//!
//! # The three-turn probe
//!
//! Every probe runs three turns against one model:
//!
//! 1. **warm** — the padded preamble, tool set, and prompt are sent for the
//!    first time. The provider writes a cache entry (or reads one an unrelated
//!    request already warmed — see [`assert_warms`]).
//! 2. **hit** — byte-identical to turn 1. This is only replayable because the
//!    cassette harness consumes interactions *in order*
//!    (`tests/common/cassettes.rs`, `matching_interaction_index`), so two
//!    identical requests replay two different recorded responses. A cache
//!    scenario must therefore never be marked `.unordered()`.
//! 3. **hit-after-append** — turn 2's assistant reply and a follow-up user turn
//!    are appended, so the prefix strictly *grows*. This is the agent-loop shape,
//!    and [`assert_growth_still_hits`] is the regression nothing in the tree
//!    caught before: a loop that re-normalizes the earlier turns on the way back
//!    in busts the cache from turn 3 onward, silently.
//!
//! # What replay does and does not prove
//!
//! Replay does not merely replay a number. The harness *matches request bodies*
//! (`request_matches`, `tests/common/cassettes.rs`), so a rig change that
//! perturbs the outbound prefix fails as a replay miss with a per-field
//! diagnostic, in CI, with no API key. That is what converts a recorded cassette
//! into a permanent cache regression test.
//!
//! It does have two blind spots, and both are covered elsewhere rather than
//! papered over:
//!
//! * Body matching compares *canonical* JSON (`canonical_json` sorts object
//!   keys), so a change in map iteration order does **not** fail replay — while
//!   it very much does bust a real provider cache. Only the in-process
//!   determinism check in `tests/cassette_cache_prefix.rs` can catch that.
//! * A cassette pins what the provider did at record time. Only the live
//!   economics suite catches the provider changing its cache semantics under us.
#![allow(dead_code)]

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, ToolDefinition, Usage,
};
use rig::message::{Message, UserContent};

use crate::cache_prefix;

/// How a provider reports cached tokens relative to its input-token counter.
///
/// This is the single most likely place to ship a vacuous assertion: divide
/// turn 2's cache read by the wrong turn-1 denominator and
/// [`assert_hit_ratio`] either always passes or always fails. Each variant
/// below cites the mapping code it was derived from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheAccounting {
    /// Cache reads and writes are reported *alongside* `input_tokens`, not
    /// inside it, so the prompt total is the sum of all three.
    ///
    /// Anthropic: `anthropic_usage_totals` computes
    /// `total = input + cached + cache_creation + output`
    /// (`crates/rig-core/src/providers/anthropic/completion.rs`, and the doc
    /// comment above it says so explicitly).
    Alongside,
    /// `cached_input_tokens` is a *subset* of `input_tokens`, so the prompt
    /// total is `input_tokens` on its own.
    ///
    /// * OpenAI chat completions: `prompt_tokens_details.cached_tokens` is a
    ///   breakdown of `prompt_tokens`
    ///   (`crates/rig-core/src/providers/openai/completion/mod.rs`, normalized
    ///   through `providers::internal::completion_usage`).
    /// * OpenAI Responses: `input_tokens_details.cached_tokens` inside
    ///   `input_tokens` (`.../openai/responses_api/mod.rs`).
    /// * Cohere: the field's own doc reads "Subset of `tokens.input_tokens`"
    ///   (`.../cohere/completion.rs`).
    /// * DeepSeek: `prompt_cache_hit_tokens + prompt_cache_miss_tokens` partition
    ///   `prompt_tokens` (`.../deepseek.rs`).
    /// * Gemini: `cachedContentTokenCount` is a breakdown of
    ///   `promptTokenCount` (`.../gemini/completion.rs`).
    /// * OpenRouter, Mistral, and every `openai::Usage` reuser (groq, xai,
    ///   venice, doubleword, perplexity): same OpenAI-compatible shape.
    Subset,
}

/// What a provider's prompt cache can do, and what its numbers mean.
///
/// Passed to every assertion so the assertions themselves stay provider-generic
/// — a new provider is a descriptor, not a new copy of the assertion logic.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheSupport {
    /// Label used in every assertion message.
    pub(crate) provider: &'static str,
    /// See [`CacheAccounting`].
    pub(crate) accounting: CacheAccounting,
    /// Whether the provider reports cache *writes* at all.
    ///
    /// Anthropic and OpenRouter populate `cache_creation_input_tokens`; every
    /// other provider here never does — rig hardcodes it to 0 on those paths
    /// (`providers::internal::completion_usage`), and Gemini's and Cohere's
    /// mappings never set it either.
    ///
    /// Note what each branch of [`assert_warms`] is worth. For a
    /// `reports_writes: true` provider it is a real check that turn 1 created or
    /// read an entry. For the others the paired `== 0` assertion is close to a
    /// tautology, since the field is a hardcoded constant on those paths — it
    /// guards only against a descriptor claiming writes the mapping cannot
    /// produce. The load-bearing part of `assert_warms` for those providers is
    /// the `min_cacheable_tokens` padding check that follows it.
    pub(crate) reports_writes: bool,
    /// Whether the provider needs explicit `cache_control` breakpoints
    /// (Anthropic) or caches long prefixes automatically (everyone else).
    ///
    /// Checked on the wire by [`assert_breakpoints_match_support`], in both
    /// directions: a provider that needs markers must actually receive them on
    /// every turn and within its budget, and a provider that does not must never
    /// be sent them.
    pub(crate) explicit_breakpoints: bool,
    /// Documented minimum cacheable prompt size. Pad above this or the provider
    /// silently declines to cache and the fixture pins a miss.
    pub(crate) min_cacheable_tokens: usize,
    /// Wire field carrying an explicit cache key, if the provider has one
    /// (`prompt_cache_key`, `cachedContent`).
    pub(crate) cache_key_field: Option<&'static str>,
    /// Floor for [`assert_hit_ratio`]. 0.80 unless a provider documents a reason
    /// it cannot reach that — providers cache in block granularity (OpenAI: a
    /// 1024-token minimum in 128-token increments), so the tail of a prompt is
    /// legitimately uncached and this is a floor, never an equality.
    pub(crate) hit_ratio_floor: f64,
}

impl CacheSupport {
    /// The prompt tokens the provider billed for a turn, however it reports them.
    ///
    /// This is [`assert_hit_ratio`]'s denominator. See [`CacheAccounting`].
    pub(crate) fn prompt_tokens(&self, usage: &Usage) -> u64 {
        match self.accounting {
            CacheAccounting::Alongside => {
                usage.input_tokens + usage.cached_input_tokens + usage.cache_creation_input_tokens
            }
            CacheAccounting::Subset => usage.input_tokens,
        }
    }
}

/// Deterministic padding sentence, shared with the Anthropic suite's idiom.
///
/// Padding must be deterministic and committed: a random nonce would guarantee a
/// turn-1 miss but churn the cassette on every re-record and break body
/// matching. The org-pre-warm risk that buys is exactly what [`assert_warms`]
/// tolerates.
const CACHE_PADDING_SENTENCE: &str = "\
This cache fixture paragraph is stable provider test padding about request routing, \
tool schemas, system instructions, and deterministic replay behavior.";

/// Repeat [`CACHE_PADDING_SENTENCE`] `repetitions` times.
///
/// The sentence is ~140 characters, so roughly 26 tokens; 180 repetitions is the
/// ~4,700 prompt tokens the Anthropic suite already uses and clears every
/// provider minimum in the matrix (the largest is Anthropic's own 2,048 for
/// Haiku-class models; OpenAI's is 1,024).
pub(crate) fn cache_padding(repetitions: usize) -> String {
    std::iter::repeat_n(CACHE_PADDING_SENTENCE, repetitions)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Default padding repetitions, matching
/// `tests/providers/anthropic/cassette/prompt_caching.rs`.
pub(crate) const CACHE_PADDING_REPETITIONS: usize = 180;

/// The deterministic three-turn probe.
///
/// One probe, one model, one conversation. Everything is fixed: `temperature: 0`,
/// a committed padded preamble, a fixed tool set in a fixed order, and a prompt
/// that asks for a short deterministic answer.
#[derive(Clone, Debug)]
pub(crate) struct CacheProbe {
    pub(crate) preamble: String,
    pub(crate) tools: Vec<ToolDefinition>,
    pub(crate) prompt: &'static str,
    pub(crate) follow_up: &'static str,
    pub(crate) max_tokens: u64,
    pub(crate) additional_params: Option<serde_json::Value>,
}

/// The prompt every probe sends, unless a provider needs its own wording.
pub(crate) const CACHE_PROBE_PROMPT: &str =
    "Do not call any tools. Reply with exactly these three words: cache probe ready";

/// Stands in for a turn-2 reply that contained no text.
///
/// Deterministic and committed, like the padding: a placeholder that varied
/// between runs would move turn 3's prefix and defeat the probe.
pub(crate) const EMPTY_ASSISTANT_TURN_PLACEHOLDER: &str = "cache probe ready";

/// The follow-up that grows the prefix on turn 3.
pub(crate) const CACHE_PROBE_FOLLOW_UP: &str =
    "Do not call any tools. Reply with exactly these three words: probe still ready";

impl CacheProbe {
    /// A probe padded above `support.min_cacheable_tokens`, labeled so two
    /// providers' fixtures never share a cache entry by accident.
    pub(crate) fn new(label: &'static str) -> Self {
        Self {
            preamble: format!(
                "You are a deterministic cassette test assistant for {label}. \
                 Never call tools for the cache probe prompt; answer only with the \
                 requested phrase.\n{}",
                cache_padding(CACHE_PADDING_REPETITIONS)
            ),
            tools: cache_probe_tools(label),
            prompt: CACHE_PROBE_PROMPT,
            follow_up: CACHE_PROBE_FOLLOW_UP,
            max_tokens: 16,
            additional_params: None,
        }
    }

    /// A probe with no preamble and no tools.
    ///
    /// For explicit-cache scenarios: the cached content owns the system
    /// instruction and the tool set, and a request that also sends its own is
    /// rejected by the provider (and, before that, by rig).
    pub(crate) fn bare(self) -> Self {
        Self {
            preamble: String::new(),
            tools: Vec::new(),
            ..self
        }
    }

    pub(crate) fn with_additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Rebuild the preamble with a different amount of padding.
    ///
    /// Needed where a provider's rate limit is tighter than the default probe:
    /// Groq's free tier allows 8,000 tokens per minute, and three turns of the
    /// default ~4,600-token prompt exceeds that before turn 2 can even be sent.
    pub(crate) fn with_padding(mut self, repetitions: usize, label: &str) -> Self {
        self.preamble = format!(
            "You are a deterministic cassette test assistant for {label}. \
             Never call tools for the cache probe prompt; answer only with the \
             requested phrase.\n{}",
            cache_padding(repetitions)
        );
        self
    }

    /// Build the request for a turn whose chat history is `chat_history`.
    ///
    /// Every field is pinned. `temperature: 0` and a fixed `max_tokens` keep the
    /// response short and the recording cheap; the tool list is cloned in a fixed
    /// order because a reordered tool set is itself one of the prefix moves this
    /// harness exists to catch.
    fn request(&self, mut chat_history: Vec<Message>) -> CompletionRequest {
        chat_history.insert(0, Message::system(self.preamble.clone()));
        CompletionRequest {
            chat_history,
            documents: vec![],
            tools: self.tools.clone(),
            temperature: Some(0.0),
            max_tokens: Some(self.max_tokens),
            tool_choice: None,
            additional_params: self.additional_params.clone(),
            model: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }
}

/// Padding repetitions used inside a tool description.
///
/// Deliberately far smaller than [`CACHE_PADDING_REPETITIONS`]. The bulk of the
/// cacheable prefix lives in the preamble, which every provider in the matrix
/// renders ahead of the conversation; tool descriptions carry only enough
/// padding to make the tools block non-trivial. OpenAI-compatible APIs bound how
/// long a function description may be, and blowing that limit would fail the
/// request outright rather than tell us anything about caching.
const TOOL_PADDING_REPETITIONS: usize = 3;

/// A probe's two fixed tools.
///
/// Two rather than one, in a fixed order: a re-ordered tool set is itself one of
/// the prefix moves this harness exists to catch, and a single-element list
/// cannot be observably re-ordered.
pub(crate) fn cache_probe_tools(label: &str) -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "lookup_cache_policy".to_string(),
            description: format!(
                "Return {label} internal prompt cache policy notes. {}",
                cache_padding(TOOL_PADDING_REPETITIONS)
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": { "type": "string", "description": "Policy topic to look up." }
                },
                "required": ["topic"]
            }),
        },
        ToolDefinition {
            name: "lookup_cache_fixture".to_string(),
            description: format!(
                "Return {label} prompt cache fixture notes. {}",
                cache_padding(TOOL_PADDING_REPETITIONS)
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "fixture": { "type": "string", "description": "Fixture identifier to look up." }
                },
                "required": ["fixture"]
            }),
        },
    ]
}

/// What the three turns reported.
///
/// Deliberately holds usage only, not the outbound request bodies. The bodies
/// are needed — [`assert_prefix_stable`], [`assert_cache_key_stable`] and
/// [`assert_breakpoints_match_support`] all read them — but they are read back
/// from the fixture the cassette wrapper wrote rather than carried here, because
/// the cassette harness exposes no in-process copy in *either* mode: on replay
/// the incoming bytes are dropped after matching, and in record mode httpmock
/// owns them until the fixture is exported. Reading the fixture gets the bytes
/// that actually went over the wire, works identically in both modes, and needs
/// no change to the cassette core; the cost is that those three assertions run
/// after the wrapper closure returns rather than inside it.
#[derive(Clone, Debug, Default)]
pub(crate) struct CacheObservation {
    pub(crate) turns: Vec<Usage>,
}

impl CacheObservation {
    fn turn(&self, index: usize, support: &CacheSupport) -> &Usage {
        self.turns.get(index).unwrap_or_else(|| {
            panic!(
                "[{}] probe recorded {} turns; turn {} is missing — the probe did not complete",
                support.provider,
                self.turns.len(),
                index + 1
            )
        })
    }

    /// A one-line-per-turn dump, printed by every failing assertion so the
    /// numbers that produced the failure are in the failure itself.
    pub(crate) fn report(&self, support: &CacheSupport) -> String {
        self.turns
            .iter()
            .enumerate()
            .map(|(index, usage)| {
                format!(
                    "  turn {}: prompt={} input={} cached={} created={} output={} total={}",
                    index + 1,
                    support.prompt_tokens(usage),
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.cache_creation_input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Run the three-turn blocking probe.
///
/// Turn 3 appends turn 2's *real* assistant reply rather than a synthetic one:
/// re-admitting the provider's own response is what exercises rig's
/// message-normalization path, and a normalizer that rewrites the assistant turn
/// on the way back in is precisely the loop-level prefix move
/// [`assert_growth_still_hits`] is looking for.
pub(crate) async fn run_cache_probe<M>(model: &M, probe: &CacheProbe) -> CacheObservation
where
    M: CompletionModel,
{
    let opening = Message::User {
        content: vec![UserContent::text(probe.prompt)],
    };

    let first = send(model, probe, vec![opening.clone()], "turn 1 (warm)").await;
    let second = send(model, probe, vec![opening.clone()], "turn 2 (hit)").await;

    let assistant = Message::Assistant {
        id: second.message_id.clone(),
        content: second.choice.clone(),
    };
    let follow_up = Message::User {
        content: vec![UserContent::text(probe.follow_up)],
    };
    let third = send(
        model,
        probe,
        vec![opening, assistant, follow_up],
        "turn 3 (hit after append)",
    )
    .await;

    CacheObservation {
        turns: vec![first.usage, second.usage, third.usage],
    }
}

async fn send<M>(
    model: &M,
    probe: &CacheProbe,
    chat_history: Vec<Message>,
    label: &str,
) -> CompletionResponse
where
    M: CompletionModel,
{
    model
        .completion(probe.request(chat_history))
        .await
        .unwrap_or_else(|error| panic!("cache probe {label} should succeed: {error}"))
}

/// The streamed twin of [`run_cache_probe`].
///
/// Worth recording separately rather than assumed to match the blocking path:
/// cache counters arrive on a *different* frame on every streaming wire (an
/// Anthropic `message_start`, an OpenAI terminal usage chunk), and the streaming
/// accumulator has to carry them forward to the final response. Cache usage
/// being dropped or overwritten specifically on the streaming path is a real bug
/// class — see the carry-forward logic in
/// `crates/rig-core/src/providers/anthropic/streaming.rs` — and only a streamed
/// probe can see it.
pub(crate) async fn run_cache_probe_streaming<M>(model: &M, probe: &CacheProbe) -> CacheObservation
where
    M: CompletionModel,
{
    let opening = Message::User {
        content: vec![UserContent::text(probe.prompt)],
    };

    let (first_usage, _, _) =
        stream_turn(model, probe, vec![opening.clone()], "turn 1 (warm)").await;
    let (second_usage, text, message_id) =
        stream_turn(model, probe, vec![opening.clone()], "turn 2 (hit)").await;

    // A model can legitimately produce no *text* within the probe's small
    // output budget — a reasoning model may spend all of it on thinking, which
    // Venice's qwen3 route does. Replaying an empty assistant message then fails
    // the next request outright ("Text content cannot be empty"), which would
    // report a provider input-validation error as a caching result. Substitute a
    // fixed, committed string so turn 3 stays valid and byte-stable.
    let assistant_text = if text.trim().is_empty() {
        EMPTY_ASSISTANT_TURN_PLACEHOLDER.to_owned()
    } else {
        text
    };
    let assistant = Message::Assistant {
        id: message_id,
        content: vec![rig::message::AssistantContent::text(&assistant_text)],
    };
    let follow_up = Message::User {
        content: vec![UserContent::text(probe.follow_up)],
    };
    let (third_usage, _, _) = stream_turn(
        model,
        probe,
        vec![opening, assistant, follow_up],
        "turn 3 (hit after append)",
    )
    .await;

    CacheObservation {
        turns: vec![first_usage, second_usage, third_usage],
    }
}

/// Drive one streamed turn, returning its final usage, accumulated text, and
/// message id.
async fn stream_turn<M>(
    model: &M,
    probe: &CacheProbe,
    chat_history: Vec<Message>,
    label: &str,
) -> (Usage, String, Option<String>)
where
    M: CompletionModel,
{
    use futures::StreamExt;
    use rig::streaming::StreamedAssistantContent;

    let mut stream = model
        .stream(probe.request(chat_history))
        .await
        .unwrap_or_else(|error| panic!("streamed cache probe {label} should start: {error}"));

    let mut text = String::new();
    let mut usage = None;

    while let Some(item) = stream.next().await {
        match item
            .unwrap_or_else(|error| panic!("streamed cache probe {label} should succeed: {error}"))
        {
            StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
            StreamedAssistantContent::Final(response) => usage = Some(response.usage),
            _ => {}
        }
    }

    let usage = usage.unwrap_or_else(|| {
        panic!(
            "streamed cache probe {label} produced no final usage — the accumulator dropped it, \
             which is exactly the streaming-path bug class this probe exists to catch"
        )
    });
    (usage, text, stream.message_id.clone())
}

/// Turn 1 must create a cache entry, or read one that was already warm.
///
/// Deliberately *not* tightened to "must be a write". Provider caches are
/// org-scoped and this padding is committed and deterministic, so unrelated
/// traffic — a previous recording session, another test, another machine on the
/// same key — legitimately pre-warms the entry and turn 1 reads instead of
/// writes. Tightening this is how a suite starts failing for reasons that have
/// nothing to do with rig.
///
/// Providers that cannot report writes at all (`reports_writes: false`, i.e.
/// everything rig normalizes through `completion_usage`) may legitimately report
/// zero for both counters on turn 1: their first turn is the miss that populates
/// the cache and they say nothing about it. For those, turn 1 is only required
/// not to *claim* a write it cannot produce.
pub(crate) fn assert_warms(observation: &CacheObservation, support: &CacheSupport, context: &str) {
    let turn = observation.turn(0, support);

    if support.reports_writes {
        assert!(
            turn.cache_creation_input_tokens > 0 || turn.cached_input_tokens > 0,
            "[{}] {context}: turn 1 should create or read cache tokens — this provider reports \
             writes, so a turn that does neither means nothing was cached.\n{}",
            support.provider,
            observation.report(support)
        );
    } else {
        assert_eq!(
            turn.cache_creation_input_tokens,
            0,
            "[{}] {context}: descriptor says this provider cannot report cache writes, but turn 1 \
             reported {} — the descriptor or the usage mapping is wrong.\n{}",
            support.provider,
            turn.cache_creation_input_tokens,
            observation.report(support)
        );
    }

    assert!(
        support.prompt_tokens(turn) as usize >= support.min_cacheable_tokens,
        "[{}] {context}: turn 1 billed only {} prompt tokens, below this provider's documented \
         {}-token cacheable minimum — the probe is under-padded and the fixture would pin a miss \
         no matter what rig does.\n{}",
        support.provider,
        support.prompt_tokens(turn),
        support.min_cacheable_tokens,
        observation.report(support)
    );
}

/// Turn 2 must read cached tokens.
pub(crate) fn assert_hits(observation: &CacheObservation, support: &CacheSupport, context: &str) {
    let turn = observation.turn(1, support);
    assert!(
        turn.cached_input_tokens > 0,
        "[{}] {context}: turn 2 is byte-identical to turn 1 and must read cached tokens, but read \
         none. Either the provider declined to cache (check the padding against \
         min_cacheable_tokens) or rig moved the wire prefix between two requests that should have \
         been identical.\n{}",
        support.provider,
        observation.report(support)
    );
}

/// **The assertion that answers the question**: turn 2's cache read must cover
/// at least `hit_ratio_floor` of turn 1's billed prompt.
///
/// A bare `cached_input_tokens > 0` passes while 39,800 of 40,000 prefix tokens
/// go uncached. This is the one that does not.
pub(crate) fn assert_hit_ratio(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    let warm = observation.turn(0, support);
    let hit = observation.turn(1, support);

    let denominator = support.prompt_tokens(warm);
    assert!(
        denominator > 0,
        "[{}] {context}: turn 1 billed zero prompt tokens, so a hit ratio is undefined — the \
         usage mapping reported nothing.\n{}",
        support.provider,
        observation.report(support)
    );

    let ratio = hit.cached_input_tokens as f64 / denominator as f64;
    assert!(
        ratio >= support.hit_ratio_floor,
        "[{}] {context}: turn 2 read {} cached tokens against turn 1's {} billed prompt tokens — \
         a {:.1}% hit ratio, below this provider's {:.0}% floor. Caching is *on* but degraded: \
         most of the prefix is being re-billed on every turn.\n{}",
        support.provider,
        hit.cached_input_tokens,
        denominator,
        ratio * 100.0,
        support.hit_ratio_floor * 100.0,
        observation.report(support)
    );
}

/// Turn 3 grew the prefix and must still be *mostly served from cache*.
///
/// This is the agent-loop regression that costs real money and that nothing in
/// the tree caught before this harness. A driver that rewrites, reorders, or
/// re-normalizes an earlier turn between iterations busts the cache from that
/// point on, and because the counters stay non-zero a `> 0` assertion never
/// notices.
///
/// Stated as a **ratio** rather than "turn 3 read at least as many tokens as
/// turn 2". Providers cache in coarse blocks, so the absolute count wobbles by a
/// few tokens as the prefix grows and the block boundaries re-align — Gemini was
/// measured going 3,765 -> 3,760 across an append that added 21 tokens. A
/// monotonic assertion fails on that noise while still passing a genuine partial
/// move; the ratio is both robust to the noise and the thing that actually
/// determines the bill.
pub(crate) fn assert_growth_still_hits(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    let grown = observation.turn(2, support);
    let denominator = support.prompt_tokens(grown);
    assert!(
        denominator > 0,
        "[{}] {context}: turn 3 billed zero prompt tokens, so its hit ratio is undefined.\n{}",
        support.provider,
        observation.report(support)
    );

    let ratio = grown.cached_input_tokens as f64 / denominator as f64;
    assert!(
        ratio >= support.hit_ratio_floor,
        "[{}] {context}: turn 3 appended an assistant turn and a user turn, so every byte turn 2 \
         already sent is still there and should still be cached — yet turn 3 read {} of its {} \
         billed prompt tokens, a {:.1}% hit ratio against this provider's {:.0}% floor. Something \
         rewrote an earlier turn on the way back in, which busts the cache for the rest of the \
         conversation.\n{}",
        support.provider,
        grown.cached_input_tokens,
        denominator,
        ratio * 100.0,
        support.hit_ratio_floor * 100.0,
        observation.report(support)
    );
}

/// Every assertion that applies to a plain three-turn probe.
pub(crate) fn assert_cache_conformance(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    assert_warms(observation, support, context);
    assert_hits(observation, support, context);
    assert_hit_ratio(observation, support, context);
    assert_growth_still_hits(observation, support, context);
}

/// Apply the corpus-wide prefix rule to the three turns this scenario just
/// recorded, reading the fixture the cassette wrapper wrote.
///
/// This runs *after* the cassette wrapper returns, because the fixture is
/// written by `ProviderCassette::finish`. That timing is deliberate: it asserts
/// on the bytes that actually went over the wire rather than on a re-serialized
/// approximation of them, and it works identically in record and replay mode.
///
/// It overlaps `tests/cassette_cache_prefix.rs`'s corpus sweep on purpose, and
/// adds two things the sweep cannot: it fails during the recording session
/// rather than on the next full run, and — unlike the sweep, which skips
/// endpoints it does not model — it *requires* this scenario's endpoint to be
/// modeled. An unmodeled cache-bearing endpoint is a finding, not a skip.
pub(crate) fn assert_prefix_stable(provider: &str, scenario: &str) {
    let interactions = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    assert!(
        interactions.len() >= 2,
        "[{provider}] {scenario}: a cache scenario must record at least two requests to compare; \
         got {}",
        interactions.len()
    );

    let path = recorded_request_paths(provider, scenario);
    let mut previous: Option<(String, Vec<cache_prefix::PrefixBlock>)> = None;
    let mut compared = 0usize;

    for (index, ((request, _), request_path)) in interactions.iter().zip(path.iter()).enumerate() {
        // GET and DELETE carry no body. A cache scenario that manages a resource
        // records those alongside its turns, and there is nothing in them to
        // compare.
        if request.trim().is_empty() {
            previous = None;
            continue;
        }
        let body: serde_json::Value = serde_json::from_str(request).unwrap_or_else(|error| {
            panic!("[{provider}] {scenario}: recorded request {index} should be JSON: {error}")
        });
        // A scenario may legitimately mix conversation turns with
        // resource-management calls — an explicit-cache cell creates and deletes
        // a `cachedContents` handle around its turns. Those carry no
        // conversation to compare, so they are skipped rather than modeled.
        // An endpoint that is neither modeled nor recognised as
        // non-conversational is still a finding.
        if cache_prefix::classify_endpoint(request_path)
            == cache_prefix::EndpointKind::NotConversational
        {
            previous = None;
            continue;
        }
        let blocks = cache_prefix::canonical_prefix_blocks(request_path, &body).unwrap_or_else(|| {
            panic!(
                "[{provider}] {scenario}: request {index} speaks {request_path}, which the cache \
                 prefix rule does not model. A cache-bearing endpoint that cannot be modeled is a \
                 finding, not a skip — add it to `canonical_prefix_blocks` in \
                 tests/common/cache_prefix.rs."
            )
        });

        if let Some((previous_path, previous_blocks)) = previous.take()
            && previous_path == *request_path
            && cache_prefix::continues_the_same_conversation(&previous_blocks, &blocks)
        {
            compared += 1;
            if let Some(violation) =
                cache_prefix::compare(scenario, index, &previous_blocks, &blocks)
            {
                panic!(
                    "[{provider}] {scenario}: the probe moved its own cache wire prefix between \
                     turns, which busts the cache on every turn:\n{violation}"
                );
            }
        }
        previous = Some((request_path.clone(), blocks));
    }

    assert!(
        compared > 0,
        "[{provider}] {scenario}: no growing same-endpoint turn pair was compared, so this check \
         proved nothing. A three-turn cache probe must produce at least one — turn 3 appends to \
         turn 2's conversation."
    );
}

/// The request path each recorded interaction targeted, in wire order.
///
/// `recorded_interaction_bodies` returns bodies only, and the prefix rule is
/// keyed on the endpoint, so the paths are read alongside them.
fn recorded_request_paths(provider: &str, scenario: &str) -> Vec<String> {
    let path = crate::cassettes::cassette_path(provider, scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    let mut paths = Vec::new();
    let mut in_request = false;
    for line in contents.lines() {
        if line == "when:" {
            in_request = true;
        } else if line == "then:" {
            in_request = false;
        } else if in_request && let Some(value) = line.trim_start().strip_prefix("path: ") {
            paths.push(value.trim().to_string());
        }
    }
    paths
}

/// Where the provider exposes an explicit cache key, it must reach the wire and
/// be identical on all three turns.
///
/// A key that changes between turns partitions the cache and guarantees a miss,
/// which is indistinguishable from "caching is off" in the usage numbers alone —
/// so it is checked against the recorded request bodies, not the response.
pub(crate) fn assert_cache_key_stable(provider: &str, scenario: &str, support: &CacheSupport) {
    let Some(field) = support.cache_key_field else {
        return;
    };

    let interactions = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let mut seen: Option<serde_json::Value> = None;

    let paths = recorded_request_paths(provider, scenario);
    for (index, ((request, _), path)) in interactions.iter().zip(paths.iter()).enumerate() {
        // Only conversation turns carry a cache key. A scenario that manages a
        // cache resource also records the create/get/update/delete calls — those
        // are bodiless or address the collection, and asserting a per-turn key on
        // them would fail for a reason that has nothing to do with caching.
        if request.trim().is_empty()
            || cache_prefix::classify_endpoint(path) != cache_prefix::EndpointKind::Modeled
        {
            continue;
        }
        let body: serde_json::Value = serde_json::from_str(request).unwrap_or_else(|error| {
            panic!("[{provider}] {scenario}: recorded request {index} should be JSON: {error}")
        });
        let value = body.get(field).unwrap_or_else(|| {
            panic!(
                "[{provider}] {scenario}: request {index} does not carry the cache key field \
                 `{field}` — the descriptor says this provider has one, so either rig is not \
                 sending it or the descriptor is wrong. Body keys: {:?}",
                body.as_object().map(|map| map.keys().collect::<Vec<_>>())
            )
        });

        match &seen {
            None => seen = Some(value.clone()),
            Some(first) => assert_eq!(
                first, value,
                "[{provider}] {scenario}: cache key `{field}` changed between turns \
                 ({first} -> {value}). A key that moves partitions the cache and guarantees a miss."
            ),
        }
    }

    assert!(
        seen.is_some(),
        "[{provider}] {scenario}: no conversation turn was inspected, so the cache key was never \
         checked. This assertion proved nothing."
    );
}

// ---------------------------------------------------------------------------
// The agent-loop probe
// ---------------------------------------------------------------------------

/// The tool a cache probe's agent run calls.
///
/// Deterministic by construction: the answer depends only on the argument, so
/// the tool result text — which becomes part of the next turn's cached prefix —
/// is byte-stable across recordings.
pub(crate) struct CacheProbeLookupTool;

#[derive(Debug, thiserror::Error)]
#[error("cache probe lookup failed")]
pub(crate) struct CacheProbeLookupError;

#[derive(serde::Deserialize)]
pub(crate) struct CacheProbeLookupArgs {
    pub(crate) topic: String,
}

impl rig::tool::Tool for CacheProbeLookupTool {
    const NAME: &'static str = "lookup_cache_policy";
    type Error = CacheProbeLookupError;
    type Args = CacheProbeLookupArgs;
    type Output = String;

    fn description(&self) -> String {
        "Look up one prompt-cache policy note by topic. Must be called for every policy question."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Policy topic to look up."}
            },
            "required": ["topic"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "Policy note for {}: prefix caching is a byte-exact prefix match.",
            args.topic
        ))
    }
}

/// The prompt driving the agent probe.
///
/// Asks for two lookups so the run is guaranteed to make a tool round-trip, and
/// therefore at least two completion calls — the minimum for a growth assertion
/// to say anything. Models that batch tool calls answer both lookups in one
/// parallel turn, which is fine: the second model call's prefix still contains
/// the first turn's assistant message and both tool results.
pub(crate) const AGENT_CACHE_PROMPT: &str = "\
Look up the cache policy for the topic 'prefix', then, in a separate tool call, \
look up the cache policy for the topic 'breakpoint'. Call the tool once per \
topic, one after the other. Then reply with exactly these three words: cache \
probe ready";

/// Turn an agent run's per-call usage into a [`CacheObservation`].
pub(crate) fn observation_from_completion_calls(
    calls: &[rig::agent::CompletionCall],
) -> CacheObservation {
    CacheObservation {
        turns: calls.iter().map(|call| call.usage).collect(),
    }
}

/// Across a real agent loop, every turn must stay mostly served from cache.
///
/// Each iteration appends an assistant turn and a tool result, so the cacheable
/// prefix only ever grows. A driver that rewrites, reorders, drops, or
/// re-normalizes an earlier turn between iterations busts the cache from that
/// point on, and because the counters stay non-zero a `> 0` assertion never
/// notices. This is the regression that costs real money in production loops.
///
/// Like [`assert_growth_still_hits`], the invariant is a ratio rather than a
/// monotonic token count: block re-alignment makes the absolute number drift
/// down slightly as the prefix grows, and failing on that noise would say
/// nothing about caching.
pub(crate) fn assert_agent_growth_still_hits(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    // Two is the true minimum: one tool round-trip is two model calls, and the
    // second one's prefix already contains the first turn's assistant message
    // and the tool results, so the append-and-still-hit property is observable.
    // Asking for three would only be satisfiable on models that refuse to batch
    // tool calls — gpt-4o-mini answers the probe's two lookups in a single
    // parallel turn — and would fail for a reason that has nothing to do with
    // caching.
    assert!(
        observation.turns.len() >= 2,
        "[{}] {context}: an agent cache probe needs at least two completion calls for a growth \
         assertion to mean anything, got {}. The model answered without ever calling the \
         tool.\n{}",
        support.provider,
        observation.turns.len(),
        observation.report(support)
    );

    let mut ever_hit = false;
    for (index, usage) in observation.turns.iter().enumerate() {
        let denominator = support.prompt_tokens(usage);
        let ratio = if denominator == 0 {
            0.0
        } else {
            usage.cached_input_tokens as f64 / denominator as f64
        };

        if ever_hit {
            assert!(
                ratio >= support.hit_ratio_floor,
                "[{}] {context}: turn {} read {} of its {} billed prompt tokens — a {:.1}% hit \
                 ratio against this provider's {:.0}% floor — after an earlier turn had already \
                 hit. The loop only ever appends, so the cacheable prefix cannot shrink; \
                 something rewrote an earlier turn between iterations.\n{}",
                support.provider,
                index + 1,
                usage.cached_input_tokens,
                denominator,
                ratio * 100.0,
                support.hit_ratio_floor * 100.0,
                observation.report(support)
            );
        }
        if ratio >= support.hit_ratio_floor {
            ever_hit = true;
        }
    }

    assert!(
        ever_hit,
        "[{}] {context}: no turn of the agent run cleared the {:.0}% hit-ratio floor. Either the \
         loop moves the prefix on every iteration or the provider declined to cache this \
         conversation.\n{}",
        support.provider,
        support.hit_ratio_floor * 100.0,
        observation.report(support)
    );
}

/// Pin a provider that does **not** do meaningful prefix caching.
///
/// Deliberately not `assert_eq!(cached, 0)`. A provider can report a small
/// non-zero count that has nothing to do with the prefix under test — enough to
/// satisfy the `cached_input_tokens > 0` assertion this whole harness exists to
/// replace, while nearly all of the prompt is re-billed every turn. Stating the
/// property as a ratio covers that case and a flat zero with one rule.
///
/// Checks **every** turn, not just the byte-identical repeat. An earlier version
/// looked only at turns 1 and 2 and therefore could not see a cache that warms
/// late — which is exactly what Cohere does (1.8% -> 16.4% -> 98.9% across the
/// three turns). It reported Cohere as having no prefix cache and would have
/// gone on passing forever. Folding over all turns is what makes the
/// self-invalidating claim below actually true.
///
/// This is the self-invalidating half of the coverage story: a provider recorded
/// here is opted out of the full conformance suite, and this assertion is what
/// makes that opt-out testable. The day the provider ships real prefix caching,
/// on any turn, this fails and says so rather than leaving the opt-out to rot.
pub(crate) fn assert_no_meaningful_prefix_cache(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    assert_probe_is_padded_enough(observation, support, context);

    for (index, usage) in observation.turns.iter().enumerate() {
        let turn_denominator = support.prompt_tokens(usage);
        let ratio = if turn_denominator == 0 {
            0.0
        } else {
            usage.cached_input_tokens as f64 / turn_denominator as f64
        };
        assert!(
            ratio < support.hit_ratio_floor,
            "[{}] {context}: this provider is recorded as doing no meaningful prefix caching, but \
             turn {} read {} of its {turn_denominator} billed prompt tokens — a {:.1}% hit ratio. \
             That is good news: it does cache. Replace this cell with the full \
             `assert_cache_conformance` suite (or `assert_cache_warms_over_turns` if it warms \
             late) and drop the provider's coverage opt-out.\n{}",
            support.provider,
            index + 1,
            usage.cached_input_tokens,
            ratio * 100.0,
            observation.report(support)
        );
    }
}

/// Pin a provider whose cache is real but **warms across turns**.
///
/// Cohere is the case this exists for, and finding it corrected a wrong
/// conclusion in an earlier revision of this branch. Its recorded *blocking*
/// probe reads 112 of 6,058 prompt tokens on turn 1 (1.8%), 992 on a
/// byte-identical turn 2 (16.4%), and 6,016 of 6,085 on the grown turn 3
/// (98.9%); the streaming probe warms the same way, to 6,048 of 6,085 (99.4%).
/// So Cohere caches properly — it just takes a couple of turns to get there,
/// which the strict three-turn conformance (turn 2 must already clear the floor)
/// would fail.
///
/// The property pinned here is **"once warm, stays warm"**: once any turn clears
/// the provider's floor, every later turn must too, and the final turn must.
/// That is deliberately *not* a monotonic token count, for the same reason
/// [`assert_agent_growth_still_hits`] is not — providers cache in coarse blocks,
/// so a flat cached count against a growing prompt makes the ratio drift down a
/// fraction of a percent with nothing wrong (6,016/6,058 then 6,016/6,085 is
/// 99.31% -> 98.87%). Failing on that would be failing on arithmetic. A real
/// prefix move collapses the ratio to near zero and trips this immediately.
///
/// Note that turn 2 alone reading 16.4% is precisely the case a bare
/// `cached_input_tokens > 0` assertion reports as "caching works".
pub(crate) fn assert_cache_warms_over_turns(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    assert_probe_is_padded_enough(observation, support, context);

    let ratios: Vec<f64> = observation
        .turns
        .iter()
        .map(|usage| {
            let denominator = support.prompt_tokens(usage);
            if denominator == 0 {
                0.0
            } else {
                usage.cached_input_tokens as f64 / denominator as f64
            }
        })
        .collect();

    let mut warm = false;
    for (index, ratio) in ratios.iter().enumerate() {
        if warm {
            assert!(
                *ratio >= support.hit_ratio_floor,
                "[{}] {context}: turn {} fell back to a {:.1}% hit ratio after an earlier turn had \
                 already warmed past this provider's {:.0}% floor. The prefix only ever appends, so \
                 a warm cache cannot cool down — something rewrote an earlier turn.\n{}",
                support.provider,
                index + 1,
                ratio * 100.0,
                support.hit_ratio_floor * 100.0,
                observation.report(support)
            );
        }
        if *ratio >= support.hit_ratio_floor {
            warm = true;
        }
    }

    let final_ratio = *ratios.last().unwrap_or(&0.0);
    assert!(
        final_ratio >= support.hit_ratio_floor,
        "[{}] {context}: this provider's cache is recorded as warming over turns to at least \
         {:.0}%, but the final turn only reached {:.1}%.\n{}",
        support.provider,
        support.hit_ratio_floor * 100.0,
        final_ratio * 100.0,
        observation.report(support)
    );
}

/// Turn 1 must bill enough prompt tokens for any conclusion about caching to
/// mean anything.
///
/// Shared so that an under-padded re-record reports *that* — rather than a
/// downstream ratio failure whose real cause is a probe too small for the
/// provider to have cached it in the first place.
fn assert_probe_is_padded_enough(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    let denominator = support.prompt_tokens(observation.turn(0, support));
    assert!(
        denominator > 0,
        "[{}] {context}: turn 1 billed zero prompt tokens, so nothing can be concluded.\n{}",
        support.provider,
        observation.report(support)
    );
    assert!(
        denominator as usize >= support.min_cacheable_tokens,
        "[{}] {context}: turn 1 billed only {denominator} prompt tokens, below the {}-token floor \
         this provider would need to cache anything. The probe is under-padded, so nothing about \
         its caching can be concluded.\n{}",
        support.provider,
        support.min_cacheable_tokens,
        observation.report(support)
    );
}

/// Pin a provider whose cache fires, but not reliably turn-over-turn.
///
/// Mistral is the case this exists for. Across repeated live passes over an
/// identical 1,783-token prefix it produced, in different runs: a hit on turn 2
/// (1,760 tokens, 98.7%), a hit on turn 1 only, and no hit at all — with rig
/// sending byte-identical requests every time. The plausible cause is routing
/// without cache affinity, and either way it is not something rig controls.
///
/// So the strict [`assert_cache_conformance`] would be a coin flip, and pinning
/// "turn 2 always hits" by re-recording until a good run landed would be
/// cherry-picking. What *is* true, deterministic, and worth protecting is
/// narrower: when the provider reports a cache read, rig surfaces it, at full
/// magnitude, on both the blocking and streaming paths. That is a claim about
/// rig's usage mapping rather than about the provider's hit rate, and it is what
/// this asserts.
///
/// The provider's actual hit rate belongs to the live economics suite, which
/// measures it against the real API instead of a frozen recording.
pub(crate) fn assert_cache_read_is_surfaced(
    observation: &CacheObservation,
    support: &CacheSupport,
    context: &str,
) {
    let best = observation
        .turns
        .iter()
        .map(|usage| {
            let denominator = support.prompt_tokens(usage);
            if denominator == 0 {
                0.0
            } else {
                usage.cached_input_tokens as f64 / denominator as f64
            }
        })
        .fold(0.0_f64, f64::max);

    assert!(
        best >= support.hit_ratio_floor,
        "[{}] {context}: no turn surfaced a cache read of at least {:.0}% of its billed prompt. \
         This provider's cache is intermittent, so a single cold turn is expected — but the \
         recorded fixture is supposed to contain a turn that *did* hit, which is what proves rig \
         maps the provider's cached-token field at all. Re-record until one does.\n{}",
        support.provider,
        support.hit_ratio_floor * 100.0,
        observation.report(support)
    );
}

/// Print one row of the live economics table, and assert conformance.
///
/// A cassette pins what a provider did at record time; only a live run catches
/// the provider changing its cache semantics under us. These cells are the
/// standing check for that, and the row they print is what the PR's economics
/// table is built from — so the table can be regenerated by anyone with keys
/// rather than trusted as a one-off transcription.
pub(crate) fn report_and_assert_live(
    observation: &CacheObservation,
    support: &CacheSupport,
    scenario: &str,
) {
    let warm = observation.turn(0, support);
    let hit = observation.turn(1, support);
    let grown = observation.turn(2, support);
    let denominator = support.prompt_tokens(warm);
    let ratio = if denominator == 0 {
        0.0
    } else {
        hit.cached_input_tokens as f64 / denominator as f64
    };

    eprintln!(
        "LIVE-CACHE-ECONOMICS | {:<11} | {:<26} | t1 prompt {:>6} | t1 write {:>6} | \
         t2 read {:>6} | ratio {:>6.1}% | t3 read {:>6}",
        support.provider,
        scenario,
        denominator,
        warm.cache_creation_input_tokens,
        hit.cached_input_tokens,
        ratio * 100.0,
        grown.cached_input_tokens,
    );

    assert_cache_conformance(observation, support, scenario);
}

/// Anthropic's documented ceiling on `cache_control` breakpoints per request.
///
/// Exceeding it is a request error, not a silent degradation, but a marker
/// *budget* that creeps upward as a conversation grows is the kind of thing that
/// only shows up on turn N of a long run — so it is asserted per turn rather
/// than assumed.
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// Check `cache_control` breakpoint placement on the recorded wire.
///
/// Covers the breakpoint-placement bug class directly, which no usage counter
/// can: markers dropped when a knob combination is set, markers past the
/// provider's limit, or markers sent to a provider that does not understand
/// them. Reads the fixture the cassette wrapper wrote, for the same reason
/// [`assert_prefix_stable`] does — it asserts on the bytes that actually went
/// over the wire rather than on a re-serialized approximation.
///
/// Asserted in both directions so the descriptor itself is testable:
///
/// * `explicit_breakpoints: true` — every turn must carry at least one marker
///   (a turn with none is caching silently switched off) and no more than
///   [`MAX_CACHE_BREAKPOINTS`].
/// * `explicit_breakpoints: false` — no turn may carry a marker. Rig sending
///   `cache_control` to a provider whose API does not define it either errors
///   the request or, worse, is accepted and ignored while looking like caching
///   is configured.
pub(crate) fn assert_breakpoints_match_support(
    provider: &str,
    scenario: &str,
    support: &CacheSupport,
) {
    let interactions = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    assert!(
        !interactions.is_empty(),
        "[{provider}] {scenario}: no recorded interactions to check breakpoints against"
    );

    for (index, (request, _)) in interactions.iter().enumerate() {
        let body: serde_json::Value = serde_json::from_str(request).unwrap_or_else(|error| {
            panic!("[{provider}] {scenario}: recorded request {index} should be JSON: {error}")
        });
        let markers = count_cache_control(&body);

        if support.explicit_breakpoints {
            assert!(
                markers > 0,
                "[{provider}] {scenario}: turn {} carries no `cache_control` marker, but this \
                 provider only caches where rig places one. Caching is silently off for this \
                 turn — the usage counters cannot tell you that, because a warm org cache can \
                 still report a read.",
                index + 1
            );
            assert!(
                markers <= MAX_CACHE_BREAKPOINTS,
                "[{provider}] {scenario}: turn {} carries {markers} `cache_control` markers, past \
                 the documented limit of {MAX_CACHE_BREAKPOINTS}. The provider rejects the \
                 request outright.",
                index + 1
            );
        } else {
            assert_eq!(
                markers,
                0,
                "[{provider}] {scenario}: turn {} carries {markers} `cache_control` marker(s), but \
                 this provider's API does not define the field. It is either an error or, worse, \
                 accepted and ignored while looking like caching is configured.",
                index + 1
            );
        }
    }
}

/// Every `cache_control` key anywhere in a request body.
fn count_cache_control(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(map) => {
            let here = usize::from(map.contains_key("cache_control"));
            here + map.values().map(count_cache_control).sum::<usize>()
        }
        serde_json::Value::Array(items) => items.iter().map(count_cache_control).sum(),
        _ => 0,
    }
}
