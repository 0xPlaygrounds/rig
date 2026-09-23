//! The recorded round-trip cell: three prompts over deterministic tools, so
//! the assistant turn that carries a signature, encrypted reasoning or
//! parallel tool calls is re-serialized on every later request.
//!
//! The run asserts the task itself: both codes reported, the transient
//! verifier failure recovered, every call answered. [`assert_recorded`]
//! then applies the wire-level rule to the exchanges the cell recorded.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use rig_agent::agent::{AgentBuilder, MultiTurnStreamItem};
use rig_agent::completion::CompletionModel;
use rig_core::message::{
    AssistantContent, ImageMediaType, Message, ReasoningContent, ToolResultContent, UserContent,
};
use rig_core::tool::{Tool, ToolOutput};

use super::{Dialect, Token, continues, lost_tokens, response_tokens, unpaired_tool_calls};

/// The alpha record's code.
pub const ALPHA_CODE: &str = "alpha-code-7431";
/// The beta record's code.
pub const BETA_CODE: &str = "beta-code-2286";

const PREAMBLE: &str = "You are a records assistant. Use the tools for every lookup and \
    verification and never guess a code. When a tool reports a transient failure, call it \
    again once. Answer briefly.";

const RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

/// Which transport a cell drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// One whole reply per model call.
    Unary,
    /// A streamed reply per model call.
    Streaming,
}

impl Transport {
    /// The lowercase label used in exemption tables.
    pub fn label(self) -> &'static str {
        match self {
            Self::Unary => "unary",
            Self::Streaming => "streaming",
        }
    }
}

/// What the wire is expected to deliver, asserted identically on both
/// transports so a streaming decoder that drops a field fails its cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expect {
    /// Reasoning blocks reach history (signed or not).
    pub reasoning: bool,
    /// Reasoning carries a provider signature (Anthropic, Gemini).
    pub signed: bool,
    /// Reasoning carries encrypted content (OpenAI Responses).
    pub encrypted: bool,
    /// The snapshot tool runs and its image lands in a tool result.
    pub image_tool_result: bool,
}

impl Expect {
    /// Tool ids only: no reasoning of any kind.
    pub const TOOLS_ONLY: Self = Self {
        reasoning: false,
        signed: false,
        encrypted: false,
        image_tool_result: false,
    };
    /// Unsigned reasoning text or ids.
    pub const REASONING: Self = Self {
        reasoning: true,
        ..Self::TOOLS_ONLY
    };
    /// Signed reasoning.
    pub const SIGNED: Self = Self {
        signed: true,
        ..Self::REASONING
    };
    /// Provider signatures without required reasoning text: Gemini 3 may
    /// sign every turn and still return no thought summary, so only the
    /// signatures are a stable expectation.
    pub const SIGNATURES: Self = Self {
        signed: true,
        ..Self::TOOLS_ONLY
    };
    /// Encrypted reasoning.
    pub const ENCRYPTED: Self = Self {
        encrypted: true,
        ..Self::REASONING
    };

    /// The same expectation with the snapshot image tool enabled.
    pub const fn with_image(self) -> Self {
        Self {
            image_tool_result: true,
            ..self
        }
    }
}

/// Transport-specific deliveries a wire legitimately omits.
///
/// `(provider, transport, field, reason)`. Every entry needs a cited provider
/// behavior; the census reports an entry no cell relies on as stale.
pub const STREAM_UNARY_DIFFERENCES: &[(&str, &str, &str, &str)] = &[];

fn difference_allowed(provider: &str, transport: Transport, field: &str) -> bool {
    STREAM_UNARY_DIFFERENCES
        .iter()
        .any(|(p, t, f, _)| *p == provider && *t == transport.label() && *f == field)
}

/// One cell: a model, its reasoning parameters, a transport and what to expect.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// The provider's cassette directory.
    pub provider: &'static str,
    /// The model name handed to the provider.
    pub model: &'static str,
    /// Provider-specific request parameters, typically the reasoning switch.
    pub params: fn() -> Option<Value>,
    /// Output budget per model call. Thinking models spend it on reasoning
    /// first, so a wire whose thinking cannot be capped needs more.
    pub max_tokens: u64,
    /// Which transport the cell drives.
    pub transport: Transport,
    /// What the wire must deliver.
    pub expect: Expect,
}

/// What the tools observed.
#[derive(Clone, Debug, Default)]
pub struct Journal {
    /// Records looked up, in call order.
    pub lookups: Vec<String>,
    /// Codes verified, in call order, including the failed attempt.
    pub verifications: Vec<String>,
    /// How many verifications failed transiently.
    pub verify_failures: usize,
    /// How many snapshot images were served.
    pub snapshots: usize,
}

/// The run's observable outcome.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Every committed message across the three prompts.
    pub history: Vec<Message>,
    /// The final prompt's answer.
    pub final_text: String,
    /// The tools' journal.
    pub journal: Journal,
}

/// Shared slot the cell body fills and the post-cassette assertion reads.
pub type Observed = Arc<Mutex<Option<Observation>>>;

#[derive(Debug, thiserror::Error)]
enum ToolFailure {
    #[error("unknown record {0}; the records are alpha and beta")]
    UnknownRecord(String),
    #[error("verifier temporarily unavailable; call verify_code again with the same code")]
    Transient,
}

#[derive(Deserialize)]
struct LookupArgs {
    record: String,
}

struct LookupTool(Arc<Mutex<Journal>>);

impl Tool for LookupTool {
    const NAME: &'static str = "lookup_record";
    type Error = ToolFailure;
    type Args = LookupArgs;
    type Output = String;

    fn description(&self) -> String {
        "Look up a record by name and return its code.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "record": { "type": "string", "description": "alpha or beta" } },
            "required": ["record"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let record = args.record.trim().to_ascii_lowercase();
        self.0.lock().expect("journal").lookups.push(record.clone());
        match record.as_str() {
            "alpha" => Ok(format!("record alpha: code {ALPHA_CODE}")),
            "beta" => Ok(format!("record beta: code {BETA_CODE}")),
            _ => Err(ToolFailure::UnknownRecord(args.record)),
        }
    }
}

#[derive(Deserialize)]
struct VerifyArgs {
    code: String,
}

struct VerifyTool(Arc<Mutex<Journal>>);

impl Tool for VerifyTool {
    const NAME: &'static str = "verify_code";
    type Error = ToolFailure;
    type Args = VerifyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Verify a code returned by lookup_record. May fail transiently; retry once.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "code": { "type": "string", "description": "the code to verify" } },
            "required": ["code"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let mut journal = self.0.lock().expect("journal");
        journal.verifications.push(args.code.clone());
        // The first beta verification fails, once, so the history carries a
        // failed result the model must recover from.
        if args.code == BETA_CODE && journal.verify_failures == 0 {
            journal.verify_failures += 1;
            return Err(ToolFailure::Transient);
        }
        Ok(format!("verified {}", args.code))
    }
}

#[derive(Deserialize)]
struct NoArgs {}

struct SnapshotTool(Arc<Mutex<Journal>>);

impl Tool for SnapshotTool {
    const NAME: &'static str = "reference_snapshot";
    type Error = ToolFailure;
    type Args = NoArgs;
    type Output = ToolOutput;

    fn description(&self) -> String {
        "Return the reference snapshot image for the records.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.lock().expect("journal").snapshots += 1;
        ToolOutput::content(vec![
            ToolResultContent::text("snapshot: one red pixel"),
            ToolResultContent::image_base64(RED_PIXEL_PNG_BASE64, Some(ImageMediaType::PNG), None),
        ])
        .map_err(|_| ToolFailure::Transient)
    }
}

fn prompts(with_image: bool) -> [String; 3] {
    let first = if with_image {
        "Fetch the reference snapshot, then look up the alpha record and the beta record, \
         verify each returned code, and report both codes with their verification status."
    } else {
        "Look up the alpha record and the beta record, verify each returned code, and report \
         both codes with their verification status."
    };
    [
        first.to_owned(),
        "Verify the alpha code once more and confirm whether it is unchanged.".to_owned(),
        "Repeat both codes exactly as reported earlier, separated by a comma, and nothing else."
            .to_owned(),
    ]
}

/// Drive the three-prompt run and return what happened.
pub async fn run<M>(model: M, cell: Cell) -> Observation
where
    M: CompletionModel + 'static,
{
    let journal = Arc::new(Mutex::new(Journal::default()));
    let mut builder = AgentBuilder::new(model)
        .preamble(PREAMBLE)
        .max_tokens(cell.max_tokens)
        .default_max_turns(8);
    if let Some(params) = (cell.params)() {
        builder = builder.additional_params(params);
    }
    let builder = builder
        .tool(LookupTool(Arc::clone(&journal)))
        .tool(VerifyTool(Arc::clone(&journal)));
    let agent = if cell.expect.image_tool_result {
        builder.tool(SnapshotTool(Arc::clone(&journal))).build()
    } else {
        builder.build()
    };

    let mut history: Vec<Message> = Vec::new();
    let mut final_text = String::new();
    for prompt in prompts(cell.expect.image_tool_result) {
        final_text = match cell.transport {
            Transport::Unary => {
                let response = agent
                    .chat(prompt.as_str(), &mut history)
                    .await
                    .unwrap_or_else(|error| panic!("[{}] chat failed: {error}", cell.provider));
                response.output
            }
            Transport::Streaming => {
                let mut stream = agent
                    .prompt(prompt.as_str())
                    .history(history.clone())
                    .max_turns(8)
                    .stream();
                let mut text = None;
                while let Some(item) = stream.next().await {
                    let item = item.unwrap_or_else(|error| {
                        panic!("[{}] stream failed: {error}", cell.provider)
                    });
                    if let MultiTurnStreamItem::FinalResponse(response) = item {
                        text = Some(response.output().to_owned());
                        history.extend(response.messages.unwrap_or_default());
                    }
                }
                text.unwrap_or_else(|| panic!("[{}] stream ended without a final", cell.provider))
            }
        };
    }
    let journal = journal.lock().expect("journal").clone();
    Observation {
        history,
        final_text,
        journal,
    }
}

fn reasoning_blocks(history: &[Message]) -> Vec<&ReasoningContent> {
    history
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(&reasoning.content),
            _ => None,
        })
        .flatten()
        .collect()
}

/// Signatures outside reasoning blocks: on tool calls, and on Gemini answer
/// text.
fn tool_call_signatures(history: &[Message]) -> usize {
    history
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .flatten()
        .filter(|content| match content {
            AssistantContent::ToolCall(call) => call.signature.is_some(),
            AssistantContent::Text(text) => {
                rig_core::providers::gemini::text_thought_signature(text).is_some()
            }
            _ => false,
        })
        .count()
}

/// The task outcome and the normalized history's shape.
pub fn assert_run(cell: Cell, observation: &Observation) {
    let provider = cell.provider;
    let journal = &observation.journal;
    assert!(
        journal.lookups.iter().any(|record| record == "alpha")
            && journal.lookups.iter().any(|record| record == "beta"),
        "[{provider}] both records must be looked up, got {:?}",
        journal.lookups
    );
    assert_eq!(
        journal.verify_failures, 1,
        "[{provider}] exactly one transient verifier failure; lookups {:?}, verifications {:?}",
        journal.lookups, journal.verifications
    );
    assert!(
        journal
            .verifications
            .iter()
            .filter(|code| *code == BETA_CODE)
            .count()
            >= 2,
        "[{provider}] the beta code must be verified again after the transient failure, got {:?}",
        journal.verifications
    );
    assert!(
        journal.verifications.iter().any(|code| code == ALPHA_CODE),
        "[{provider}] the alpha code must be verified, got {:?}",
        journal.verifications
    );
    if cell.expect.image_tool_result {
        assert!(
            journal.snapshots >= 1,
            "[{provider}] the snapshot must be fetched"
        );
    }
    let normalized = observation.final_text.to_ascii_lowercase();
    assert!(
        normalized.contains(ALPHA_CODE) && normalized.contains(BETA_CODE),
        "[{provider}] the final answer must repeat both codes, got {:?}",
        observation.final_text
    );

    // Every call in the normalized history is answered by a result with the
    // same correlation handle, including the failed verification.
    let mut open = Vec::new();
    let mut prompts = 0usize;
    for message in &observation.history {
        match message {
            Message::Assistant { content, .. } => {
                assert!(
                    open.is_empty(),
                    "[{provider}] assistant turn follows unanswered calls {open:?}"
                );
                for content in content {
                    if let AssistantContent::ToolCall(call) = content {
                        open.push(call.id.clone());
                    }
                }
            }
            Message::User { content } => {
                for content in content {
                    match content {
                        UserContent::ToolResult(result) => {
                            let index = open
                                .iter()
                                .position(|id| *id == result.call)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "[{provider}] tool result {:?} answers no open call",
                                        result.call
                                    )
                                });
                            open.remove(index);
                        }
                        UserContent::Text(_) => prompts += 1,
                        _ => {}
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
    assert!(open.is_empty(), "[{provider}] unanswered calls {open:?}");
    assert!(
        prompts >= 3,
        "[{provider}] three prompts expected, got {prompts}"
    );

    let blocks = reasoning_blocks(&observation.history);
    let signed = blocks.iter().any(|block| {
        matches!(
            block,
            ReasoningContent::Text {
                signature: Some(_),
                ..
            }
        )
    }) || tool_call_signatures(&observation.history) > 0;
    let encrypted = blocks
        .iter()
        .any(|block| matches!(block, ReasoningContent::Encrypted(_)));
    let expectations = [
        ("reasoning", cell.expect.reasoning, !blocks.is_empty()),
        ("signed", cell.expect.signed, signed),
        ("encrypted", cell.expect.encrypted, encrypted),
    ];
    for (field, expected, observed) in expectations {
        if expected && !observed && !difference_allowed(provider, cell.transport, field) {
            panic!(
                "[{provider} {}] expected {field} reasoning in the normalized history, found none: {:?}",
                cell.transport.label(),
                blocks
            );
        }
    }
    if cell.expect.image_tool_result {
        let image_results = observation
            .history
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter(|content| {
                matches!(content, UserContent::ToolResult(result)
                    if result.content.iter().any(|item| matches!(item, ToolResultContent::Image(_))))
            })
            .count();
        assert!(
            image_results >= 1,
            "[{provider}] the snapshot image must reach a tool result in history"
        );
    }
}

/// The recorded exchanges: every delivered opaque field reaches every later
/// request of the same conversation, every request pairs its calls, and the
/// expected content kinds were actually delivered.
pub fn assert_recorded(cell: Cell, scenario: &str) {
    let provider = cell.provider;
    let paths = crate::cassettes::recorded_request_paths(provider, scenario);
    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    assert_eq!(
        paths.len(),
        bodies.len(),
        "[{provider}] path per interaction"
    );
    let exchanges: Vec<(Dialect, Value, String)> = paths
        .iter()
        .zip(bodies)
        .map(|(path, (request, response))| {
            let request = serde_json::from_str::<Value>(&request)
                .unwrap_or_else(|error| panic!("[{provider}] request JSON: {error}"));
            (Dialect::from_path(path), request, response)
        })
        .collect();
    assert!(
        exchanges.len() >= 6,
        "[{provider}] a long task needs at least six model calls, recorded {}",
        exchanges.len()
    );

    let mut delivered: Vec<Token> = Vec::new();
    let mut failures = Vec::new();
    for (index, (dialect, request, _)) in exchanges.iter().enumerate() {
        for unpaired in unpaired_tool_calls(*dialect, request) {
            failures.push(format!("request {index}: {unpaired}"));
        }
    }
    // Each response's tokens must survive into every later request of the
    // same conversation, not only the next one.
    for (earlier_index, (dialect, earlier_request, response)) in exchanges.iter().enumerate() {
        let tokens = response_tokens(*dialect, response);
        delivered.extend(tokens.iter().cloned());
        for (later_index, (later_dialect, later_request, _)) in
            exchanges.iter().enumerate().skip(earlier_index + 1)
        {
            if later_dialect != dialect || !continues(*dialect, earlier_request, later_request) {
                continue;
            }
            for token in lost_tokens(*dialect, response, later_request) {
                failures.push(format!(
                    "response {earlier_index} delivered {} {:?}; request {later_index} lacks it",
                    token.kind, token.value
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "[{provider} {scenario}] recorded exchanges:\n{}",
        failures.join("\n")
    );

    let kinds = |kind: &str| delivered.iter().filter(|token| token.kind == kind).count();
    // A wire that always issues ids must have delivered at least two; a wire
    // whose ids are optional must still deliver at least two once it issues
    // any, so a single id is a decoding gap on either.
    let issues_ids = exchanges
        .iter()
        .any(|(dialect, _, _)| dialect.issues_tool_call_ids());
    let ids = kinds("tool_call_id");
    assert!(
        ids >= 2 || (!issues_ids && ids == 0),
        "[{provider}] the provider issued {ids} tool-call ids; at least two were expected"
    );
    let requirements = [
        (
            "signed",
            cell.expect.signed,
            kinds("signature") + kinds("thought_signature") > 0,
        ),
        (
            "encrypted",
            cell.expect.encrypted,
            kinds("encrypted_content") > 0,
        ),
    ];
    for (field, expected, observed) in requirements {
        if expected && !observed && !difference_allowed(provider, cell.transport, field) {
            panic!(
                "[{provider} {}] expected a {field} reasoning field on the wire; none was delivered",
                cell.transport.label()
            );
        }
    }
}
