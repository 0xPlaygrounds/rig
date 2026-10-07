//! The grid's failure rows as data: every way a run ends badly, as a
//! [`Cell`] over the same `Program` table the happy paths use. The
//! per-wire file serves a recorded fault from its cassette. A scripted
//! fault runs here, over the sequenced transport and labelled frames the
//! wire's [`Scripted`] names.

use rig_core::effect::EffectFamily;

use rig_core::effect::EffectFamily::Completion;

use rig_core::effect::EffectFamily::Memory as Mem;

use rig_core::effect::EffectFamily::Tool;

use rig_core::error::ErrorKind;

use super::Wire;
use super::agent::run_agent;
use super::cells::{
    BASIC_PREAMBLE, BASIC_PROMPT, CELL, Cell, Memory, READY_PROMPT, TOOLS_PREAMBLE,
    TWO_TOOL_STREAM_PREAMBLE, ToolKind,
};
use super::corpus::{CONVERSATION, Ending, Program};
use rig_core::http_client::DynHttpClient;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};
use rig_test_support::stream_faults::{SseShape, scripted, sse_bytes};

pub(crate) const BROKEN_ADD_PROMPT: &str = "Use the add tool to add 17 and 25. If the tool reports an error, do not call it again: reply with the error message it gave.";
pub(crate) const BROKEN_ORCHARD_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
If a tool reports an error, do not call it again. \
After both tools have answered, respond in one short sentence that includes the harbor label and the orchard tool's error text.";
/// The setup cells' request: the #2490 stream-fault cells' own.
pub(crate) const SETUP_PROMPT: &str = "Say hi.";
pub(crate) const SETUP_MAX_TOKENS: u64 = 16;
/// The orchard tool's error (`FailingOrchard`).
pub(crate) const BROKEN_ORCHARD: &str = "the orchard lookup is broken";

const C: &[EffectFamily] = &[Completion];
const CT: &[EffectFamily] = &[Completion, Tool];
const CTC: &[EffectFamily] = &[Completion, Tool, Completion];
const CTTC: &[EffectFamily] = &[Completion, Tool, Tool, Completion];
const CCCC: &[EffectFamily] = &[Completion, Completion, Completion, Completion];

const SETUP: Program = Program {
    // No system instruction at all: strict matching distinguishes it from
    // the corpus default's empty one.
    preamble: None,
    prompt: SETUP_PROMPT,
    max_tokens: Some(SETUP_MAX_TOKENS),
    ending: Ending::ProviderError,
    ..Program::DEFAULT
};
const TEXT_STREAM: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: BASIC_PROMPT,
    temperature: Some(0.0),
    streamed: true,
    ..Program::DEFAULT
};
const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: BROKEN_ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

// -- row 1: a setup failure ---------------------------------------------------

/// The wire refuses the request, unary.
pub(crate) const SETUP_UNARY: Cell = Cell {
    name: "fault_setup_unary",
    program: SETUP,
    families: C,
    ..CELL
};
pub(crate) const SETUP_STREAMED: Cell = Cell {
    name: "fault_setup_streamed",
    program: Program {
        streamed: true,
        ..SETUP
    },
    ..SETUP_UNARY
};

// -- row 2: a retryable status ------------------------------------------------

/// A 429 with `Retry-After`: the run fails at once with the report a host
/// needs to retry.
pub(crate) const STATUS_429: Cell = Cell {
    name: "fault_status_429",
    program: SETUP,
    families: C,
    ..CELL
};
pub(crate) const STATUS_503: Cell = Cell {
    name: "fault_status_503",
    ..STATUS_429
};

// -- rows 3, 4, 5: the stream ends badly --------------------------------------

/// EOF after text: a truncation (`Failed(Response)`), the prefix kept,
/// nothing committed.
pub(crate) const TRUNCATED_AFTER_TEXT: Cell = Cell {
    name: "fault_truncated_after_text",
    program: Program {
        ending: Ending::Failed(ErrorKind::Response),
        ..TEXT_STREAM
    },
    events: true,
    families: C,
    ..CELL
};
/// EOF after a complete tool call: the tool never runs.
pub(crate) const TRUNCATED_AFTER_TOOL_CALL: Cell = Cell {
    name: "fault_truncated_after_tool_call",
    program: Program {
        prompt: super::cells::ADD_PROMPT,
        streamed: true,
        ending: Ending::Failed(ErrorKind::Response),
        ..TOOLS
    },
    tools: &[ToolKind::Adder],
    events: true,
    families: C,
    ..CELL
};
/// An error frame after text: the provider's error, the prefix kept, the
/// error item at its position.
pub(crate) const ERROR_AFTER_TEXT: Cell = Cell {
    name: "fault_error_after_text",
    program: Program {
        ending: Ending::ProviderError,
        ..TEXT_STREAM
    },
    events: true,
    families: C,
    ..CELL
};

// -- row 6: a refusal ---------------------------------------------------------

/// The wire's refusal in its own shape: OpenAI streams it as the answer
/// (`refusal` deltas); Gemini refuses the prompt outright
/// (`promptFeedback`), which the per-wire file names as `Failed(Provider)`.
pub(crate) const REFUSAL: Cell = Cell {
    name: "fault_refusal",
    program: TEXT_STREAM,
    events: true,
    families: C,
    ..CELL
};
/// The wire filters the turn after some text (`finish_reason:
/// content_filter`, Gemini's `SAFETY`): the turn failed, so the run fails
/// although text streamed, as replay leaves the turn out.
pub(crate) const FILTERED_WITH_TEXT: Cell = Cell {
    name: "fault_filtered_with_text",
    program: Program {
        ending: Ending::Failed(ErrorKind::Response),
        ..TEXT_STREAM
    },
    ..REFUSAL
};
/// The wire filters the whole turn: no text, `content_filter`. rig-agent
/// fails the run (the model produced no answer).
pub(crate) const FILTERED_EMPTY: Cell = Cell {
    name: "fault_filtered_empty",
    program: Program {
        ending: Ending::Failed(ErrorKind::Response),
        ..TEXT_STREAM
    },
    ..REFUSAL
};

// -- row 7: a tool that errs --------------------------------------------------

pub(crate) const TOOL_ERROR: Cell = Cell {
    name: "fault_tool_error",
    program: TOOLS,
    tools: &[ToolKind::BrokenAdder],
    families: CTC,
    ..CELL
};
pub(crate) const TOOL_ERROR_STREAMED: Cell = Cell {
    name: "fault_tool_error_streamed",
    program: Program {
        streamed: true,
        ..TOOLS
    },
    events: true,
    ..TOOL_ERROR
};

// -- row 8: a batch with one failure ------------------------------------------

const TWO_TOOLS: Program = Program {
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: BROKEN_ORCHARD_PROMPT,
    max_turns: Some(8),
    streamed: true,
    tool_concurrency: Some(1),
    ..Program::DEFAULT
};
/// Two calls in one turn, the second tool errs, concurrency one: both
/// parts in call order, the failing part the error's output.
pub(crate) const BATCH_SECOND_FAILS: Cell = Cell {
    name: "fault_batch_second_fails",
    program: TWO_TOOLS,
    tools: &[ToolKind::Alpha, ToolKind::BrokenBeta],
    events: true,
    families: CTTC,
    ..CELL
};
/// The same over the same recording, concurrency two.
pub(crate) const BATCH_SECOND_FAILS_CONCURRENT: Cell = Cell {
    name: "fault_batch_second_fails_concurrent",
    program: Program {
        tool_concurrency: Some(2),
        ..TWO_TOOLS
    },
    ..BATCH_SECOND_FAILS
};

// -- row 10: a failing store --------------------------------------------------

const MEMORY: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: READY_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    conversation: Some(CONVERSATION),
    ending: Ending::MemoryError,
    ..Program::DEFAULT
};
/// The load fails before any completion: the memory record first and
/// alone, `Failed(Memory)`.
pub(crate) const FAILING_LOAD: Cell = Cell {
    name: "fault_failing_load",
    program: MEMORY,
    memory: Memory::FailingLoad,
    families: &[Mem],
    ..CELL
};
pub(crate) const FAILING_LOAD_STREAMED: Cell = Cell {
    name: "fault_failing_load_streamed",
    program: Program {
        streamed: true,
        ..MEMORY
    },
    ..FAILING_LOAD
};

/// A wire's scripted rows: the faults served by the sequenced transport
/// over labelled frames cut or rewritten from the reply bank's replies of
/// the shapes the wire recorded, each run by the rig-agent runner.
pub(crate) struct Scripted<M> {
    /// The provider directory the frames and the status reply are read from.
    pub(crate) provider: &'static str,
    pub(crate) shape: SseShape,
    /// The streamed text answer the text rows cut, on the wire's own model.
    pub(crate) text_stream: &'static str,
    /// The streamed tool call the tool rows cut, on the wire's own model.
    pub(crate) tool_stream: &'static str,
    /// The recorded setup failure the status rows rewrite.
    pub(crate) setup_reply: &'static str,
    /// The wire over a scripted transport. The client's key must never reach
    /// a recording or a trace.
    pub(crate) wire: fn(DynHttpClient) -> Wire<M>,
}

impl<W, T> Scripted<rig::driver::Model<W, T>>
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    /// The frames of `scenario`'s first reply in the bank.
    pub(crate) fn recorded(&self, scenario: &str) -> Vec<String> {
        let reply = rig_test_support::bank::script(self.provider, scenario)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("{scenario} recorded no reply"));
        rig_test_support::stream_faults::sse_frames(&String::from_utf8_lossy(&reply.body()))
    }

    /// The wire over a transport that answers one streaming request with
    /// `frames`, then EOF.
    pub(crate) fn stream(&self, frames: &[String]) -> Wire<rig::driver::Model<W, T>> {
        (self.wire)(DynHttpClient::new(scripted(vec![sse_bytes(frames)])))
    }

    /// The wire over a transport that answers each unary request with the
    /// next of `replies`.
    fn unary(&self, replies: Vec<MockHttpResponse>) -> Wire<rig::driver::Model<W, T>> {
        (self.wire)(DynHttpClient::new(SequencedHttpClient::new(replies)))
    }

    fn reply(&self, status: u16, retry_after: bool) -> MockHttpResponse {
        let recorded = rig_test_support::bank::script(self.provider, self.setup_reply)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("{} recorded no reply", self.setup_reply));
        let mut headers = rig_core::http_client::HeaderMap::new();
        if retry_after {
            headers.insert(
                "retry-after",
                rig_core::http_client::HeaderValue::from_static("1"),
            );
        }
        MockHttpResponse::ErrorWithHeaders(
            rig_core::http_client::StatusCode::from_u16(status).expect("a status"),
            String::from_utf8_lossy(&recorded.body()).into_owned(),
            headers,
        )
    }

    async fn stream_row(&self, cell: &Cell, frames: Vec<String>) {
        run_agent(&self.stream(&frames), cell, |_| {}).await;
    }

    pub(crate) async fn truncated_after_text(&self) {
        let frames = self.shape.text_prefix(&self.recorded(self.text_stream));
        self.stream_row(&TRUNCATED_AFTER_TEXT, frames).await;
    }

    pub(crate) async fn truncated_after_tool_call(&self) {
        let frames = self.shape.tool_prefix(&self.recorded(self.tool_stream));
        self.stream_row(&TRUNCATED_AFTER_TOOL_CALL, frames).await;
    }

    pub(crate) async fn error_after_text(&self) {
        let frames = self.shape.error_frames(&self.recorded(self.text_stream));
        self.stream_row(&ERROR_AFTER_TEXT, frames).await;
    }

    pub(crate) async fn filtered_with_text(&self) {
        let frames = self.shape.filtered(&self.recorded(self.text_stream), true);
        self.stream_row(&FILTERED_WITH_TEXT, frames).await;
    }

    pub(crate) async fn filtered_empty(&self) {
        let frames = self.shape.filtered(&self.recorded(self.text_stream), false);
        self.stream_row(&FILTERED_EMPTY, frames).await;
    }

    /// Row 10: no request reaches the wire; the transport answers nothing.
    pub(crate) async fn failing_load(&self) {
        self.stream_row(&FAILING_LOAD, Vec::new()).await;
    }

    pub(crate) async fn failing_load_streamed(&self) {
        self.stream_row(&FAILING_LOAD_STREAMED, Vec::new()).await;
    }

    pub(crate) async fn status_429(&self) {
        run_agent(
            &self.unary(vec![self.reply(429, true)]),
            &STATUS_429,
            |_| {},
        )
        .await;
    }

    pub(crate) async fn status_503(&self) {
        run_agent(
            &self.unary(vec![self.reply(503, false)]),
            &STATUS_503,
            |_| {},
        )
        .await;
    }
}

/// `lookup_orchard_label` that fails every call (row 8's second tool).
#[derive(Clone)]
pub(crate) struct FailingOrchard;

impl rig_core::tool::Tool for FailingOrchard {
    const NAME: &'static str = "lookup_orchard_label";
    type Error = rig_core::tool::ToolExecutionError;
    type Args = crate::support::EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return the beta signal marker.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(rig_core::tool::ToolExecutionError::other(BROKEN_ORCHARD))
    }
}
