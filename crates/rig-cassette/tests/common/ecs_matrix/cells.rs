//! The grid's cells as data: every cell the Anthropic families pin, with
//! the rig-cassette corpus's `Program` for the run and what a live cell needs
//! beside it. The programs are the corpus's own tables (`crates/rig-cassette/
//! tests/corpus_*.rs`), verbatim, so the same cell means the same thing on
//! every wire and in every interpreter.

use rig_core::effect::EffectFamily;

use rig_core::effect::EffectFamily::Completion;

use rig_core::effect::EffectFamily::Custom;

use rig_core::effect::EffectFamily::Memory as Mem;

use rig_core::effect::EffectFamily::Tool;

use super::corpus::{
    CANCEL_ADD_DISPATCH, CANCEL_ADD_OUTCOME, CANCEL_ANSWER, CONVERSATION, Choice, Ending, Hook,
    LATE_ROUTE, LayerAt, LayerKind, LayerSpec, NESTING, Nesting, Output, Program, REPLACED_ANSWER,
    ROUTE, STOP_AFTER_TURN, STOP_AT_ANSWER, STOP_ON_TEXT_DELTA, STOP_ON_TOOL_CALL_DELTA,
};

pub(crate) const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
pub(crate) const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
pub(crate) const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
pub(crate) const ADD_PROMPT: &str =
    "Use the add tool to add 17 and 25, then reply with just the number.";
#[allow(dead_code)] // the Gemini wire reuses the breadth recording, whose essay is shorter
pub(crate) const ESSAY_PROMPT: &str =
    "Write a 600-word essay on the history of the Rust programming language.";
pub(crate) const NOTE_PREAMBLE: &str =
    "You are a note-taking assistant. Use the write_note tool to save notes.";
pub(crate) const NOTE_PROMPT: &str = "Save a note titled 'Rust' whose body is a 400-word essay on the history of the Rust programming language, then reply with just the word saved.";
pub(crate) const READY_PROMPT: &str = "Reply with the single word: ready.";
pub(crate) const SECOND_PROMPT: &str = "Now reply with the single word: again.";
pub(crate) const NAME_PROMPT: &str = "What is my name? Reply with just the name.";
pub(crate) const CONTEXT_PROMPT: &str = "What is a glarb-glarb? Answer in one sentence.";
pub(crate) const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
pub(crate) const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";
pub(crate) const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
pub(crate) const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
pub(crate) const LOOKUP_PREAMBLE: &str =
    "You are a research assistant. Use the lookup tool to answer.";
pub(crate) const LOOKUP_PROMPT: &str = "Use the lookup tool with q set to exactly \"What is the capital of France?\" and reply with just the lookup result.";
pub(crate) const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

pub(crate) fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

/// A Qwen 3.5 wire asked not to think (`reasoning_effort: none`), so a
/// tiny cap cuts an answer rather than hidden tokens (Doubleword's wire).
#[allow(dead_code)]
pub(crate) fn reasoning_off() -> serde_json::Value {
    serde_json::json!({ "reasoning_effort": "none" })
}

/// When the matrix asks the model to think. Existing cells default to Off
/// without changing their programs; the explicit off row opts into rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Thinking {
    On,
    #[default]
    Off,
    SecondTurnOnly,
}

/// The six request dialects, independent of the provider model's Rust type.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "each provider test target constructs only its own dialect"
)]
pub(crate) enum ThinkingWire {
    Anthropic,
    OpenAiChat,
    OpenAiResponses,
    Gemini,
    DeepSeek,
    Doubleword,
    Venice,
}

/// The reasoning matrix's contract shape, asserted beside record parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReasoningCase {
    Text,
    Tool,
    Output,
    Off,
    Capped,
}

fn openai_chat_thinking(on: bool) -> serde_json::Value {
    serde_json::json!({"reasoning_effort": if on { "low" } else { "minimal" }})
}

fn openai_responses_thinking(on: bool) -> serde_json::Value {
    if on {
        serde_json::json!({
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"]
        })
    } else {
        serde_json::json!({"reasoning": {"effort": "minimal"}})
    }
}

fn gemini_thinking(on: bool) -> serde_json::Value {
    serde_json::json!({"generationConfig": {"thinkingConfig": {
        "includeThoughts": on, "thinkingBudget": if on { 128 } else { 0 }
    }}})
}

fn deepseek_thinking(on: bool) -> serde_json::Value {
    serde_json::json!({"thinking": {"type": if on { "enabled" } else { "disabled" }}})
}

fn doubleword_thinking(on: bool) -> serde_json::Value {
    if on {
        // An empty object would leave SecondTurnOnly's baseline `none`
        // intact under RequestPatch's shallow merge.
        openai_chat_thinking(true)
    } else {
        reasoning_off()
    }
}

fn venice_thinking(on: bool) -> serde_json::Value {
    use rig_core::providers::venice::VeniceParameters;

    if on {
        VeniceParameters::default().strip_thinking_response(false)
    } else {
        VeniceParameters::default().disable_thinking(true)
    }
    .into_additional_params()
}

impl ThinkingWire {
    /// A function pointer fits the corpus's static programs. The closures
    /// delegate to one rendering per wire, so on/off cannot drift between
    /// a builder's defaults and a second-turn patch.
    pub(crate) fn params(self, on: bool) -> fn() -> serde_json::Value {
        match (self, on) {
            (Self::Anthropic, true) => {
                || serde_json::json!({"thinking": {"type": "enabled", "budget_tokens": 1024}})
            }
            (Self::Anthropic, false) => || serde_json::json!({"thinking": {"type": "disabled"}}),
            (Self::OpenAiChat, true) => || openai_chat_thinking(true),
            (Self::OpenAiChat, false) => || openai_chat_thinking(false),
            (Self::OpenAiResponses, true) => || openai_responses_thinking(true),
            (Self::OpenAiResponses, false) => || openai_responses_thinking(false),
            (Self::Gemini, true) => || gemini_thinking(true),
            (Self::Gemini, false) => || gemini_thinking(false),
            (Self::DeepSeek, true) => || deepseek_thinking(true),
            (Self::DeepSeek, false) => || deepseek_thinking(false),
            (Self::Doubleword, true) => || doubleword_thinking(true),
            (Self::Doubleword, false) => || doubleword_thinking(false),
            (Self::Venice, true) => || venice_thinking(true),
            (Self::Venice, false) => || venice_thinking(false),
        }
    }
}

pub(crate) fn bypass_history() -> Vec<rig_core::message::Message> {
    super::corpus::bypass_history()
}

/// The tools a cell grants, in registration order (`golden/tool:<name>#<n>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolKind {
    LongTask,
    CheckpointStep,
    CheckpointBatch,
    CheckpointLarge,
    /// The long tool loop's repository (`super::long_loop`): `list_files`.
    RepoListFiles,
    /// `read_file`.
    RepoReadFile,
    /// `write_file`.
    RepoWriteFile,
    /// `run_tests`.
    RepoRunTests,
    /// `add` (`crate::support::Adder`).
    Adder,
    /// `lookup_harbor_label`.
    Alpha,
    /// `lookup_orchard_label`.
    Beta,
    /// `write_note` (`crate::goldens::WriteNote`).
    WriteNote,
    /// `lookup`, the nesting tool of Matrix Q (served by the world; over a
    /// tool server by the agent).
    Lookup,
    /// `add` that fails every call (`crate::goldens::FailingAdd`).
    BrokenAdder,
    /// `lookup_orchard_label` that fails every call
    /// (`super::faults::FailingOrchard`).
    BrokenBeta,
}

/// The conversation store a cell remembers in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum Memory {
    None,
    InMemory,
    /// `FailingMemory::append_fails()`: the append is refused.
    FailingAppend,
    /// `FailingMemory::load_fails()`: the load is refused.
    FailingLoad,
}

/// How the cell is served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bus {
    /// The agent's own bus under the default policy (the header names it).
    Own,
    /// The agent's own bus, `serial_per_handler`.
    Serial,
    /// The agent's own bus, every capacity at one.
    CapacityOne,
    /// A host's bus (the header names no policy).
    Host,
    /// A host's bus, `serial_per_handler`.
    HostSerial,
}

impl Bus {
    pub(crate) fn policy(self) -> rig_core::serve::ServingPolicy {
        let default = rig_core::serve::ServingPolicy::default();
        match self {
            Self::Own | Self::Host => default,
            Self::Serial | Self::HostSerial => rig_core::serve::ServingPolicy {
                serial_per_handler: true,
                ..default
            },
            Self::CapacityOne => rig_core::serve::ServingPolicy {
                command_capacity: 1,
                stream_capacity: 1,
                serial_per_handler: false,
            },
        }
    }

    /// Whether the header names the policy (the agent's own bus).
    pub(crate) fn declared(self) -> bool {
        matches!(self, Self::Own | Self::Serial | Self::CapacityOne)
    }
}

/// One cell of the grid.
#[derive(Clone, Copy)]
pub(crate) struct Cell {
    /// `<family>_<cell>`: the golden is `<wire>_<name>`.
    pub(crate) name: &'static str,
    pub(crate) program: Program,
    pub(crate) thinking: Thinking,
    /// Render Off explicitly for the reasoning-off row. Ordinary existing
    /// cells keep their original additional parameters byte-for-byte.
    pub(crate) explicit_thinking_off: bool,
    pub(crate) reasoning: Option<ReasoningCase>,
    /// The image the first prompt carries, and how (`super::image`).
    pub(crate) image: Option<super::image::ImageCell>,
    pub(crate) tools: &'static [ToolKind],
    pub(crate) memory: Memory,
    pub(crate) bus: Bus,
    /// The host serves `host/note`.
    pub(crate) notes: bool,
    /// The recorder keeps stream events (`EffectLogRecorder::keeping_stream_events`).
    pub(crate) events: bool,
    /// The record's families, in order.
    pub(crate) families: &'static [EffectFamily],
    /// Save a scene after this many tool turns' results and resume in a
    /// fresh world over the log's tail (CONTRACT §13).
    pub(crate) resume_after: Option<usize>,
    /// The cut is resumed in a fresh world over fresh live adapters and
    /// the same cassette (the head world sends the head, only the restored
    /// world can send the tail) rather than over replayers of the log's
    /// tail; `resume_after` names the cut.
    pub(crate) live_resume: bool,
    /// The fault the cell drives, if it is a failure-row cell
    /// (`super::faults`).
    pub(crate) fault: Option<super::faults::Fault>,
    /// The scene the cell saves beside its cut (`super::faults::Scene`).
    pub(crate) scene: super::faults::Scene,
    /// The agent's provider-retry budget (CONTRACT §5); `None` is the
    /// library's default.
    pub(crate) provider_retries: Option<usize>,
}

pub(crate) const CELL: Cell = Cell {
    name: "",
    program: Program::DEFAULT,
    thinking: Thinking::Off,
    explicit_thinking_off: false,
    reasoning: None,
    image: None,
    tools: &[],
    memory: Memory::None,
    bus: Bus::Own,
    notes: false,
    events: false,
    families: &[],
    resume_after: None,
    live_resume: false,
    fault: None,
    scene: super::faults::Scene::None,
    provider_retries: None,
};

const C: &[EffectFamily] = &[Completion];
const CC: &[EffectFamily] = &[Completion, Completion];
const CT: &[EffectFamily] = &[Completion, Tool];
const CTC: &[EffectFamily] = &[Completion, Tool, Completion];
const CTTC: &[EffectFamily] = &[Completion, Tool, Tool, Completion];
const MCTCM: &[EffectFamily] = &[Mem, Completion, Tool, Completion, Mem];

const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: BASIC_PROMPT,
    temperature: Some(0.0),
    ..Program::DEFAULT
};
const ADD: Cell = Cell {
    program: TOOLS,
    tools: &[ToolKind::Adder],
    families: CTC,
    ..CELL
};
const TEXT: Cell = Cell {
    program: BASIC,
    families: C,
    ..CELL
};

// -- Reasoning, on the same six wires ----------------------------------------

pub(crate) const REASONING_TEXT_UNARY: Cell = Cell {
    name: "reasoning_text_unary",
    thinking: Thinking::On,
    reasoning: Some(ReasoningCase::Text),
    program: Program {
        max_tokens: Some(4096),
        ..BASIC
    },
    ..TEXT
};

pub(crate) const REASONING_TEXT_STREAMED: Cell = Cell {
    name: "reasoning_text_streamed",
    program: Program {
        streamed: true,
        ..REASONING_TEXT_UNARY.program
    },
    events: true,
    ..REASONING_TEXT_UNARY
};

pub(crate) const REASONING_TOOL_UNARY: Cell = Cell {
    name: "reasoning_tool_unary",
    thinking: Thinking::On,
    reasoning: Some(ReasoningCase::Tool),
    program: Program {
        max_tokens: Some(4096),
        ..TOOLS
    },
    resume_after: Some(1),
    live_resume: true,
    ..ADD
};

pub(crate) const REASONING_TOOL_STREAMED: Cell = Cell {
    name: "reasoning_tool_streamed",
    program: Program {
        streamed: true,
        ..REASONING_TOOL_UNARY.program
    },
    events: true,
    resume_after: None,
    live_resume: false,
    ..REASONING_TOOL_UNARY
};

pub(crate) const REASONING_OFF: Cell = Cell {
    name: "reasoning_off",
    thinking: Thinking::Off,
    explicit_thinking_off: true,
    reasoning: Some(ReasoningCase::Off),
    ..REASONING_TEXT_UNARY
};

pub(crate) const REASONING_CAPPED: Cell = Cell {
    name: "reasoning_capped",
    reasoning: Some(ReasoningCase::Capped),
    program: Program {
        max_tokens: Some(16),
        hooks: &[Hook::RecordSettled],
        ending: Ending::Failed(rig_core::error::ErrorKind::Response),
        ..REASONING_TEXT_UNARY.program
    },
    provider_retries: Some(0),
    ..REASONING_TEXT_UNARY
};

#[allow(
    dead_code,
    reason = "only Responses and Gemini require a streamed cap twin"
)]
pub(crate) const REASONING_CAPPED_STREAMED: Cell = Cell {
    name: "reasoning_capped_streamed",
    program: Program {
        streamed: true,
        ..REASONING_CAPPED.program
    },
    events: true,
    ..REASONING_CAPPED
};

// -- Images: the user's content, verbatim, through record, history and scene --

use super::image::{ImageCase, ImageCell, ImageSource};

#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
const IMAGE_TEXT: Program = Program {
    preamble: Some(super::image::IMAGE_PREAMBLE),
    prompt: super::image::COLOR_PROMPT,
    temperature: Some(0.0),
    max_tokens: Some(2048),
    ..Program::DEFAULT
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
const IMAGE_TOOLS: Program = Program {
    preamble: Some(super::image::IMAGE_TOOL_PREAMBLE),
    prompt: super::image::IMAGE_TOOL_PROMPT,
    temperature: Some(0.0),
    max_tokens: Some(2048),
    max_turns: Some(3),
    ..Program::DEFAULT
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
const INLINE_TEXT: ImageCell = ImageCell {
    case: ImageCase::Text,
    source: ImageSource::Inline,
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
const INLINE_TOOL: ImageCell = ImageCell {
    case: ImageCase::Tool,
    source: ImageSource::Inline,
};

#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_INLINE_TEXT_UNARY: Cell = Cell {
    name: "image_inline_text_unary",
    program: IMAGE_TEXT,
    image: Some(INLINE_TEXT),
    families: C,
    ..CELL
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_INLINE_TEXT_STREAMED: Cell = Cell {
    name: "image_inline_text_streamed",
    program: Program {
        streamed: true,
        ..IMAGE_TEXT
    },
    events: true,
    ..IMAGE_INLINE_TEXT_UNARY
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_INLINE_MIXED_ORDER: Cell = Cell {
    name: "image_inline_mixed_order",
    program: Program {
        prompt: super::image::MIXED_AFTER,
        ..IMAGE_TEXT
    },
    image: Some(ImageCell {
        case: ImageCase::MixedOrder,
        source: ImageSource::Inline,
    }),
    ..IMAGE_INLINE_TEXT_UNARY
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_INLINE_TOOL_UNARY: Cell = Cell {
    name: "image_inline_tool_unary",
    program: IMAGE_TOOLS,
    image: Some(INLINE_TOOL),
    tools: &[ToolKind::Adder],
    families: CTC,
    resume_after: Some(1),
    live_resume: true,
    ..CELL
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_INLINE_TOOL_STREAMED: Cell = Cell {
    name: "image_inline_tool_streamed",
    program: Program {
        streamed: true,
        ..IMAGE_TOOLS
    },
    events: true,
    resume_after: None,
    live_resume: false,
    ..IMAGE_INLINE_TOOL_UNARY
};
/// The first run answers the image; the second run, a text prompt about
/// it, is fed the first run's committed history through the conversation
/// store both interpreters append to and load from (§11).
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_INLINE_FOLLOWUP: Cell = Cell {
    name: "image_inline_followup",
    program: Program {
        max_turns: Some(3),
        conversation: Some(CONVERSATION),
        second_prompt: Some(super::image::FOLLOWUP_PROMPT),
        ..IMAGE_TEXT
    },
    image: Some(ImageCell {
        case: ImageCase::Followup,
        source: ImageSource::Inline,
    }),
    memory: Memory::InMemory,
    families: TWO_RUNS,
    ..CELL
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_URL_TEXT_UNARY: Cell = Cell {
    name: "image_url_text_unary",
    image: Some(ImageCell {
        case: ImageCase::Text,
        source: ImageSource::Url,
    }),
    ..IMAGE_INLINE_TEXT_UNARY
};
#[allow(dead_code, reason = "the image matrix runs on four of the six wires")]
pub(crate) const IMAGE_URL_TOOL_UNARY: Cell = Cell {
    name: "image_url_tool_unary",
    image: Some(ImageCell {
        case: ImageCase::Tool,
        source: ImageSource::Url,
    }),
    ..IMAGE_INLINE_TOOL_UNARY
};

// -- §5 budgets and endings; §9.1 stopping (`ecs_endings`) --------------------

pub(crate) const ENDINGS_TOOL_DISPATCH_CANCELLED: Cell = Cell {
    name: "endings_tool_dispatch_cancelled",
    program: Program {
        hooks: &[Hook::CancelAddDispatch, Hook::RecordSettled],
        ending: Ending::Cancelled(CANCEL_ADD_DISPATCH),
        ..TOOLS
    },
    families: C,
    ..ADD
};
pub(crate) const ENDINGS_TOOL_OUTCOME_CANCELLED: Cell = Cell {
    name: "endings_tool_outcome_cancelled",
    program: Program {
        hooks: &[Hook::CancelAddOutcome, Hook::RecordSettled],
        ending: Ending::Cancelled(CANCEL_ADD_OUTCOME),
        ..TOOLS
    },
    families: CT,
    ..ADD
};
pub(crate) const ENDINGS_ANSWER_OUTCOME_CANCELLED: Cell = Cell {
    name: "endings_answer_outcome_cancelled",
    program: Program {
        hooks: &[Hook::CancelAnswer, Hook::RecordSettled],
        ending: Ending::Cancelled(CANCEL_ANSWER),
        ..BASIC
    },
    ..TEXT
};
pub(crate) const ENDINGS_TURN_FINISHED_STOP: Cell = Cell {
    name: "endings_turn_finished_stop",
    program: Program {
        hooks: &[Hook::StopAfterTurn, Hook::RecordSettled],
        ending: Ending::Cancelled(STOP_AFTER_TURN),
        ..TOOLS
    },
    families: C,
    ..ADD
};
pub(crate) const ENDINGS_ANSWER_TURN_STOP: Cell = Cell {
    name: "endings_answer_turn_stop",
    program: Program {
        hooks: &[Hook::StopAtAnswer, Hook::RecordSettled],
        ending: Ending::Cancelled(STOP_AT_ANSWER),
        ..TOOLS
    },
    resume_after: Some(1),
    ..ADD
};
#[allow(dead_code)] // the Gemini wire reuses the breadth recording
pub(crate) const ENDINGS_TEXT_DELTA_STOP: Cell = Cell {
    name: "endings_text_delta_stop",
    program: Program {
        prompt: ESSAY_PROMPT,
        hooks: &[Hook::StopOnTextDelta, Hook::RecordSettled],
        ending: Ending::Cancelled(STOP_ON_TEXT_DELTA),
        streamed: true,
        max_turns: Some(3),
        ..BASIC
    },
    events: true,
    ..TEXT
};
pub(crate) const ENDINGS_TOOL_CALL_DELTA_STOP: Cell = Cell {
    name: "endings_tool_call_delta_stop",
    program: Program {
        preamble: Some(NOTE_PREAMBLE),
        prompt: NOTE_PROMPT,
        hooks: &[Hook::StopOnToolCallDelta, Hook::RecordSettled],
        ending: Ending::Cancelled(STOP_ON_TOOL_CALL_DELTA),
        streamed: true,
        ..TOOLS
    },
    tools: &[ToolKind::WriteNote],
    events: true,
    families: C,
    ..CELL
};
pub(crate) const ENDINGS_TOOL_DISPATCH_CANCELLED_STREAMED: Cell = Cell {
    name: "endings_tool_dispatch_cancelled_streamed",
    program: Program {
        streamed: true,
        ..ENDINGS_TOOL_DISPATCH_CANCELLED.program
    },
    events: true,
    ..ENDINGS_TOOL_DISPATCH_CANCELLED
};
pub(crate) const ENDINGS_TURN_FINISHED_STOP_STREAMED: Cell = Cell {
    name: "endings_turn_finished_stop_streamed",
    program: Program {
        streamed: true,
        ..ENDINGS_TURN_FINISHED_STOP.program
    },
    events: true,
    ..ENDINGS_TURN_FINISHED_STOP
};
pub(crate) const ENDINGS_TOOL_OUTCOME_CANCELLED_STREAMED: Cell = Cell {
    name: "endings_tool_outcome_cancelled_streamed",
    program: Program {
        streamed: true,
        ..ENDINGS_TOOL_OUTCOME_CANCELLED.program
    },
    events: true,
    ..ENDINGS_TOOL_OUTCOME_CANCELLED
};

// -- §9 steering: every hook is a system (`ecs_hooks`) ------------------------

pub(crate) const HOOKS_OBSERVE_EVERYTHING: Cell = Cell {
    name: "hooks_observe_everything",
    program: Program {
        conversation: Some(CONVERSATION),
        hooks: &[Hook::ObserveEverything],
        ..TOOLS
    },
    memory: Memory::InMemory,
    families: MCTCM,
    resume_after: Some(1),
    ..ADD
};
pub(crate) const HOOKS_PATCH_TOOL_ARGS: Cell = Cell {
    name: "hooks_patch_tool_args",
    program: Program {
        hooks: &[Hook::PatchAddArgs],
        ..TOOLS
    },
    resume_after: Some(1),
    ..ADD
};
pub(crate) const HOOKS_PATCH_TOOL_ARGS_STREAMED: Cell = Cell {
    name: "hooks_patch_tool_args_streamed",
    program: Program {
        hooks: &[Hook::PatchAddArgs],
        streamed: true,
        ..TOOLS
    },
    events: true,
    resume_after: Some(1),
    ..ADD
};
pub(crate) const HOOKS_DENY_TOOL: Cell = Cell {
    name: "hooks_deny_tool",
    program: Program {
        hooks: &[Hook::DenyAdd],
        ..TOOLS
    },
    families: CC,
    ..ADD
};
pub(crate) const HOOKS_DENY_TOOL_STREAMED: Cell = Cell {
    name: "hooks_deny_tool_streamed",
    program: Program {
        hooks: &[Hook::DenyAdd],
        streamed: true,
        ..TOOLS
    },
    events: true,
    families: CC,
    ..ADD
};
pub(crate) const HOOKS_REPLACE_TOOL_RESULT: Cell = Cell {
    name: "hooks_replace_tool_result",
    program: Program {
        hooks: &[Hook::ReplaceAddResult],
        ..TOOLS
    },
    resume_after: Some(1),
    ..ADD
};
pub(crate) const HOOKS_REPLACE_ANSWER: Cell = Cell {
    name: "hooks_replace_answer",
    program: Program {
        hooks: &[Hook::ReplaceAnswer],
        expected_output: Some(REPLACED_ANSWER),
        ..BASIC
    },
    ..TEXT
};
pub(crate) const HOOKS_PREAMBLE_OVERRIDE: Cell = Cell {
    name: "hooks_preamble_override",
    program: Program {
        hooks: &[Hook::PreambleOverride],
        ..BASIC
    },
    ..TEXT
};
pub(crate) const HOOKS_DEMAND_DONE: Cell = Cell {
    name: "hooks_demand_done",
    program: Program {
        hooks: &[Hook::DemandDone],
        max_turns: Some(3),
        ..BASIC
    },
    families: CC,
    ..TEXT
};
pub(crate) const HOOKS_LOOKUP_BEFORE_RUN: Cell = Cell {
    name: "hooks_lookup_before_run",
    program: Program {
        hooks: &[Hook::LookupBeforeRun],
        ..TOOLS
    },
    families: &[Tool, Completion, Tool, Completion],
    resume_after: Some(1),
    ..ADD
};
pub(crate) const HOOKS_TWO_HOOKS: Cell = Cell {
    name: "hooks_two_hooks",
    program: Program {
        hooks: &[Hook::PatchAddArgs, Hook::ReplaceAddResult],
        ..TOOLS
    },
    resume_after: Some(1),
    ..ADD
};

// -- §9.5 a hook's own dispatch; host effects (`ecs_host`) --------------------

const HOST: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: READY_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const HOST_WITH_TOOL: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    ..HOST
};
const NOTES: Cell = Cell {
    program: HOST,
    bus: Bus::Host,
    notes: true,
    ..CELL
};

pub(crate) const HOST_CUSTOM_AT_START: Cell = Cell {
    name: "host_custom_at_start",
    program: Program {
        hooks: &[Hook::NoteAtStart],
        ..HOST
    },
    families: &[Custom, Completion],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_AT_COMPLETION_CALL: Cell = Cell {
    name: "host_custom_at_completion_call",
    program: Program {
        hooks: &[Hook::NoteAtCompletionCall],
        ..HOST
    },
    families: &[Custom, Completion],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_AT_OUTCOME: Cell = Cell {
    name: "host_custom_at_outcome",
    program: Program {
        hooks: &[Hook::NoteAtOutcome],
        ..HOST_WITH_TOOL
    },
    tools: &[ToolKind::Adder],
    families: &[Completion, Tool, Custom, Completion],
    resume_after: Some(1),
    ..NOTES
};
pub(crate) const HOST_CUSTOM_AT_SETTLED: Cell = Cell {
    name: "host_custom_at_settled",
    program: Program {
        hooks: &[Hook::NoteAtSettled],
        ..HOST
    },
    families: &[Completion, Custom],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_START_AND_SETTLED: Cell = Cell {
    name: "host_custom_start_and_settled",
    program: Program {
        hooks: &[Hook::NoteAtStart, Hook::NoteAtSettled],
        ..HOST
    },
    families: &[Custom, Completion, Custom],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_TWICE_SERIAL: Cell = Cell {
    name: "host_custom_twice_serial",
    program: Program {
        hooks: &[Hook::NoteTwice],
        ..HOST
    },
    bus: Bus::HostSerial,
    families: &[Custom, Custom, Completion],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_TWICE_CONCURRENT: Cell = Cell {
    name: "host_custom_twice_concurrent",
    program: Program {
        hooks: &[Hook::NoteTwice],
        ..HOST
    },
    families: &[Custom, Custom, Completion],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_AT_START_STREAMED: Cell = Cell {
    name: "host_custom_at_start_streamed",
    program: Program {
        hooks: &[Hook::NoteAtStart],
        streamed: true,
        ..HOST
    },
    events: true,
    families: &[Custom, Completion],
    ..NOTES
};
pub(crate) const HOST_CUSTOM_AT_OUTCOME_STREAMED: Cell = Cell {
    name: "host_custom_at_outcome_streamed",
    program: Program {
        hooks: &[Hook::NoteAtOutcome],
        streamed: true,
        ..HOST_WITH_TOOL
    },
    tools: &[ToolKind::Adder],
    events: true,
    families: &[Completion, Tool, Custom, Completion],
    resume_after: Some(1),
    ..NOTES
};
pub(crate) const HOST_CUSTOM_UNSERVED: Cell = Cell {
    name: "host_custom_unserved",
    program: Program {
        hooks: &[Hook::NoteUnserved],
        ..HOST
    },
    notes: false,
    families: C,
    ..NOTES
};

// -- §7 keys; serving (`ecs_serving`) -----------------------------------------

const TWO_TOOLS: Program = Program {
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    max_turns: Some(8),
    streamed: true,
    ..Program::DEFAULT
};
const SIGNALS: Cell = Cell {
    program: TWO_TOOLS,
    tools: &[ToolKind::Alpha, ToolKind::Beta],
    families: CTTC,
    ..CELL
};

pub(crate) const SERVING_SERIAL_CONCURRENCY_ONE: Cell = Cell {
    name: "serving_serial_concurrency_one",
    program: Program {
        tool_concurrency: Some(1),
        ..TWO_TOOLS
    },
    bus: Bus::Serial,
    ..SIGNALS
};
pub(crate) const SERVING_CONCURRENT_CONCURRENCY_ONE: Cell = Cell {
    name: "serving_concurrent_concurrency_one",
    program: Program {
        tool_concurrency: Some(1),
        ..TWO_TOOLS
    },
    ..SIGNALS
};
pub(crate) const SERVING_CONCURRENT_CONCURRENCY_TWO: Cell = Cell {
    name: "serving_concurrent_concurrency_two",
    program: Program {
        tool_concurrency: Some(2),
        ..TWO_TOOLS
    },
    ..SIGNALS
};
pub(crate) const SERVING_CONCURRENT_CONCURRENCY_TWO_EVENTS: Cell = Cell {
    name: "serving_concurrent_concurrency_two_events",
    program: Program {
        tool_concurrency: Some(2),
        ..TWO_TOOLS
    },
    events: true,
    ..SIGNALS
};
pub(crate) const SERVING_CAPACITY_ONE: Cell = Cell {
    name: "serving_capacity_one",
    program: Program {
        tool_concurrency: Some(2),
        ..TWO_TOOLS
    },
    bus: Bus::CapacityOne,
    ..SIGNALS
};
pub(crate) const SERVING_SERIAL_MEMORY_TOOLS: Cell = Cell {
    name: "serving_serial_memory_tools",
    program: Program {
        conversation: Some(CONVERSATION),
        ..TOOLS
    },
    memory: Memory::InMemory,
    bus: Bus::Serial,
    families: MCTCM,
    resume_after: Some(1),
    ..ADD
};
pub(crate) const SERVING_MODEL_ROUTE: Cell = Cell {
    name: "serving_model_route",
    program: Program {
        route: Some(ROUTE),
        hooks: &[Hook::RouteAfterFirstTurn],
        ..TOOLS
    },
    resume_after: Some(1),
    ..ADD
};
pub(crate) const SERVING_MODEL_ROUTE_UNSELECTED: Cell = Cell {
    name: "serving_model_route_unselected",
    resume_after: Some(1),
    program: Program {
        route: Some(ROUTE),
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SERVING_HOST_BUS: Cell = Cell {
    name: "serving_host_bus",
    bus: Bus::Host,
    ..ADD
};
pub(crate) const SERVING_HOST_BUS_STREAMED: Cell = Cell {
    name: "serving_host_bus_streamed",
    program: Program {
        streamed: true,
        ..TOOLS
    },
    bus: Bus::Host,
    events: true,
    ..ADD
};

// -- §9.6 layers (`ecs_layers`) -----------------------------------------------

const fn at_tool(layer: LayerKind) -> LayerSpec {
    LayerSpec {
        at: LayerAt::Tool,
        layer,
    }
}

pub(crate) const LAYERS_DENY_TOOL: Cell = Cell {
    name: "layers_deny_tool",
    program: Program {
        layers: &[at_tool(LayerKind::DenyAdd)],
        ..TOOLS
    },
    families: CC,
    ..ADD
};
pub(crate) const LAYERS_PATCH_TOOL_ARGS: Cell = Cell {
    name: "layers_patch_tool_args",
    resume_after: Some(1),
    program: Program {
        layers: &[at_tool(LayerKind::PatchAddArgs)],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const LAYERS_REPLACE_TOOL_RESULT: Cell = Cell {
    name: "layers_replace_tool_result",
    resume_after: Some(1),
    program: Program {
        layers: &[at_tool(LayerKind::ReplaceAddResult)],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const LAYERS_TWO_LAYERS: Cell = Cell {
    name: "layers_two_layers",
    resume_after: Some(1),
    program: Program {
        layers: &[
            at_tool(LayerKind::PatchAddArgs),
            at_tool(LayerKind::ReplaceAddResult),
        ],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const LAYERS_HOST_DENY_OVER_HOST_BUS: Cell = Cell {
    name: "layers_host_deny_over_host_bus",
    program: Program {
        layers: &[at_tool(LayerKind::DenyAdd)],
        ..TOOLS
    },
    bus: Bus::Host,
    families: CC,
    ..ADD
};
pub(crate) const LAYERS_PATCH_BENEATH_HOOK_PATCH: Cell = Cell {
    name: "layers_patch_beneath_hook_patch",
    resume_after: Some(1),
    program: Program {
        hooks: &[Hook::PatchAddArgs],
        layers: &[at_tool(LayerKind::PatchAgain)],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const LAYERS_MEMORY_LOAD_REPLACED: Cell = Cell {
    name: "layers_memory_load_replaced",
    program: Program {
        preamble: Some(BASIC_PREAMBLE),
        prompt: NAME_PROMPT,
        temperature: Some(0.0),
        conversation: Some(CONVERSATION),
        hooks: &[Hook::HistoryIsReplaced],
        layers: &[LayerSpec {
            at: LayerAt::Memory,
            layer: LayerKind::ReplaceLoad,
        }],
        ..Program::DEFAULT
    },
    memory: Memory::InMemory,
    families: &[Mem, Completion, Mem],
    ..CELL
};

// -- §11 memory is the graph (`ecs_memory`) -----------------------------------

const MEMORY: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: READY_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    conversation: Some(CONVERSATION),
    ..Program::DEFAULT
};
const REMEMBERS: Cell = Cell {
    program: MEMORY,
    memory: Memory::InMemory,
    families: &[Mem, Completion, Mem],
    ..CELL
};
const TWO_RUNS: &[EffectFamily] = &[Mem, Completion, Mem, Mem, Completion, Mem];

pub(crate) const MEMORY_CLEAR_AT_START: Cell = Cell {
    name: "memory_clear_at_start",
    program: Program {
        hooks: &[Hook::ClearAtStart],
        ..MEMORY
    },
    families: &[Mem, Mem, Completion, Mem],
    ..REMEMBERS
};
pub(crate) const MEMORY_CLEAR_AT_SETTLED: Cell = Cell {
    name: "memory_clear_at_settled",
    program: Program {
        hooks: &[Hook::ClearAtSettled],
        ..MEMORY
    },
    families: &[Mem, Completion, Mem, Mem],
    ..REMEMBERS
};
pub(crate) const MEMORY_TWO_RUNS: Cell = Cell {
    name: "memory_two_runs",
    program: Program {
        second_prompt: Some(SECOND_PROMPT),
        ..MEMORY
    },
    families: TWO_RUNS,
    ..REMEMBERS
};
pub(crate) const MEMORY_TWO_RUNS_STREAMED: Cell = Cell {
    name: "memory_two_runs_streamed",
    program: Program {
        second_prompt: Some(SECOND_PROMPT),
        max_turns: Some(8),
        streamed: true,
        ..MEMORY
    },
    events: true,
    families: TWO_RUNS,
    ..REMEMBERS
};
pub(crate) const MEMORY_CLEAR_AT_SETTLED_TWO_RUNS: Cell = Cell {
    name: "memory_clear_at_settled_two_runs",
    program: Program {
        hooks: &[Hook::ClearAtSettled],
        second_prompt: Some(SECOND_PROMPT),
        ..MEMORY
    },
    families: &[Mem, Completion, Mem, Mem, Mem, Completion, Mem, Mem],
    ..REMEMBERS
};
pub(crate) const MEMORY_CLEAR_AT_START_TWO_RUNS: Cell = Cell {
    name: "memory_clear_at_start_two_runs",
    program: Program {
        hooks: &[Hook::ClearAtStart],
        second_prompt: Some(SECOND_PROMPT),
        ..MEMORY
    },
    families: &[Mem, Mem, Completion, Mem, Mem, Mem, Completion, Mem],
    ..REMEMBERS
};
pub(crate) const MEMORY_HISTORY_BYPASS: Cell = Cell {
    name: "memory_history_bypass",
    program: Program {
        prompt: NAME_PROMPT,
        history: Some(bypass_history),
        max_turns: None,
        ..MEMORY
    },
    families: C,
    ..REMEMBERS
};
pub(crate) const MEMORY_HOST_BUS_MEMORY: Cell = Cell {
    name: "memory_host_bus_memory",
    program: Program {
        max_turns: None,
        ..MEMORY
    },
    bus: Bus::Host,
    ..REMEMBERS
};
pub(crate) const MEMORY_SERIAL_TWO_TOOLS: Cell = Cell {
    name: "memory_serial_two_tools",
    program: Program {
        preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
        prompt: TWO_TOOL_STREAM_PROMPT,
        max_turns: Some(8),
        streamed: true,
        ..MEMORY
    },
    tools: &[ToolKind::Alpha, ToolKind::Beta],
    bus: Bus::Serial,
    events: true,
    families: &[Mem, Completion, Tool, Tool, Completion, Mem],
    resume_after: Some(1),
    ..REMEMBERS
};
pub(crate) const MEMORY_FAILING_APPEND: Cell = Cell {
    name: "memory_failing_append",
    memory: Memory::FailingAppend,
    ..REMEMBERS
};
pub(crate) const MEMORY_FAILING_APPEND_STREAMED: Cell = Cell {
    name: "memory_failing_append_streamed",
    program: Program {
        max_turns: Some(8),
        streamed: true,
        ..MEMORY
    },
    memory: Memory::FailingAppend,
    events: true,
    ..REMEMBERS
};

// -- §3 output modes (`ecs_output`) -------------------------------------------

const SCHEMA: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: STRUCTURED_OUTPUT_PROMPT,
    temperature: Some(0.0),
    output_schema: Some(event_schema),
    ..Program::DEFAULT
};
const WITH_TOOL: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: SUM_EVENT_PROMPT,
    max_turns: Some(3),
    ..SCHEMA
};

pub(crate) const OUTPUT_TOOL_UNARY: Cell = Cell {
    name: "output_tool_unary",
    program: Program {
        output_mode: Some(Output::Tool),
        ..SCHEMA
    },
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_TOOL_STREAMED: Cell = Cell {
    name: "output_tool_streamed",
    program: Program {
        output_mode: Some(Output::Tool),
        streamed: true,
        ..SCHEMA
    },
    events: true,
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_PROMPTED_UNARY: Cell = Cell {
    name: "output_prompted_unary",
    program: Program {
        output_mode: Some(Output::Prompted),
        ..SCHEMA
    },
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_PROMPTED_STREAMED: Cell = Cell {
    name: "output_prompted_streamed",
    program: Program {
        output_mode: Some(Output::Prompted),
        streamed: true,
        ..SCHEMA
    },
    events: true,
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_TOOL_WITH_REAL_TOOL: Cell = Cell {
    name: "output_tool_with_real_tool",
    program: Program {
        output_mode: Some(Output::Tool),
        ..WITH_TOOL
    },
    tools: &[ToolKind::Adder],
    families: CTC,
    resume_after: Some(1),
    ..CELL
};
pub(crate) const OUTPUT_PROMPTED_WITH_REAL_TOOL: Cell = Cell {
    name: "output_prompted_with_real_tool",
    program: Program {
        output_mode: Some(Output::Prompted),
        ..WITH_TOOL
    },
    tools: &[ToolKind::Adder],
    families: CTC,
    resume_after: Some(1),
    ..CELL
};
pub(crate) const OUTPUT_TOOL_CHOICE_SPECIFIC_OUTPUT: Cell = Cell {
    name: "output_tool_choice_specific_output",
    program: Program {
        output_mode: Some(Output::Tool),
        tool_choice: Some(Choice::Specific("final_result")),
        ..SCHEMA
    },
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_TOOL_CHOICE_REQUIRED: Cell = Cell {
    name: "output_tool_choice_required",
    program: Program {
        output_mode: Some(Output::Tool),
        tool_choice: Some(Choice::Required),
        max_turns: Some(2),
        ..SCHEMA
    },
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_TOOL_UNDER_NONE_DEGRADES: Cell = Cell {
    name: "output_tool_under_none_degrades",
    program: Program {
        output_mode: Some(Output::Tool),
        tool_choice: Some(Choice::None),
        ..SCHEMA
    },
    families: C,
    ..CELL
};
pub(crate) const OUTPUT_TOOL_THINKING: Cell = Cell {
    name: "output_tool_thinking",
    thinking: Thinking::On,
    reasoning: Some(ReasoningCase::Output),
    program: Program {
        output_mode: Some(Output::Tool),
        temperature: None,
        max_tokens: Some(4096),
        ..SCHEMA
    },
    families: C,
    ..CELL
};

// -- §2 verbatim strings; §9.3 the completion call (`ecs_shaping`) ------------

const SHAPING_BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: CONTEXT_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

pub(crate) const SHAPING_TOOL_CHOICE_REQUIRED_FIRST: Cell = Cell {
    name: "shaping_tool_choice_required_first",
    resume_after: Some(1),
    program: Program {
        hooks: &[Hook::PatchToolChoiceRequiredFirst],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT: Cell = Cell {
    name: "shaping_tool_choice_none_on_committed_output",
    program: Program {
        prompt: SUM_EVENT_PROMPT,
        output_schema: Some(event_schema),
        output_mode: Some(Output::Tool),
        hooks: &[Hook::PatchToolChoiceNoneSecond],
        ..TOOLS
    },
    // What the run does after the patched turn is the wire's: a text answer
    // that already satisfies the schema is accepted, one that does not is
    // reprompted — the record pins it, the families are not asserted.
    families: &[],
    ..ADD
};
pub(crate) const SHAPING_EXTRA_CONTEXT: Cell = Cell {
    name: "shaping_extra_context",
    program: Program {
        hooks: &[Hook::PatchExtraContext],
        ..SHAPING_BASIC
    },
    families: C,
    ..CELL
};
pub(crate) const SHAPING_EXTRA_CONTEXT_STREAMED: Cell = Cell {
    name: "shaping_extra_context_streamed",
    program: Program {
        hooks: &[Hook::PatchExtraContext],
        streamed: true,
        max_turns: None,
        ..SHAPING_BASIC
    },
    events: true,
    families: C,
    ..CELL
};
pub(crate) const SHAPING_MERGED_THREE: Cell = Cell {
    name: "shaping_merged_three",
    resume_after: Some(1),
    program: Program {
        hooks: &[
            Hook::PreambleOverride,
            Hook::PatchExtraContext,
            Hook::PatchToolChoiceRequiredFirst,
        ],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_ROUTE_ON_FIRST_TURN: Cell = Cell {
    name: "shaping_route_on_first_turn",
    resume_after: Some(1),
    program: Program {
        route: Some(ROUTE),
        hooks: &[Hook::RouteOnFirstTurn],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_LATE_ROUTE: Cell = Cell {
    name: "shaping_late_route",
    resume_after: Some(1),
    program: Program {
        late_route: Some(LATE_ROUTE),
        hooks: &[Hook::SelectLate],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_MAX_TOKENS_SECOND_TURN: Cell = Cell {
    name: "shaping_max_tokens_second_turn",
    resume_after: Some(1),
    program: Program {
        hooks: &[Hook::PatchMaxTokensSecond],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_THINKING_SECOND_TURN: Cell = Cell {
    name: "shaping_thinking_second_turn",
    thinking: Thinking::SecondTurnOnly,
    reasoning: Some(ReasoningCase::Tool),
    resume_after: Some(1),
    program: Program {
        hooks: &[Hook::PatchThinkingSecond],
        max_tokens: Some(4096),
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_PREAMBLE_SECOND_TURN: Cell = Cell {
    name: "shaping_preamble_second_turn",
    resume_after: Some(1),
    program: Program {
        hooks: &[Hook::PatchPreambleSecond],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_ACTIVE_TOOLS_NONE_SECOND_TURN: Cell = Cell {
    name: "shaping_active_tools_none_second_turn",
    resume_after: Some(1),
    program: Program {
        hooks: &[Hook::PatchActiveToolsNoneSecond],
        ..TOOLS
    },
    ..ADD
};
pub(crate) const SHAPING_HISTORY_FIRST_TURN: Cell = Cell {
    name: "shaping_history_first_turn",
    program: Program {
        prompt: NAME_PROMPT,
        hooks: &[Hook::PatchHistoryFirst],
        ..SHAPING_BASIC
    },
    families: C,
    ..CELL
};

// -- §8.3 nested dispatch; causal ids (`ecs_causal`) --------------------------

const LOOKUP: Program = Program {
    preamble: Some(LOOKUP_PREAMBLE),
    prompt: LOOKUP_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    nesting: Some(Nesting {
        child: super::corpus::NestedChild::Completion,
        ..NESTING
    }),
    ..Program::DEFAULT
};
const NESTED: Cell = Cell {
    program: LOOKUP,
    tools: &[ToolKind::Lookup],
    bus: Bus::Host,
    families: &[Completion, Tool, Completion, Completion],
    ..CELL
};

pub(crate) const CAUSAL_COMPLETION_SERIAL: Cell = Cell {
    name: "causal_completion_serial",
    resume_after: Some(1),
    program: Program {
        host_serial: true,
        ..LOOKUP
    },
    bus: Bus::HostSerial,
    ..NESTED
};
pub(crate) const CAUSAL_COMPLETION_CONCURRENT: Cell = Cell {
    name: "causal_completion_concurrent",
    resume_after: Some(1),
    ..NESTED
};
pub(crate) const CAUSAL_COMPLETION_STREAMED: Cell = Cell {
    name: "causal_completion_streamed",
    resume_after: Some(1),
    program: Program {
        streamed: true,
        ..LOOKUP
    },
    events: true,
    ..NESTED
};

// -- §13 resume is a scene load (`corpus_resume.rs`) --------------------------

/// The plain two-turn tool run every other row reads: saved after its tool
/// turn's results and resumed over the log's tail.
pub(crate) const RESUME_TOOL_TURN: Cell = Cell {
    name: "resume_tool_turn",
    resume_after: Some(1),
    ..ADD
};

// -- Matrix N's shapes, for the cells that reuse a breadth recording ----------

/// The breadth matrix's text-delta stop (`corpus_breadth/text_delta_stop`):
/// a shorter essay, no settled observer, the run's default budget (the
/// OpenAI Responses and Gemini wires reuse that recording).
#[allow(dead_code)]
pub(crate) const BREADTH_TEXT_DELTA_STOP: Cell = Cell {
    name: "endings_text_delta_stop",
    program: Program {
        prompt: "Write four paragraphs about the history of the Rust programming language.",
        hooks: &[Hook::StopOnTextDelta],
        ending: Ending::Cancelled(STOP_ON_TEXT_DELTA),
        streamed: true,
        ..BASIC
    },
    events: true,
    ..TEXT
};
/// The breadth matrix's cancelled dispatch (`corpus_breadth/tool_dispatch_cancelled`):
/// no settled observer (the same two wires reuse it).
#[allow(dead_code)]
pub(crate) const BREADTH_TOOL_DISPATCH_CANCELLED: Cell = Cell {
    name: "endings_tool_dispatch_cancelled",
    program: Program {
        hooks: &[Hook::CancelAddDispatch],
        ending: Ending::Cancelled(CANCEL_ADD_DISPATCH),
        ..TOOLS
    },
    families: C,
    ..ADD
};
