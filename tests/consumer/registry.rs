//! Executable case registry; fixture names and plans derive from these entries.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Approval {
    Approve,
    Deny,
    Cancel,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Arrival {
    ReadFirst,
    SearchFirst,
    Together,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Fault {
    None,
    WriteError,
    StreamErrorBeforeFinal,
    StreamErrorAfterFinal,
    RemoveModelBeforeDispatch,
    RemoveModelBetweenTurns,
    CancelBeforeServe,
    CancelPartial,
    LostWriteOutcome,
    CancelBackground,
    FoldedReplay,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Provider {
    Synthetic,
    Anthropic,
    Openai,
    Gemini,
}

impl Provider {
    pub(crate) fn cassette_provider(self) -> Option<&'static str> {
        match self {
            Self::Synthetic => None,
            Self::Anthropic => Some("anthropic"),
            Self::Openai => Some("openai"),
            Self::Gemini => Some("gemini"),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Case {
    pub id: &'static str,
    pub matrix: &'static str,
    pub provider: Provider,
    pub stream: bool,
    pub approval: Approval,
    pub concurrency: usize,
    pub intake: usize,
    pub arrival: Arrival,
    pub stream_batch: usize,
    pub serial_keys: bool,
    pub spawn_noise: usize,
    pub custom_answers: bool,
    pub fault: Fault,
    pub interleaved: bool,
    pub capture_zero: bool,
    pub identity_checks: bool,
}

pub(crate) fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (id, provider, stream) in [
        ("synthetic-approve", Provider::Synthetic, false),
        ("anthropic-unary", Provider::Anthropic, false),
        ("anthropic-stream", Provider::Anthropic, true),
        ("openai-unary", Provider::Openai, false),
        ("openai-stream", Provider::Openai, true),
        ("gemini-unary", Provider::Gemini, false),
        ("gemini-stream", Provider::Gemini, true),
    ] {
        cases.push(Case {
            id,
            matrix: if provider == Provider::Synthetic {
                "consumer"
            } else {
                "provider"
            },
            provider,
            stream,
            approval: Approval::Approve,
            concurrency: 2,
            intake: 16,
            arrival: Arrival::Together,
            stream_batch: 4096,
            serial_keys: false,
            spawn_noise: 0,
            custom_answers: false,
            fault: Fault::None,
            interleaved: false,
            capture_zero: false,
            identity_checks: false,
        });
    }
    for (
        id,
        matrix,
        approval,
        concurrency,
        intake,
        arrival,
        stream,
        stream_batch,
        serial_keys,
        spawn_noise,
    ) in [
        (
            "serial-tools",
            "concurrency",
            Approval::Approve,
            1,
            16,
            Arrival::ReadFirst,
            false,
            4096,
            false,
            0,
        ),
        (
            "concurrent-read-first",
            "concurrency",
            Approval::Approve,
            2,
            16,
            Arrival::ReadFirst,
            false,
            4096,
            false,
            0,
        ),
        (
            "concurrent-search-first",
            "concurrency",
            Approval::Approve,
            2,
            16,
            Arrival::SearchFirst,
            false,
            4096,
            false,
            7,
        ),
        (
            "bounded-intake",
            "concurrency",
            Approval::Approve,
            2,
            1,
            Arrival::ReadFirst,
            false,
            4096,
            false,
            3,
        ),
        (
            "serial-handler-keys",
            "concurrency",
            Approval::Approve,
            2,
            16,
            Arrival::Together,
            false,
            4096,
            true,
            4,
        ),
        (
            "stream-single-events",
            "stream",
            Approval::Approve,
            2,
            16,
            Arrival::Together,
            true,
            1,
            false,
            0,
        ),
        (
            "stream-groups",
            "stream",
            Approval::Approve,
            2,
            16,
            Arrival::Together,
            true,
            3,
            false,
            5,
        ),
        (
            "approval-denied",
            "approval",
            Approval::Deny,
            2,
            16,
            Arrival::Together,
            false,
            4096,
            false,
            0,
        ),
        (
            "cancel-at-approval",
            "approval",
            Approval::Cancel,
            2,
            16,
            Arrival::Together,
            false,
            4096,
            false,
            0,
        ),
    ] {
        cases.push(Case {
            id,
            matrix,
            provider: Provider::Synthetic,
            stream,
            approval,
            concurrency,
            intake,
            arrival,
            stream_batch,
            serial_keys,
            spawn_noise,
            custom_answers: serial_keys,
            fault: Fault::None,
            interleaved: false,
            capture_zero: false,
            identity_checks: false,
        });
    }
    let Some(baseline) = cases.first().cloned() else {
        return cases;
    };
    cases.push(Case {
        id: "cancel-before-serve",
        matrix: "lifecycle",
        fault: Fault::CancelBeforeServe,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "cancel-at-partial-stream",
        matrix: "stream",
        fault: Fault::CancelPartial,
        stream: true,
        stream_batch: 1,
        ..baseline.clone()
    });
    let mut custom = baseline.clone();
    cases.push(Case {
        id: "replay-guarantees",
        matrix: "stream",
        fault: Fault::FoldedReplay,
        stream: true,
        stream_batch: 3,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "cancelled-background-loser",
        matrix: "lifecycle",
        fault: Fault::CancelBackground,
        stream: true,
        stream_batch: 1,
        interleaved: true,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "external-write-outcome-lost",
        matrix: "lifecycle",
        fault: Fault::LostWriteOutcome,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "identity-dependencies",
        matrix: "identity",
        custom_answers: true,
        identity_checks: true,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "zero-progress-stream",
        matrix: "persistence",
        stream: true,
        stream_batch: 3,
        capture_zero: true,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "zero-progress-unary",
        matrix: "persistence",
        capture_zero: true,
        ..baseline.clone()
    });
    cases.push(Case {
        id: "interleaved-streams",
        matrix: "stream",
        stream: true,
        stream_batch: 1,
        interleaved: true,
        ..baseline.clone()
    });
    custom.id = "custom-answers";
    custom.matrix = "custom_answers";
    custom.custom_answers = true;
    cases.push(custom);
    cases.push(Case {
        id: "concurrent-handler-keys",
        matrix: "concurrency",
        custom_answers: true,
        ..baseline.clone()
    });
    for (id, matrix, fault, stream) in [
        ("tool-write-failure", "approval", Fault::WriteError, false),
        (
            "stream-error-before-final",
            "stream",
            Fault::StreamErrorBeforeFinal,
            true,
        ),
        (
            "stream-error-after-final",
            "stream",
            Fault::StreamErrorAfterFinal,
            true,
        ),
        (
            "model-removed-before-dispatch",
            "lifecycle",
            Fault::RemoveModelBeforeDispatch,
            false,
        ),
        (
            "model-removed-between-turns",
            "lifecycle",
            Fault::RemoveModelBetweenTurns,
            false,
        ),
    ] {
        cases.push(Case {
            id,
            matrix,
            fault,
            stream,
            stream_batch: 1,
            ..baseline.clone()
        });
    }
    cases
}
