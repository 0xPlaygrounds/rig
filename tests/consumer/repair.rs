//! Rust project repair tools and their isolated validation boundary.

mod process;
mod project;
mod state;
mod validation;

mod scripted;
#[cfg(test)]
mod tests;

pub(super) use scripted::choice as scripted_choice;
pub(super) use state::{Patch, Receipt, Snapshot, State};
pub(super) use validation::Phase;

use super::{Approval, Error};
use rig_core::{effect::FamilyDescriptor, tool::ContextValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(super) const TOOLS: &[&str] = &[
    "repo_list_files",
    "repo_read_file",
    "repo_search",
    "repo_propose_patch",
    "repo_apply_patch",
    "repo_validate",
];
pub(super) const PREAMBLE: &str = "You maintain a small Rust project using the supplied tools. Read README.md and the relevant source/tests, reproduce the failure with initial validation, and explain its cause briefly. Follow the documented public contract. Add a focused regression in tests/regression.rs that asserts the intended correct behavior and fails against unchanged source; a test that expects the current bug or its panic is not a regression. Observe that behavioral failure before proposing the src/lib.rs repair. Existing tests and package files are immutable. After a successful proposal, call repo_apply_patch with its operation to request host approval and apply only if approved. The tool invokes the separate host decision; asking for approval in final prose does not invoke it. Tool results are the only approval source. A denied or cancelled production patch ends the task. Run validation in a later turn after receiving the preceding edit result; batching an edit with its validation returns a recoverable wait error. Run final validation after an approved repair and report its actual results. The host accepts one regression edit and one production edit per run. Keep both edits focused and rustfmt-formatted. Do not delete, ignore or weaken tests, hardcode example answers, or add harness-dependent behavior. Use only the declared tools. Do not claim successful repair unless final validation accepted it.";
pub(super) const PROMPT: &str = "Repair the failing tests in this Rust project while preserving its documented behavior. Inspect the repository and reproduce the failure, choose and prove a regression, then propose your repair through the approval-controlled tools.";

pub(super) fn initial_source() -> String {
    include_str!("../fixtures/repository_repair/src/lib.rs").into()
}

pub(super) fn acceptance(
    state: &State,
    case: &super::Case,
    observations: &[super::Observation],
    has_cut: bool,
) -> Result<Value, Error> {
    use super::Fault;
    let files = state.project.image()?;
    if !state.initial.as_ref().is_some_and(|report| report.accepted)
        || !state
            .regression
            .as_ref()
            .is_some_and(|report| report.accepted)
    {
        return Err(Error::Invariant(
            "repair did not observe initial and regression behavioral failures".into(),
        ));
    }
    match case.fault {
        Fault::RepairInsufficient => {
            let rejected=state.final_report.as_ref().is_some_and(|report|!report.accepted && report.steps.iter().any(|step|step.command.first().is_some_and(|name|name=="<test:contract_oracle>") && step.output.code==Some(101)));
            if !rejected || state.project.writes()!=2 {return Err(Error::Invariant("insufficient repair was not rejected by independent behavior checks".into()));}
        }
        Fault::RepairTimeout => {
            let rejected=state.final_report.as_ref().is_some_and(|report|!report.accepted && report.steps.iter().any(|step|step.output.timed_out));
            if !rejected || state.project.writes()!=2 {return Err(Error::Invariant("timeout did not prevent repository repair success".into()));}
        }
        Fault::RepairStaleApproval => {
            let changed=observations.iter().any(|item|item.boundary=="repair.changed-proposal");
            let refused=observations.iter().any(|item|item.data.pointer("/outcome/Err/message").and_then(Value::as_str).is_some_and(|message|message.contains("apply does not match approved diff and source")));
            if !changed || !refused || state.project.writes()!=1 || files.get("src/lib.rs")!=Some(&initial_source()) || state.validated() {return Err(Error::Invariant("changed approved diff did not refuse the production write".into()));}
        }
        _=>match case.approval {
            Approval::Approve if !state.validated() || state.project.writes()!=2 =>return Err(Error::Invariant("approved repository repair lacks successful independent validation and exactly two edits".into())),
            Approval::Deny|Approval::Cancel if state.project.writes()!=1 || files.get("src/lib.rs")!=Some(&initial_source()) || state.validated()=>return Err(Error::Invariant("unapproved production repair changed or validated source".into())),
            _=>(),
        }
    }
    if state.project.writes() == 2 && !has_cut {
        return Err(Error::Invariant(
            "repair did not capture its completed-production-edit cut".into(),
        ));
    }
    Ok(
        json!({"repaired":state.validated(),"production_decision":case.approval,"fault":case.fault,"writes":state.project.writes()}),
    )
}

pub(super) fn tool_family(name: &str) -> FamilyDescriptor {
    let (description, properties, required) = match name {
        "repo_list_files" => (
            "List the declared Rust project files and current project digest",
            json!({}),
            vec![],
        ),
        "repo_read_file" => (
            "Read a declared project file",
            json!({"path":{"type":"string"}}),
            vec!["path"],
        ),
        "repo_search" => (
            "Search the declared project files for literal text",
            json!({"needle":{"type":"string"}}),
            vec!["needle"],
        ),
        "repo_propose_patch" => (
            "Propose full replacement contents for tests/regression.rs (new desired-behavior regression) or src/lib.rs (repair after failed regression). All other paths are immutable. Runs rustfmt on scratch text, then returns the formatted exact diff and operation identity without writing the project",
            json!({"path":{"type":"string","enum":["tests/regression.rs","src/lib.rs"]},"content":{"type":"string"},"justification":{"type":"string"}}),
            vec!["path", "content", "justification"],
        ),
        "repo_apply_patch" => (
            "Request the host approval decision for a successful proposal operation, then apply that exact diff only if approved. Call this tool after repo_propose_patch returns its operation; inspect applied and decision in the result",
            json!({"operation":{"type":"string"}}),
            vec!["operation"],
        ),
        _ => (
            "Run named offline validation: initial reproduces existing failure, regression proves the new test fails before the repair, final checks tests, formatting and independent behavior",
            json!({"phase":{"type":"string","enum":["initial","regression","final"]}}),
            vec!["phase"],
        ),
    };
    FamilyDescriptor::Tool {
        name: name.into(),
        description: description.into(),
        parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
        embedding: None,
    }
}

fn string<'a>(args: &'a Value, name: &str) -> Result<&'a str, Error> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Invariant(format!("repair tool requires string {name}")))
}

pub(super) fn tool_answer(
    state: &mut State,
    name: &str,
    args: &Value,
    deadline: std::time::Instant,
) -> Result<Value, Error> {
    match name {
        "repo_list_files" => {
            let files = state.project.image()?;
            Ok(
                json!({"files":files.keys().collect::<Vec<_>>(),"project_digest":project::digest(&files)?}),
            )
        }
        "repo_read_file" => {
            let path = string(args, "path")?;
            Ok(
                json!({"path":path,"content":state.project.read(path)?,"project_digest":project::digest(&state.project.image()?)?}),
            )
        }
        "repo_search" => {
            let needle = string(args, "needle")?;
            if needle.is_empty() || needle.len() > 256 {
                return Err(Error::Invariant("search needs 1..256 bytes".into()));
            }
            let mut matches = Vec::new();
            for (path, contents) in state.project.image()? {
                for (line, text) in contents
                    .lines()
                    .enumerate()
                    .filter(|(_, text)| text.contains(needle))
                {
                    matches.push(json!({"path":path,"line":line+1,"text":text}));
                }
            }
            Ok(json!({"matches":matches}))
        }
        "repo_propose_patch" => {
            let proposed = state.propose(
                string(args, "path")?,
                string(args, "content")?,
                string(args, "justification")?,
            )?;
            let formatting = validation::format_patch(&state.project, &proposed.content, deadline)?;
            let proposal = state.propose(
                &proposed.path,
                &formatting.output.stdout,
                &proposed.justification,
            )?;
            Ok(
                json!({"applied":false,"approval":"not_requested","formatted":proposed.content!=proposal.content,"formatting":formatting,"next_tool":{"name":"repo_apply_patch","arguments":{"operation":proposal.operation}},"proposal":proposal}),
            )
        }
        "repo_apply_patch" => {
            let operation = string(args, "operation")?;
            let receipt = state.apply(operation)?;
            Ok(
                json!({"operation":operation,"applied":receipt.is_some(),"decision":state.approvals.get(operation),"receipt":receipt}),
            )
        }
        "repo_validate" => {
            let phase: Phase = serde_json::from_value(
                args.get("phase")
                    .cloned()
                    .ok_or_else(|| Error::Invariant("validation needs phase".into()))?,
            )?;
            Ok(json!({"validation":state.validate_until(phase, Some(deadline))?}))
        }
        _ => Err(Error::Invariant(format!("unknown repository tool {name}"))),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Publication {
    pub operation: String,
    pub applied: bool,
    pub before: Option<String>,
    pub after: Option<String>,
}
impl ContextValue for Publication {
    const KEY: &'static str = "maintenance.repair-write-receipt.v1";
}
impl Publication {
    pub fn from_answer(value: &Value) -> Result<Self, Error> {
        Ok(Self {
            operation: string(value, "operation")?.into(),
            applied: value
                .get("applied")
                .and_then(Value::as_bool)
                .ok_or_else(|| Error::Invariant("write result lacks applied status".into()))?,
            before: value
                .pointer("/receipt/patch/before")
                .and_then(Value::as_str)
                .map(str::to_owned),
            after: value
                .pointer("/receipt/after")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }
}

/// Project only results visible after Collect. These observations are outputs;
/// replay must run the same approval policy before it may project an edit.
pub(super) fn observe(
    state: &mut State,
    name: &str,
    args: &Value,
    value: &Value,
    replay: bool,
) -> Result<(), Error> {
    match name {
        "repo_propose_patch" => {
            let proposal: Patch = serde_json::from_value(
                value
                    .get("proposal")
                    .cloned()
                    .ok_or_else(|| Error::Invariant("proposal missing at Collect".into()))?,
            )?;
            let formatting: validation::Step =
                serde_json::from_value(value.get("formatting").cloned().ok_or_else(|| {
                    Error::Invariant("proposal formatting missing at Collect".into())
                })?)?;
            let expected_command = ["rustfmt", "--edition=2024", "--emit=stdout", "<proposal>"];
            if formatting.command != expected_command
                || formatting.output.code != Some(0)
                || formatting.output.timed_out
                || formatting.output.output_limit
                || formatting.output.stdout != proposal.content
                || string(args, "path")? != proposal.path
                || string(args, "justification")? != proposal.justification
                || value.get("formatted").and_then(Value::as_bool)
                    != Some(string(args, "content")? != proposal.content)
                || value.get("applied").and_then(Value::as_bool) != Some(false)
                || value.get("approval").and_then(Value::as_str) != Some("not_requested")
                || value.get("next_tool")
                    != Some(
                        &json!({"name":"repo_apply_patch","arguments":{"operation":proposal.operation}}),
                    )
            {
                return Err(Error::Invariant(
                    "proposal formatting or issued request evidence differs at Collect".into(),
                ));
            }
            state.observe_proposal(proposal)
        }
        "repo_validate" => state.observe_report(serde_json::from_value(
            value
                .get("validation")
                .cloned()
                .ok_or_else(|| Error::Invariant("validation missing at Collect".into()))?,
        )?),
        "repo_apply_patch" => {
            let operation = string(value, "operation")?;
            let decision = state
                .approvals
                .get(operation)
                .ok_or_else(|| Error::Invariant("observed write has no host decision".into()))?;
            if serde_json::to_value(decision)? != value["decision"] {
                return Err(Error::Invariant(
                    "write outcome changed host approval decision".into(),
                ));
            }
            if value["applied"] == true {
                if *decision != Approval::Approve {
                    return Err(Error::Invariant("unapproved replayed write".into()));
                }
                let receipt: Receipt = serde_json::from_value(value["receipt"].clone())?;
                if receipt.patch.operation != operation {
                    return Err(Error::Invariant("write receipt operation differs".into()));
                }
                state.observe_receipt(&receipt, replay)
            } else if matches!(decision, Approval::Deny | Approval::Cancel)
                && value["applied"] == false
                && value["receipt"].is_null()
            {
                Ok(())
            } else {
                Err(Error::Invariant(
                    "approved write has no applied receipt".into(),
                ))
            }
        }
        _ => Ok(()),
    }
}
