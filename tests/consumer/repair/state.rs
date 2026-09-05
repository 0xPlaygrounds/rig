//! Approval and phase state, separate from provider assertions and process execution.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::{Approval, Error};
use super::{
    project::{self, Image, Project},
    validation::{Phase, Report},
};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Patch {
    pub operation: String,
    pub path: String,
    pub content: String,
    pub before: String,
    pub diff: String,
    pub justification: String,
}

impl Patch {
    fn new(image: &Image, path: &str, content: &str, justification: &str) -> Result<Self, Error> {
        if !matches!(path, "src/lib.rs" | "tests/regression.rs")
            || content.len() > 16 * 1024
            || !content.ends_with('\n')
            || justification.trim().is_empty()
            || justification.len() > 4096
        {
            return Err(Error::Invariant("patch needs a declared path, newline-terminated content <=16 KiB and justification <=4 KiB".into()));
        }
        let old = image.get(path).map_or("", String::as_str);
        if old == content {
            return Err(Error::Invariant("patch makes no change".into()));
        }
        let before = project::digest(image)?;
        let mut diff = format!(
            "--- a/{path}\n+++ b/{path}\n@@ -1,{} +1,{} @@\n",
            old.lines().count(),
            content.lines().count()
        );
        for line in old.lines() {
            diff.push('-');
            diff.push_str(line);
            diff.push('\n');
        }
        for line in content.lines() {
            diff.push('+');
            diff.push_str(line);
            diff.push('\n');
        }
        let operation = project::content_digest(&serde_json::to_string(&(
            path,
            content,
            &before,
            &diff,
            justification,
        ))?);
        Ok(Self {
            operation,
            path: path.into(),
            content: content.into(),
            before,
            diff,
            justification: justification.into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Receipt {
    pub patch: Patch,
    pub after: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub image: Image,
    pub writes: usize,
    pub initial: Option<Report>,
    pub regression: Option<Report>,
    pub final_report: Option<Report>,
    pub pending: Option<Patch>,
    pub approvals: BTreeMap<String, Approval>,
    pub ledger: Vec<Receipt>,
}

pub(crate) struct State {
    pub project: Project,
    pub initial: Option<Report>,
    pub regression: Option<Report>,
    pub final_report: Option<Report>,
    pub pending: Option<Patch>,
    pub approvals: BTreeMap<String, Approval>,
    pub ledger: Vec<Receipt>,
}

impl State {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            project: Project::new()?,
            initial: None,
            regression: None,
            final_report: None,
            pending: None,
            approvals: BTreeMap::new(),
            ledger: Vec::new(),
        })
    }

    fn phase_ready(&self, phase: Phase) -> Result<(), Error> {
        let ready = match phase {
            Phase::Initial => self.ledger.is_empty(),
            Phase::Regression => {
                self.initial.as_ref().is_some_and(|report| report.accepted)
                    && self.ledger.len() == 1
                    && self
                        .ledger
                        .first()
                        .is_some_and(|receipt| receipt.patch.path == "tests/regression.rs")
            }
            Phase::Final => {
                self.regression
                    .as_ref()
                    .is_some_and(|report| report.accepted)
                    && self.ledger.len() == 2
                    && self
                        .ledger
                        .get(1)
                        .is_some_and(|receipt| receipt.patch.path == "src/lib.rs")
            }
        };
        if !ready {
            return Err(Error::Invariant(format!(
                "validation phase {phase:?} is not ready"
            )));
        }
        Ok(())
    }

    /// Invoked by the live tool. Reports become policy state only at Collect.
    pub fn validate(&self, phase: Phase) -> Result<Report, Error> {
        self.validate_until(phase, None)
    }

    pub fn validate_until(
        &self,
        phase: Phase,
        deadline: Option<std::time::Instant>,
    ) -> Result<Report, Error> {
        self.phase_ready(phase)?;
        super::validation::validate_until(&self.project, phase, deadline)
    }

    pub fn observe_report(&mut self, report: Report) -> Result<(), Error> {
        report.verify()?;
        self.phase_ready(report.phase)?;
        if report.project_digest != project::digest(&self.project.image()?)? {
            return Err(Error::Invariant(
                "validation belongs to a different project revision".into(),
            ));
        }
        match report.phase {
            Phase::Initial => self.initial = Some(report),
            Phase::Regression => self.regression = Some(report),
            Phase::Final => self.final_report = Some(report),
        }
        Ok(())
    }

    pub fn propose(&self, path: &str, content: &str, justification: &str) -> Result<Patch, Error> {
        if !matches!(path, "tests/regression.rs" | "src/lib.rs") {
            return Err(Error::Invariant("immutable or undeclared patch path; add the regression in tests/regression.rs, then repair src/lib.rs after its behavioral failure".into()));
        }
        let image = self.project.image()?;
        let digest = project::digest(&image)?;
        let proof = match path {
            "tests/regression.rs" if self.ledger.is_empty() => self.initial.as_ref(),
            "src/lib.rs" if self.ledger.len() == 1 => self.regression.as_ref(),
            _ => None,
        };
        if !proof.is_some_and(|report| report.accepted && report.project_digest == digest) {
            return Err(Error::Invariant("patch requires observed behavioral failure on this exact project revision; regression precedes production".into()));
        }
        Patch::new(&image, path, content, justification)
    }

    pub fn observe_proposal(&mut self, patch: Patch) -> Result<(), Error> {
        if self.propose(&patch.path, &patch.content, &patch.justification)? != patch {
            return Err(Error::Invariant(
                "proposal identity or exact diff changed".into(),
            ));
        }
        self.pending = Some(patch);
        Ok(())
    }

    /// This decision is supplied by the declared host script, never tool args.
    /// Unchanged approval is reused; changing its decision requires a new run.
    pub fn approve(&mut self, operation: &str, decision: Approval) -> Result<Patch, Error> {
        let patch = self
            .pending
            .as_ref()
            .ok_or_else(|| Error::Invariant("approval has no observed proposal".into()))?;
        if patch.operation != operation
            || self.propose(&patch.path, &patch.content, &patch.justification)? != *patch
        {
            return Err(Error::Invariant(
                "approval does not match current diff and source".into(),
            ));
        }
        if let Some(previous) = self.approvals.get(operation) {
            if *previous != decision {
                return Err(Error::Invariant(
                    "host approval decision changed for an existing operation".into(),
                ));
            }
        } else {
            self.approvals.insert(operation.into(), decision);
        }
        Ok(patch.clone())
    }

    pub fn apply(&mut self, operation: &str) -> Result<Option<Receipt>, Error> {
        if self
            .ledger
            .iter()
            .any(|receipt| receipt.patch.operation == operation)
        {
            return Err(Error::Invariant(
                "duplicate operation: external edit already happened; reconcile its saved outcome"
                    .into(),
            ));
        }
        let patch = self
            .pending
            .clone()
            .ok_or_else(|| Error::Invariant("apply has no observed proposal".into()))?;
        if patch.operation != operation
            || self.propose(&patch.path, &patch.content, &patch.justification)? != patch
        {
            return Err(Error::Invariant(
                "apply does not match approved diff and source".into(),
            ));
        }
        match self.approvals.get(operation) {
            Some(Approval::Approve) => (),
            Some(Approval::Deny | Approval::Cancel) => return Ok(None),
            None => return Err(Error::Invariant("apply has no host approval".into())),
        }
        let after = self
            .project
            .apply(&patch.path, &patch.content, &patch.before)?;
        let receipt = Receipt { patch, after };
        self.ledger.push(receipt.clone());
        self.final_report = None;
        Ok(Some(receipt))
    }

    /// Live execution already wrote; replay projects the recorded edit into its
    /// own workspace using the same approval/digest checks, without a subprocess.
    pub fn observe_receipt(&mut self, receipt: &Receipt, replay: bool) -> Result<(), Error> {
        if replay
            && self
                .ledger
                .iter()
                .any(|saved| saved.patch.operation == receipt.patch.operation)
        {
            return Err(Error::Invariant(
                "duplicate operation: external edit already happened; reconcile its saved outcome"
                    .into(),
            ));
        }
        if replay {
            let mut image = self.project.image()?;
            if self.pending.as_ref() != Some(&receipt.patch)
                || receipt.patch.before != project::digest(&image)?
            {
                return Err(Error::Invariant(
                    "replay receipt does not match current approved proposal".into(),
                ));
            }
            image.insert(receipt.patch.path.clone(), receipt.patch.content.clone());
            if project::digest(&image)? != receipt.after {
                return Err(Error::Invariant(
                    "replay receipt postimage differs before mutation".into(),
                ));
            }
        }
        if replay && self.apply(&receipt.patch.operation)?.as_ref() != Some(receipt) {
            return Err(Error::Invariant(
                "replayed edit differs from approved operation".into(),
            ));
        }
        if self.ledger.last() != Some(receipt)
            || project::digest(&self.project.image()?)? != receipt.after
        {
            return Err(Error::Invariant(
                "write outcome differs from external ledger or project".into(),
            ));
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Result<Snapshot, Error> {
        Ok(Snapshot {
            image: self.project.image()?,
            writes: self.project.writes(),
            initial: self.initial.clone(),
            regression: self.regression.clone(),
            final_report: self.final_report.clone(),
            pending: self.pending.clone(),
            approvals: self.approvals.clone(),
            ledger: self.ledger.clone(),
        })
    }

    pub fn restore(snapshot: Snapshot) -> Result<Self, Error> {
        if snapshot.writes != snapshot.ledger.len() || snapshot.ledger.len() > 2 {
            return Err(Error::Invariant(
                "saved mutation count differs from operation ledger".into(),
            ));
        }
        let mut image = project::initial();
        let initial_digest = project::digest(&image)?;
        for (report, phase, digest) in [
            (
                snapshot.initial.as_ref(),
                Phase::Initial,
                Some(initial_digest.as_str()),
            ),
            (
                snapshot.regression.as_ref(),
                Phase::Regression,
                snapshot
                    .ledger
                    .first()
                    .map(|receipt| receipt.after.as_str()),
            ),
            (
                snapshot.final_report.as_ref(),
                Phase::Final,
                snapshot.ledger.get(1).map(|receipt| receipt.after.as_str()),
            ),
        ] {
            if let Some(report) = report {
                report.verify()?;
            }
            if let Some(report) = report
                && (report.phase != phase || Some(report.project_digest.as_str()) != digest)
            {
                return Err(Error::Invariant(
                    "saved validation phase or project revision is inconsistent".into(),
                ));
            }
        }
        for (index, receipt) in snapshot.ledger.iter().enumerate() {
            let expected_path = if index == 0 {
                "tests/regression.rs"
            } else {
                "src/lib.rs"
            };
            let proof = if index == 0 {
                snapshot.initial.as_ref()
            } else {
                snapshot.regression.as_ref()
            };
            if receipt.patch.path != expected_path
                || snapshot.approvals.get(&receipt.patch.operation) != Some(&Approval::Approve)
                || !proof.is_some_and(|proof| {
                    proof.accepted && proof.project_digest == receipt.patch.before
                })
                || Patch::new(
                    &image,
                    &receipt.patch.path,
                    &receipt.patch.content,
                    &receipt.patch.justification,
                )? != receipt.patch
            {
                return Err(Error::Invariant(
                    "saved edit lacks matching preimage, regression proof or approval".into(),
                ));
            }
            image.insert(receipt.patch.path.clone(), receipt.patch.content.clone());
            if project::digest(&image)? != receipt.after {
                return Err(Error::Invariant("saved edit postimage differs".into()));
            }
        }
        if image != snapshot.image {
            return Err(Error::Invariant(
                "saved project differs from recorded edits".into(),
            ));
        }
        let state = Self {
            project: Project::restore(&snapshot.image, snapshot.writes)?,
            initial: snapshot.initial,
            regression: snapshot.regression,
            final_report: snapshot.final_report,
            pending: snapshot.pending,
            approvals: snapshot.approvals,
            ledger: snapshot.ledger,
        };
        if let Some(report) = &state.final_report {
            state.phase_ready(Phase::Final)?;
            if report.phase != Phase::Final
                || report.project_digest != project::digest(&state.project.image()?)?
            {
                return Err(Error::Invariant("saved final validation is stale".into()));
            }
        }
        Ok(state)
    }

    pub fn validated(&self) -> bool {
        self.final_report.as_ref().is_some_and(|report| {
            report.accepted
                && self.ledger.len() == 2
                && project::digest(&self.project.image().unwrap_or_default())
                    .is_ok_and(|digest| digest == report.project_digest)
        })
    }
}
