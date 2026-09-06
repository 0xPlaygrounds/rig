//! A finite work queue from every manifest cell plus explicit programme gates.
//! Historical verification is reported separately, never inferred as current.

use super::*;
use serde_json::json;
use std::collections::BTreeSet;

fn generate(manifest: &Value, requirements: &Value) -> Result<Value> {
    let mut seen = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut groups = BTreeMap::<(String, String), Vec<String>>::new();
    let mut mapped = 0;
    let mut historical_verified = 0;
    for row in array(manifest, "scenarios")? {
        let id = text(row, "id")?;
        if !seen.insert(id) {
            return Err(format!("duplicate scenario {id}").into());
        }
        let classification = text(row, "classification")?;
        *counts.entry(classification.into()).or_default() += 1;
        if !field(row, "ecs").is_null() {
            mapped += 1;
        }
        let verified = field(field(row, "scoped_parity"), "status") == "verified";
        historical_verified += usize::from(verified);
        let action = match classification {
            "unclassified" => Some("classify_helper_closure"),
            "agent" if field(row, "ecs").is_null() => Some("implement_native_family"),
            "agent" if !verified => Some("verify_mapped_family"),
            "agent" | "shared_provider" | "infrastructure"
                if field(row, "inventory_complete") != true
                    || field(row, "fixture_inventory_complete") != true =>
            {
                Some("complete_source_and_fixture_inventory")
            }
            "agent" | "shared_provider" | "infrastructure" => None,
            other => return Err(format!("unknown classification {other}").into()),
        };
        if let Some(action) = action {
            groups
                .entry((text(row, "source")?.into(), action.into()))
                .or_default()
                .push(id.into());
        }
    }
    let mut items = Vec::new();
    for ((family, action), ids) in groups {
        let (proof, prerequisite, next) = match action.as_str() {
            "classify_helper_closure" => (
                "positive classification of every entry point and helper exception",
                "pinned source and complete compiled registration",
                "review the complete helper family, record one source-bound rule and all exceptions",
            ),
            "implement_native_family" => (
                "native counterpart preserving original obligations",
                "reviewed original helper/assertion closure",
                "implement the family using existing native APIs; fix any shared runtime gap",
            ),
            "verify_mapped_family" => (
                "accepted semantic review plus paired source-bound execution",
                "implemented counterparts and comparison contract",
                "run exact family IDs, review obligations, apply accepted decisions",
            ),
            _ => (
                "complete source/helper and fixture inventory",
                "classification rule",
                "resolve inherited helpers, literal and computed fixture consumers, and feature cells",
            ),
        };
        items.push(json!({"family":family,"scenario_ids":ids,"missing_proof":proof,"prerequisite":prerequisite,"next_action":next,"action":action,"evidence_reference":"provider-baseline.json scenario rows"}));
    }
    let mut requirement_ids = BTreeSet::new();
    for requirement in array(requirements, "requirements")? {
        let id = text(requirement, "id")?;
        if !requirement_ids.insert(id) {
            return Err(format!("duplicate requirement {id}").into());
        }
        for key in [
            "requirement_reference",
            "missing_proof",
            "prerequisite",
            "next_action",
            "evidence_reference",
        ] {
            text(requirement, key)?;
        }
        // These are outstanding gates supplied from the objective, not statuses
        // synthesized from the presence of a file or a historical green result.
        items.push(requirement.clone());
    }
    Ok(
        json!({"schema":1,"baseline_revision":field(manifest,"baseline_revision"),
        "manifest_sha256":hash(&bytes(manifest)?),"requirements_sha256":hash(&bytes(requirements)?),
        "counts":{"scenarios":seen.len(),"classifications":counts,"mapped":mapped,"historical_scoped_verified":historical_verified,"queue_items":items.len()},
        "limits":"Queue is not a completion verdict. Historical verified rows still require dependency-aware current-evidence and aggregate gates listed below. Root provider manifest is not companion-crate discovery.","items":items}),
    )
}

pub(crate) fn run(args: Vec<String>) -> Result<()> {
    let [manifest, requirements, output] = args.as_slice() else {
        return Err("expected manifest, requirements and output paths".into());
    };
    let queue = generate(&read(Path::new(manifest))?, &read(Path::new(requirements))?)?;
    write(Path::new(output), &queue)?;
    println!("{}", field(&queue, "counts"));
    Ok(())
}

#[cfg(test)]
mod tests;
