//! Apply explicit independent review decisions, not inferred semantic verdicts.
//! Source hashes protect the reviewed inputs; humans/agents still own the
//! completeness and validity of the stated assertion or classification review.

use super::*;
use std::collections::BTreeSet;

fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.into()) }
}

fn apply(
    manifest: &mut Value,
    review: &Value,
    source: &Path,
    evidence: &Path,
    review_path: &str,
) -> Result<()> {
    require(field(review, "status") == "accepted", "review not accepted")?;
    require(
        !text(review, "reviewer")?.trim().is_empty(),
        "missing independent reviewer",
    )?;
    require(
        field(review, "baseline_revision") == field(manifest, "baseline_revision"),
        "baseline mismatch",
    )?;
    let kind = text(review, "kind")?;
    require(
        matches!(kind, "parity" | "classification"),
        "unsupported review kind",
    )?;
    let sources = field(review, "sources")
        .as_object()
        .ok_or("missing reviewed sources")?;
    require(!sources.is_empty(), "empty reviewed source closure")?;
    let mut manifest_ids = BTreeSet::new();
    for row in array(manifest, "scenarios")? {
        require(
            manifest_ids.insert(text(row, "id")?),
            "duplicate manifest scenario",
        )?;
    }
    for (path, digest) in sources {
        require(
            digest.as_str() == Some(hash(&fs::read(checked(source, path)?)?).as_str()),
            "reviewed source drift",
        )?;
    }
    let mut executed = BTreeMap::<String, BTreeSet<String>>::new();
    let mut paired_batch_name: Option<String> = None;
    if kind == "parity" {
        for surface in ["original", "native"] {
            let name = text(field(review, "runs"), surface)?;
            let path = checked(evidence, &format!("runs/{name}"))?;
            let raw = fs::read(path)?;
            require(
                name == format!("{}.json", hash(&raw)),
                "run artifact hash mismatch",
            )?;
            let run: Value = serde_json::from_slice(&raw)?;
            require(field(&run, "surface") == surface, "run surface mismatch")?;
            let batch_name = text(&run, "batch")?;
            if let Some(expected) = &paired_batch_name {
                require(expected == batch_name, "runs belong to different batches")?;
            } else {
                paired_batch_name = Some(batch_name.into());
            }
            let batch_raw = fs::read(checked(evidence, &format!("batches/{batch_name}"))?)?;
            require(
                batch_name == format!("{}.json", hash(&batch_raw)),
                "batch artifact hash mismatch",
            )?;
            let batch: Value = serde_json::from_slice(&batch_raw)?;
            require(
                field(&run, "status") == "executed" && field(&run, "exit_code") == 0,
                "run incomplete",
            )?;
            require(
                field(&run, "source_index") == field(&run, "source_index_after"),
                "execution source drift",
            )?;
            if surface == "original" {
                require(
                    field(&run, "revision") == field(manifest, "baseline_revision"),
                    "original revision mismatch",
                )?;
            }
            let index_name = text(&run, "source_index")?;
            let index_raw = fs::read(checked(evidence, &format!("provenance/{index_name}"))?)?;
            require(
                index_name == format!("{}.json", hash(&index_raw)),
                "source index hash mismatch",
            )?;
            let index: Value = serde_json::from_slice(&index_raw)?;
            if surface == "native" {
                for (path, digest) in sources {
                    require(
                        index.get(path) == Some(digest),
                        "review not bound to tested native source",
                    )?;
                }
            }
            let mut ids = BTreeSet::new();
            for result in array(&run, "results")? {
                require(
                    field(result, "status") == "ok" && field(result, "expected") == "ok",
                    "nonpassing selected result",
                )?;
                require(
                    ids.insert(text(result, "id")?.to_owned()),
                    "duplicate executed ID",
                )?;
            }
            require(!ids.is_empty(), "empty execution")?;
            let batch_cells = array(&batch, "cells")?;
            let selected: BTreeSet<String> = batch_cells
                .iter()
                .map(|cell| text(cell, surface).map(str::to_owned))
                .collect::<Result<_>>()?;
            require(
                selected.len() == batch_cells.len() && selected == ids,
                "executed IDs differ from batch",
            )?;
            executed.insert(surface.into(), ids);
        }
    }
    let mut decisions = BTreeSet::new();
    // Validate into a clone; failure cannot partially update the manifest.
    let mut updated = manifest.clone();
    for cell in array(review, "cells")? {
        let id = text(cell, "id")?;
        require(decisions.insert(id), "duplicate review cell")?;
        let row = updated
            .get_mut("scenarios")
            .and_then(Value::as_array_mut)
            .ok_or("missing scenarios")?
            .iter_mut()
            .find(|row| field(row, "id") == id)
            .ok_or("unknown review cell")?;
        require(
            sources.contains_key(text(row, "source")?),
            "original source not reviewed",
        )?;
        let fields = field(cell, "fields")
            .as_object()
            .ok_or("missing explicit reviewed fields")?;
        if kind == "parity" {
            require(
                field(row, "classification") == "agent",
                "provider-only row cannot receive agent parity",
            )?;
            for surface in ["original", "native"] {
                require(
                    executed
                        .get(surface)
                        .is_some_and(|ids| ids.contains(text(cell, surface).unwrap_or(""))),
                    "reviewed case not executed",
                )?;
            }
            let native = fields.get("ecs").unwrap_or_else(|| field(row, "ecs"));
            require(
                sources.contains_key(text(native, "source")?),
                "native source not reviewed",
            )?;
            require(
                text(cell, "native")?
                    == format!("rig::{}${}", text(native, "binary")?, text(native, "test")?),
                "native mapping mismatch",
            )?;
            require(
                text(cell, "original")?.replace('$', "::") == id,
                "original mapping mismatch",
            )?;
        }
        for (key, value) in fields {
            let allowed = if kind == "classification" {
                ["classification", "classification_reason", "evidence_scope"]
                    .contains(&key.as_str())
            } else {
                [
                    "ecs",
                    "assertion_mappings",
                    "helper_obligations",
                    "family_contract",
                    "candidate_execution_evidence",
                    "inventory_complete",
                    "assertion_sources",
                    "fixture_inventory_complete",
                    "fixtures",
                ]
                .contains(&key.as_str())
            };
            require(allowed, "unapproved review field")?;
            row.as_object_mut()
                .ok_or("row object")?
                .insert(key.clone(), value.clone());
        }
        if kind == "classification" {
            require(
                matches!(
                    text(row, "classification")?,
                    "agent" | "shared_provider" | "infrastructure"
                ),
                "invalid classification",
            )?;
            row["classification_review"] = Value::String(review_path.into());
        } else {
            if field(row, "inventory_complete") == true {
                require(
                    !array(row, "assertion_sources")?.is_empty(),
                    "completed inventory requires assertion sources",
                )?;
                for anchor in array(row, "assertion_sources")? {
                    require(
                        sources.contains_key(text(anchor, "path")?),
                        "assertion source outside reviewed closure",
                    )?;
                }
            }
            // Exact assertion anchors are checked against pinned discovery by
            // parity-manifest report. Here protect reviewed fixture bytes and
            // require them to have been part of the tested native source index.
            let fixtures = fields
                .get("fixtures")
                .map(|value| value.as_array().ok_or("fixtures must be an array"))
                .transpose()?;
            for fixture in fixtures.into_iter().flatten() {
                let path = text(fixture, "path")?;
                require(
                    sources.get(path).and_then(Value::as_str) == Some(text(fixture, "sha256")?),
                    "fixture outside reviewed tested closure",
                )?;
            }
            require(
                !array(row, "assertion_mappings")?.is_empty(),
                "missing assertion mappings",
            )?;
            row["status"] = Value::String("executed".into());
            row["scoped_parity"] = serde_json::json!({"status":"verified","review":review_path,"configuration":text(review,"configuration")?,"limits":text(review,"limits")?});
            row.as_object_mut()
                .ok_or("row object")?
                .remove("native_gap");
        }
    }
    require(!decisions.is_empty(), "empty review")?;
    *manifest = updated;
    Ok(())
}

pub(crate) fn run(args: Vec<String>) -> Result<()> {
    let [manifest_path, review_path, source, evidence] = args.as_slice() else {
        return Err("expected manifest, review, source root, evidence root".into());
    };
    let mut manifest = read(Path::new(manifest_path))?;
    apply(
        &mut manifest,
        &read(Path::new(review_path))?,
        Path::new(source),
        Path::new(evidence),
        review_path,
    )?;
    write(Path::new(manifest_path), &manifest)?;
    println!("Applied explicit accepted review: {review_path}");
    Ok(())
}

#[cfg(test)]
mod tests;
