//! Source-validated manifest inventory. Execution verdicts remain separate.

use super::*;
use serde_json::json;
use std::{collections::BTreeSet, io::Write};

#[cfg(test)]
mod tests;

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn rows(discovery: &Value) -> Result<BTreeMap<String, &Value>> {
    require(
        field(discovery, "evidence_kind") == "discovery_not_coverage",
        "expected source/compiled discovery",
    )?;
    require(
        !text(discovery, "source_revision")?.is_empty() && field(discovery, "source_status") == "",
        "baseline discovery must identify a clean immutable revision",
    )?;
    require(
        array(discovery, "unresolved_compiled_tests")?.is_empty(),
        "resolve compiled registrations before inspecting manifest",
    )?;
    let mut rows = BTreeMap::new();
    for row in array(discovery, "tests")? {
        let id = format!("rig::{}::{}", text(row, "binary")?, text(row, "test")?);
        require(
            rows.insert(id.clone(), row).is_none(),
            format!("duplicate discovery ID: {id}"),
        )?;
    }
    require(!rows.is_empty(), "empty baseline discovery")?;
    Ok(rows)
}

fn configuration(discovery: &Value, evidence: &Value) -> Result<Value> {
    require(
        field(evidence, "schema") == 1
            && field(evidence, "baseline_listing_sha256") == field(discovery, "listing_sha256"),
        "configuration evidence does not match baseline listing",
    )?;
    let configurations = field(evidence, "configurations")
        .as_object()
        .ok_or("missing configurations")?;
    require(
        configurations.len() == 1,
        "one listing requires one configuration",
    )?;
    for value in configurations.values() {
        require(
            !text(value, "package")?.is_empty() && !text(value, "listing_command")?.is_empty(),
            "incomplete configuration evidence",
        )?;
        array(value, "features")?;
    }
    Ok(json!(configurations))
}

fn insert_source(sources: &mut BTreeMap<String, String>, path: &str, hash: &str) -> Result<()> {
    require(!hash.is_empty(), "empty source hash")?;
    if let Some(prior) = sources.insert(path.into(), hash.into()) {
        require(prior == hash, format!("inconsistent source hash: {path}"))?;
    }
    Ok(())
}

fn optional_array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value]> {
    match value.get(field) {
        None => Ok(&[]),
        Some(v) => v
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| format!("invalid array {field}").into()),
    }
}

fn initialize(discovery: &Value, evidence: &Value) -> Result<Value> {
    let originals = rows(discovery)?;
    let configurations = configuration(discovery, evidence)?;
    let configuration = configurations
        .as_object()
        .and_then(|c| c.keys().next())
        .ok_or("empty configuration")?;
    let mut sources = BTreeMap::new();
    let mut scenarios = Vec::new();
    for (id, row) in originals {
        insert_source(
            &mut sources,
            text(row, "source")?,
            text(row, "source_sha256")?,
        )?;
        for source in optional_array(row, "registration_sources")? {
            insert_source(&mut sources, text(source, "path")?, text(source, "sha256")?)?;
        }
        scenarios.push(json!({"id":id, "source":field(row, "source"), "line":field(row, "line"), "configuration":configuration,
            "declared_gates":field(row, "attributes"), "classification":field(row, "classification"), "classification_reason":field(row, "classification_reason"),
            "inventory_complete":false, "fixture_inventory_complete":false, "fixtures":[], "assertion_mappings":[], "ecs":null, "status":"unreviewed"}));
    }
    for category in ["source_helpers", "source_imports", "source_macros"] {
        for source in optional_array(discovery, category)? {
            insert_source(
                &mut sources,
                text(source, "source")?,
                text(source, "source_sha256")?,
            )?;
        }
    }
    Ok(
        json!({"schema":1, "scope":"root provider baseline; companion and supplemental inventory remains separate and incomplete",
        "baseline_revision":field(discovery, "source_revision"), "baseline_listing_sha256":field(discovery, "listing_sha256"),
        "configurations":configurations, "sources":sources, "scenarios":scenarios}),
    )
}

fn anchors(source: &Value) -> Result<Value> {
    let anchors: Vec<_> = array(source, "assertions")?.iter().map(|a| Ok(json!({"line":a["line"], "expression_sha256":hash(text(a, "assertion")?.as_bytes())}))).collect::<Result<_>>()?;
    Ok(json!(anchors))
}

fn validate_assertions(
    row: &Value,
    original: &Value,
    discovery: &Value,
    sources: &Value,
) -> Result<()> {
    let assertions = optional_array(row, "assertion_sources")?;
    if field(row, "inventory_complete") != true && assertions.is_empty() {
        return Ok(());
    }
    require(
        !assertions.is_empty(),
        "completed inventory lacks assertion sources",
    )?;
    for assertion in assertions {
        require(
            sources.get(text(assertion, "path")?).is_some(),
            "unhashed assertion source",
        )?;
    }
    if field(original, "registration_kind") == "reviewed_macro_mapping" {
        return require(
            field(row, "assertion_sources") == field(original, "assertion_sources"),
            "generated assertion provenance drift",
        );
    }
    let direct: Vec<_> = assertions
        .iter()
        .filter(|a| {
            field(a, "path") == field(original, "source")
                && field(a, "symbol") == field(original, "test")
        })
        .collect();
    require(
        direct.len() == 1,
        "direct assertion inventory must have one anchor",
    )?;
    require(
        direct.first().ok_or("missing direct anchor")?["direct_assertions"] == anchors(original)?,
        "direct assertion inventory drift",
    )?;
    for assertion in assertions {
        if field(assertion, "path") == field(original, "source")
            && field(assertion, "symbol") == field(original, "test")
        {
            continue;
        }
        let helpers: Vec<_> = optional_array(discovery, "source_helpers")?
            .iter()
            .filter(|h| {
                field(h, "binary") == field(original, "binary")
                    && field(h, "function") == field(assertion, "symbol")
                    && field(h, "source") == field(assertion, "path")
            })
            .collect();
        require(
            helpers.len() == 1,
            "unresolved or ambiguous helper assertion source",
        )?;
        require(
            *field(assertion, "direct_assertions")
                == anchors(helpers.first().ok_or("missing helper")?)?,
            "helper assertion inventory drift",
        )?;
    }
    Ok(())
}

fn inspect(
    manifest: &Value,
    discovery: &Value,
    baseline: &Path,
    candidate: &Path,
    candidate_discovery: &Value,
    evidence: &Value,
) -> Result<Value> {
    require(
        field(manifest, "schema") == 1,
        "unsupported manifest schema",
    )?;
    let originals = rows(discovery)?;
    require(
        *field(manifest, "configurations") == configuration(discovery, evidence)?,
        "configuration evidence drift",
    )?;
    require(
        field(manifest, "baseline_revision") == field(discovery, "source_revision"),
        "baseline revision mismatch",
    )?;
    require(
        field(manifest, "baseline_listing_sha256") == field(discovery, "listing_sha256"),
        "baseline listing mismatch",
    )?;
    let expected = initialize(discovery, evidence)?;
    for (source, hash) in field(&expected, "sources")
        .as_object()
        .ok_or("missing expected sources")?
    {
        require(
            field(manifest, "sources").get(source) == Some(hash),
            "missing or changed discovered source hash",
        )?;
    }
    for (source, expected) in field(manifest, "sources")
        .as_object()
        .ok_or("missing sources")?
    {
        require(
            hash(&fs::read(checked(baseline, source)?)?)
                == expected.as_str().ok_or("invalid source hash")?,
            format!("baseline source drift: {source}"),
        )?;
    }
    require(
        field(candidate_discovery, "evidence_kind") == "discovery_not_coverage",
        "expected candidate source/compiled discovery",
    )?;
    let mut compiled = BTreeMap::new();
    for row in optional_array(candidate_discovery, "tests")? {
        if field(row, "compiled") == true {
            let key = (text(row, "binary")?, text(row, "test")?);
            require(
                compiled.insert(key, row).is_none(),
                "duplicate candidate registration",
            )?;
        }
    }
    require(!compiled.is_empty(), "empty candidate compiled discovery")?;
    let mut seen = BTreeSet::new();
    let mut counterparts = BTreeSet::new();
    let mut classifications = BTreeMap::<String, usize>::new();
    let mut agent_scopes = BTreeMap::<String, usize>::new();
    let mut unresolved = Vec::new();
    let mut mapped = 0usize;
    for row in array(manifest, "scenarios")? {
        let id = text(row, "id")?;
        require(
            seen.insert(id.to_owned()),
            format!("duplicate manifest ID: {id}"),
        )?;
        let original = originals
            .get(id)
            .ok_or_else(|| format!("manifest scenario absent from discovery: {id}"))?;
        let source = text(original, "source")?;
        require(
            field(row, "source") == source
                && field(field(manifest, "sources"), source) == field(original, "source_sha256"),
            format!("source mapping drift: {id}"),
        )?;
        require(
            field(row, "declared_gates") == field(original, "attributes"),
            format!("declared configuration drift: {id}"),
        )?;
        for dependency in optional_array(original, "registration_sources")? {
            require(
                field(field(manifest, "sources"), text(dependency, "path")?)
                    == field(dependency, "sha256"),
                "missing generated dependency hash",
            )?;
        }
        let kind = text(row, "classification")?;
        require(
            matches!(
                kind,
                "agent" | "shared_provider" | "infrastructure" | "unclassified"
            ),
            format!("unknown classification: {id}"),
        )?;
        *classifications.entry(kind.into()).or_default() += 1;
        if kind == "agent" {
            *agent_scopes
                .entry(
                    field(row, "evidence_scope")
                        .as_str()
                        .unwrap_or("unresolved")
                        .into(),
                )
                .or_default() += 1;
        }
        if kind != "unclassified" {
            require(
                !text(row, "classification_reason")?.is_empty(),
                "missing classification reason",
            )?;
        }
        require(
            field(manifest, "configurations")
                .get(text(row, "configuration")?)
                .is_some(),
            "unknown configuration",
        )?;
        for dependency in optional_array(row, "configuration_sources")? {
            require(
                hash(&fs::read(checked(baseline, text(dependency, "path")?)?)?)
                    == text(dependency, "sha256")?,
                "configuration source drift",
            )?;
        }
        for fixture in array(row, "fixtures")? {
            let path = text(fixture, "path")?;
            for root in [baseline, candidate] {
                require(
                    hash(&fs::read(checked(root, path)?)?) == text(fixture, "sha256")?,
                    format!("baseline or candidate fixture drift: {path}"),
                )?;
            }
        }
        let ecs = &field(row, "ecs");
        if !ecs.is_null() {
            require(
                kind == "agent",
                "non-agent scenario receives migration credit",
            )?;
            mapped += 1;
            let path = checked(candidate, text(ecs, "source")?)?;
            let key = (text(ecs, "binary")?, text(ecs, "test")?);
            require(counterparts.insert(key), "duplicate ECS counterpart")?;
            let actual = compiled
                .get(&key)
                .ok_or("ECS counterpart absent from compiled discovery")?;
            require(
                field(ecs, "source") == field(actual, "source"),
                "ECS source mapping mismatch",
            )?;
            require(
                hash(&fs::read(path)?) == text(actual, "source_sha256")?,
                "candidate source changed since discovery",
            )?;
        }
        validate_assertions(row, original, discovery, field(manifest, "sources"))?;
        let mut reasons = Vec::new();
        if kind == "unclassified" {
            reasons.push("classification");
        }
        if field(row, "inventory_complete") != true {
            reasons.push("behavior/assertion inventory");
        }
        if field(row, "fixture_inventory_complete") != true {
            reasons.push("fixture inventory");
        }
        if kind == "agent" {
            if ecs.is_null() {
                reasons.push("native counterpart");
            }
            if array(row, "assertion_mappings")?.is_empty() {
                reasons.push("assertion mapping");
            }
        }
        if !reasons.is_empty() {
            unresolved.push(json!({"id":id, "missing":reasons}));
        }
    }
    require(
        seen == originals.keys().cloned().collect(),
        "missing manifest scenarios",
    )?;
    let gaps = json!({"unresolved_compiled":array(discovery, "unresolved_compiled_tests")?.len(), "unlisted_targets":array(discovery, "unlisted_provider_targets")?.len(), "source_only":originals.values().filter(|r| r["compiled"] != true).count()});
    let complete = unresolved.is_empty()
        && gaps
            .as_object()
            .ok_or("missing gaps")?
            .values()
            .all(|v| v == 0);
    Ok(
        json!({"baseline_revision":field(manifest, "baseline_revision"), "scope":field(manifest, "scope"),
        "counts":{"scenarios":seen.len(), "classifications":classifications, "native_mappings":mapped, "classified_agent_scopes":agent_scopes, "unresolved_scenarios":unresolved.len()},
        "discovery_gaps":gaps, "inventory_complete":complete, "unresolved":unresolved}),
    )
}

fn markdown(report: &Value, manifest: &Value) -> Result<String> {
    let mut output = format!(
        "# Provider baseline inventory\n\nBaseline: `{}`.\n\n{}\n\nGenerated by `cargo xtask parity-manifest report`. This reports source inventory and compiled mappings, not executed parity.\n\n{} scenarios; {} native mappings; {} scenarios with incomplete inventory.\n\n| Classification | Count |\n| --- | ---: |\n",
        text(report, "baseline_revision")?,
        text(report, "scope")?,
        field(field(report, "counts"), "scenarios"),
        field(field(report, "counts"), "native_mappings"),
        field(field(report, "counts"), "unresolved_scenarios")
    );
    for (kind, count) in field(field(report, "counts"), "classifications")
        .as_object()
        .ok_or("missing counts")?
    {
        output += &format!("| {kind} | {count} |\n");
    }
    output += "\nAgent scope counts are provisional: unclassified rows may add further obligations.\n\n| Classified agent scope | Count |\n| --- | ---: |\n";
    for (kind, count) in field(field(report, "counts"), "classified_agent_scopes")
        .as_object()
        .ok_or("missing scopes")?
    {
        output += &format!("| {kind} | {count} |\n");
    }
    output += "\n| Scenario | Classification | ECS counterpart | Inventory gaps |\n| --- | --- | --- | --- |\n";
    let gaps: BTreeMap<_, _> = array(report, "unresolved")?
        .iter()
        .map(|r| Ok((text(r, "id")?, strings(r, "missing")?.join(", "))))
        .collect::<Result<_>>()?;
    for row in array(manifest, "scenarios")? {
        let id = text(row, "id")?;
        output += &format!(
            "| `{id}` | {} | `{}` | {} |\n",
            text(row, "classification")?,
            field(field(row, "ecs"), "test").as_str().unwrap_or("—"),
            gaps.get(id)
                .map(String::as_str)
                .unwrap_or("none in this scoped inventory")
        );
    }
    Ok(output)
}

pub(crate) fn run(args: Vec<String>) -> Result<()> {
    let (action, options) = args
        .split_first()
        .ok_or("parity-manifest requires init or report")?;
    let mut flags = BTreeMap::new();
    for pair in options.chunks(2) {
        let [key, value] = pair else {
            return Err("manifest options require values".into());
        };
        require(
            flags.insert(key.as_str(), PathBuf::from(value)).is_none(),
            "duplicate option",
        )?;
    }
    let path = |key| {
        flags
            .get(key)
            .ok_or_else(|| format!("missing option {key}"))
    };
    let discovery = read(path("--discovery")?)?;
    let evidence = read(path("--configuration-evidence")?)?;
    let manifest_path = path("--manifest")?;
    if action == "init" {
        let initialized = initialize(&discovery, &evidence)?;
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(manifest_path)?
            .write_all(&bytes(&initialized)?)?;
        return Ok(());
    }
    require(action == "report", "expected init or report")?;
    let manifest = read(manifest_path)?;
    let report = inspect(
        &manifest,
        &discovery,
        path("--baseline-root")?,
        path("--candidate-root")?,
        &read(path("--candidate-discovery")?)?,
        &evidence,
    )?;
    write(path("--output")?, &report)?;
    if let Some(path) = flags.get("--markdown-output") {
        fs::write(path, markdown(&report, &manifest)?)?;
    }
    println!("{}", field(&report, "counts"));
    println!(
        "{}",
        if field(&report, "inventory_complete") == true {
            "Scoped inventory complete; execution coverage is separate"
        } else {
            "Inventory incomplete"
        }
    );
    Ok(())
}
