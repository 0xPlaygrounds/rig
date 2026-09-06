//! Reviewed mappings for macro-generated registrations. Source hashes make
//! these annotations fail closed when the macro or fixture changes; the
//! compiled listing independently establishes whether each test exists.

use super::*;

pub(super) fn load(root: &Path, path: Option<&Path>) -> Result<Value, String> {
    let Some(path) = path else {
        return Ok(json!({"suites": []}));
    };
    let document: Value = serde_json::from_str(&read(path)?).map_err(|e| e.to_string())?;
    if document.get("schema") != Some(&json!(1)) {
        return Err("unsupported generated-registration schema".into());
    }
    for suite in document
        .get("suites")
        .and_then(Value::as_array)
        .ok_or("missing generated suites")?
    {
        let dependencies = suite["source_dependencies"]
            .as_array()
            .ok_or("missing source dependencies")?;
        for field in [
            "binary",
            "module",
            "macro",
            "classification",
            "classification_reason",
            "provenance",
        ] {
            string(suite, field)?;
        }
        let invocation = string(suite, "invocation_source")?;
        let definition = string(suite, "macro_definition_source")?;
        if !dependencies.iter().any(|d| d["path"] == definition) {
            return Err(format!("unhashed macro definition {definition}"));
        }
        let cases = suite["cases"]
            .as_array()
            .filter(|c| !c.is_empty())
            .ok_or("missing nonempty generated cases")?;
        for case in cases {
            string(case, "name")?;
            string(case, "skip_obligation")?;
            let requirements = case["capability_requirements"]
                .as_array()
                .ok_or("missing capability requirements")?;
            if requirements
                .iter()
                .any(|r| r.as_str().is_none_or(str::is_empty))
            {
                return Err("invalid capability requirement".into());
            }
            let assertions = case["assertion_sources"]
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or("missing nonempty assertion sources")?;
            for assertion in assertions {
                let source = string(assertion, "path")?;
                string(assertion, "symbol")?;
                if !dependencies.iter().any(|d| d["path"] == source) {
                    return Err(format!("unhashed assertion source {source}"));
                }
            }
        }
        if !dependencies.iter().any(|d| d["path"] == invocation) {
            return Err(format!("unhashed generated invocation {invocation}"));
        }
        for dependency in dependencies {
            let relative = string(dependency, "path")?;
            let source = root
                .join(relative)
                .canonicalize()
                .map_err(|e| e.to_string())?;
            if !source.starts_with(root) {
                return Err(format!("generated source escapes root: {relative}"));
            }
            let digest = format!("{:x}", Sha256::digest(read(&source)?.as_bytes()));
            if dependency["sha256"] != digest {
                return Err(format!(
                    "generated registration source changed: {relative}; review mapping before reuse"
                ));
            }
        }
    }
    Ok(document)
}

pub(super) fn apply(
    binary: &str,
    macros: &[Value],
    found: &mut BTreeMap<String, Value>,
    document: &Value,
) -> Result<(), String> {
    for suite in document
        .get("suites")
        .and_then(Value::as_array)
        .ok_or("missing generated suites")?
    {
        if suite["binary"] != binary {
            continue;
        }
        let module = string(suite, "module")?;
        let source = string(suite, "invocation_source")?;
        let expected_macro = string(suite, "macro")?;
        let invocations: Vec<_> = macros
            .iter()
            .filter(|m| {
                m["binary"] == binary
                    && m["module"] == module
                    && m["source"] == source
                    && m["macro"]
                        .as_str()
                        .is_some_and(|p| p.split_whitespace().collect::<String>() == expected_macro)
            })
            .collect();
        if invocations.len() != 1 {
            return Err(format!(
                "generated suite {binary}::{module} needs exactly one matching source invocation"
            ));
        }
        let invocation = invocations.first().ok_or("missing invocation")?;
        let cases = suite["cases"].as_array().ok_or("missing generated cases")?;
        if cases.is_empty() {
            return Err("empty generated suite mapping".into());
        }
        for case in cases {
            let name = qualify(module, string(case, "name")?);
            let evidence = json!({
                "source": source, "source_sha256": invocation["source_sha256"],
                "line": invocation["line"], "module": module,
                "attributes": invocation["attributes"],
                "registration_kind": "reviewed_macro_mapping",
                "registration_sources": suite["source_dependencies"],
                "macro": expected_macro, "invocation_tokens": invocation["tokens"],
                "classification": suite["classification"],
                "classification_reason": suite["classification_reason"],
                "provenance": suite["provenance"],
                "configuration_obligations": suite["configuration_obligations"],
                "assertion_sources": case["assertion_sources"],
                "capability_requirements": case["capability_requirements"],
                "skip_obligation": case["skip_obligation"],
                "semantic_outcome": null,
            });
            if found.insert(name.clone(), evidence).is_some() {
                return Err(format!(
                    "duplicate generated/source registration {binary}::{name}"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing nonempty {key}"))
}
