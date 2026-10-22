//! The golden of what every completion wire sends, and the corpus it is
//! encoded over.

use std::collections::BTreeSet;

use rig::catalog::Catalog;

use super::*;

#[test]
fn default_user_agent_fixture_is_independent_of_host_and_keeps_the_caller() {
    for (os, arch) in [
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("macos", "x86_64"),
        ("macos", "aarch64"),
    ] {
        let value = format!(
            "rig/{} ({os} {arch}; custom-originator) extra/1",
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(
            fixture_user_agent(&value, os, arch),
            "rig/<version> (linux x86_64; custom-originator) extra/1"
        );
    }
    for value in [
        format!(
            "custom/1 rig/{} (macos aarch64; rig)",
            env!("CARGO_PKG_VERSION")
        ),
        format!("rig/{} custom-agent", env!("CARGO_PKG_VERSION")),
        format!("rig/{} (linux aarch64; rig)", env!("CARGO_PKG_VERSION")),
        format!("rig/{} (macos x86_64; rig)", env!("CARGO_PKG_VERSION")),
        "rig/other-version (macos aarch64; rig)".to_owned(),
        "GitHubCopilotChat/0.35.0".to_owned(),
    ] {
        assert_eq!(fixture_user_agent(&value, "macos", "aarch64"), value);
    }
}

#[test]
fn encoded_default_user_agent_reports_the_actual_host_before_fixture_normalization()
-> anyhow::Result<()> {
    use openai::responses_api::wire::Responses;
    use rig::providers::{chatgpt, openai};

    let wire = Responses::new(
        openai::wire::OpenAIConfig::with_key(&chatgpt::DIALECT, "synthetic-key"),
        chatgpt::GPT_5_4,
    );
    let prepared = Completion::prepare(CompletionRequest::new("hi"), &wire.describe())?;
    let mut encoded = wire.encode(prepared, Mode::Unary)?;
    let expected = format!(
        "rig/{} ({} {}; rig)",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    anyhow::ensure!(
        encoded
            .request
            .headers()
            .get("user-agent")
            .and_then(|value| value.to_str().ok())
            == Some(expected.as_str())
    );
    anyhow::ensure!(
        http(&encoded)
            .pointer("/headers/user-agent")
            .and_then(Value::as_str)
            == Some("rig/<version> (linux x86_64; rig)")
    );

    let custom = format!("custom/1 rig/{} unchanged", env!("CARGO_PKG_VERSION"));
    encoded
        .request
        .headers_mut()
        .insert("user-agent", custom.parse()?);
    encoded
        .request
        .headers_mut()
        .insert("x-custom", expected.parse()?);
    let captured = http(&encoded);
    anyhow::ensure!(
        captured
            .pointer("/headers/user-agent")
            .and_then(Value::as_str)
            == Some(custom.as_str())
    );
    anyhow::ensure!(
        captured
            .pointer("/headers/x-custom")
            .and_then(Value::as_str)
            == Some(expected.as_str())
    );
    Ok(())
}

/// Every model id the catalog lists is in the corpus, so a new catalog row
/// is pinned too. Regenerating adds the missing ones.
#[test]
fn the_corpus_holds_every_catalog_id() {
    let corpus: BTreeSet<String> = corpus().into_iter().collect();
    let missing: Vec<&str> = Catalog::builtin()
        .iter()
        .map(|spec| spec.id.as_str())
        .filter(|id| !corpus.contains(*id))
        .collect();
    assert!(
        missing.is_empty() || std::env::var_os(REGENERATE).is_some(),
        "{} catalog ids are not in tests/fixtures/request_bodies/corpus.txt; run with \
         {REGENERATE}=1: {missing:?}",
        missing.len()
    );
}

/// What each wire sends for each model id and shape is what the golden
/// pins; a change fails here until it is listed in `TYPED_OPTIONS.md`
/// section 12.0 and the golden is regenerated.
#[test]
fn every_wire_sends_the_pinned_bodies() {
    let golden = fixture("bodies.txt");
    if std::env::var_os(REGENERATE).is_some() {
        assert!(
            every_wire_built(),
            "regenerate with --all-features, so every companion wire is encoded"
        );
        let mut models: BTreeSet<String> = corpus().into_iter().collect();
        models.extend(Catalog::builtin().iter().map(|spec| spec.id.clone()));
        let models: Vec<String> = models.into_iter().collect();
        let mut listing = models.join("\n");
        listing.push('\n');
        std::fs::write(fixture("corpus.txt"), listing).expect("the corpus is writable");
        let outputs = encode_all(&models, &encoders());
        std::fs::write(&golden, render(&outputs, &models)).expect("the golden is writable");
        return;
    }
    let text = std::fs::read_to_string(&golden).unwrap_or_else(|error| {
        panic!(
            "{} is unreadable ({error}); run with {REGENERATE}=1",
            golden.display()
        )
    });
    let (shape_names, expected) = parse(&text);
    let names: Vec<&str> = shapes().into_iter().map(|(name, _)| name).collect();
    assert_eq!(
        shape_names, names,
        "the shapes changed; run with {REGENERATE}=1"
    );
    let models = corpus();
    let encoders = encoders();
    assert!(
        encoders
            .iter()
            .all(|encoder| expected.contains_key(&encoder.key)),
        "a wire the golden does not hold; run with {REGENERATE}=1"
    );
    assert!(
        !every_wire_built() || encoders.len() == expected.len(),
        "the golden holds a wire this build no longer has; run with {REGENERATE}=1"
    );
    let actual = encode_all(&models, &encoders);
    let mut changed = Vec::new();
    let mut count = 0;
    for (wire, by_model) in &actual {
        for (model, row) in by_model {
            let pinned = expected.get(wire).and_then(|pinned| pinned.get(model));
            for (index, output) in row.iter().enumerate() {
                let before = pinned.and_then(|row| row.get(index));
                if before != Some(output) {
                    count += 1;
                    if changed.len() < 20 {
                        changed.push(format!(
                            "{wire} {model} {}:\n  pinned: {}\n  now:    {output}",
                            names.get(index).copied().unwrap_or("?"),
                            before.map_or("(none)", String::as_str),
                        ));
                    }
                }
            }
        }
    }
    assert!(
        count == 0,
        "{count} request bodies changed. List each change in \
         crates/rig-core/TYPED_OPTIONS.md section 12.0 (with its Migration line), then run \
         with {REGENERATE}=1 --all-features. The first {}:\n{}",
        changed.len(),
        changed.join("\n")
    );
}
