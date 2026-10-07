//! The golden of what every completion wire sends, and the corpus it is
//! encoded over.

use std::collections::BTreeSet;

use rig::catalog::Catalog;

use super::*;

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
