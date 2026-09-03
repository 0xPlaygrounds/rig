//! Every golden effect log in the corpus (`crates/rig-verify/fixtures/
//! *.effects.json`) is paired with exactly one producer — the root test
//! that records it under `RIG_REGENERATE_GOLDEN=1` by naming it in a
//! `golden_effects("<name>", ..)` call — and every producer names a golden
//! that is committed. A golden nobody can regenerate is a golden that can
//! only be hand-edited; a producer whose golden is missing is a test that
//! only ever fails.

use std::collections::BTreeMap;
use std::path::Path;

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The committed goldens, by name.
fn fixtures() -> Vec<String> {
    let dir = root().join("crates/rig-verify/fixtures");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("the corpus directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter_map(|name| name.strip_suffix(".effects.json").map(str::to_owned))
        .collect();
    names.sort();
    names
}

/// Every `golden_effects("<name>", ..)` site under `tests/`, by name, as
/// `file:line` locations.
fn producers() -> BTreeMap<String, Vec<String>> {
    let mut sites: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut pending = vec![root().join("tests")];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("a tests directory") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().is_some_and(|ext| ext == "rs")
                && path.file_name().is_some_and(|name| name != "goldens.rs")
            {
                let text = std::fs::read_to_string(&path).expect("source");
                for (number, line) in text.lines().enumerate() {
                    // Code lines only: a comment may quote the call.
                    if line.trim_start().starts_with("//") {
                        continue;
                    }
                    let Some(rest) = line.split("golden_effects(\"").nth(1) else {
                        continue;
                    };
                    let Some((name, _)) = rest.split_once('"') else {
                        continue;
                    };
                    let relative = path.strip_prefix(root()).expect("under the root");
                    sites.entry(name.to_owned()).or_default().push(format!(
                        "{}:{}",
                        relative.display(),
                        number + 1
                    ));
                }
            }
        }
    }
    sites
}

#[test]
fn every_golden_has_exactly_one_producer_and_every_producer_a_golden() {
    let fixtures = fixtures();
    let producers = producers();
    assert!(!fixtures.is_empty(), "the corpus is not empty");
    let mut problems = Vec::new();
    for name in &fixtures {
        match producers.get(name).map(Vec::as_slice) {
            Some([_]) => {}
            Some(sites) => problems.push(format!(
                "golden `{name}` is named by {} producers: {}",
                sites.len(),
                sites.join(", ")
            )),
            None => problems.push(format!(
                "golden `{name}` has no producer: no root test names it in `golden_effects`"
            )),
        }
    }
    for (name, sites) in &producers {
        if !fixtures.contains(name) {
            problems.push(format!(
                "{} names golden `{name}`, which is not committed under crates/rig-verify/fixtures",
                sites.join(", ")
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "goldens and producers are not paired:\n{}",
        problems.join("\n")
    );
}
