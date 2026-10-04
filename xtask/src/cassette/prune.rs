//! `cargo xtask cassette prune [--check]`: delete the cassette tests, their
//! fixtures and the effect goldens whose coverage the kept tests already
//! give, by a fixed rule, and list each deletion in
//! `crates/rig-cassette/coverage/pruned.tsv`.
//!
//! The candidates are the tests of the provider targets that record,
//! replay or read a fixture, or name an effect golden. Every other test is
//! kept: the crates' unit tests, the conformance targets, the `runtime`
//! target (which replays the reply bank and is preferred over a provider's
//! copy of a runtime scenario), `verify` and `world_replay`. The selection
//! keeps the fewest candidates whose union, with the always-kept tests,
//! holds everything the coverage gate measures; see [`select`] and the
//! manifest's preamble for the rule.
//!
//! A rewrite needs `target/coverage/per-test.tsv` from
//! `cargo xtask coverage --per-test`. It deletes the fixtures and goldens it
//! selects and lists the tests to take out of the source; the rows of
//! earlier deletions carry forward. `--check` reruns the selection over the
//! tree and fails when it would delete more, when a listed item is still
//! there, or when the manifest is not the one a rewrite would write.

mod edit;
mod names;
mod select;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::BufRead as _;
use std::path::Path;

use serde_json::Value;

use self::names::{Corpus, Golden, Named, TestId};
use crate::cassette::acceptance;
use crate::coverage::lines::{self, Branch};
use crate::coverage::shapes::{self, ShapeKey};

const CASSETTES: &str = "crates/rig-cassette/fixtures/cassettes";
/// The manifest, relative to the workspace root.
pub(crate) const MANIFEST: &str = "crates/rig-cassette/coverage/pruned.tsv";
const LINES: &str = "crates/rig-cassette/coverage/lines.tsv";
const SHAPES: &str = "crates/rig-cassette/coverage/shapes.tsv";
/// The fixtures and goldens kept by hand, each with its reason.
pub(crate) const KEEP: &str = "crates/rig-cassette/coverage/prune-keep.txt";
const PARITY: &str = "crates/rig-cassette/fixtures/parity";

/// Tests whose coverage depends on which fixtures or goldens exist: they
/// sweep a corpus directory. They stay, but what they cover is not
/// credited, so a region only they reach keeps a test of its own. Each
/// entry is a binary (`*` for any) and a test-name prefix (empty for all).
const SWEEPS: &[(&str, &str)] = &[
    ("rig-cassette::cassette_cache_prefix", ""),
    ("rig-cassette::cassette_history_survival", ""),
    ("rig-cassette::cassette_usage_census", ""),
    ("rig-cassette::chat_parity", ""),
    ("rig-cassette::world_replay", ""),
    ("rig-cassette::world_replay_world", ""),
    ("rig-cassette::verify", "corpus_oracle::"),
    (
        "rig-cassette::verify",
        "corpus_checkpoint::hash_mode_accepts_every_golden",
    ),
    ("*", "cassette_safety::"),
    (
        "rig-cassette::anthropic",
        "anthropic::cassette::restated_history::",
    ),
    (
        "rig-cassette::gemini",
        "gemini::cassette::restated_replies::",
    ),
    (
        "rig-cassette::llamacpp",
        "llamacpp::cassette::response_shape_matrix::the_finish_reason_vocabulary_is_covered_end_to_end",
    ),
    (
        "rig-core",
        "providers::openai::responses_api::streaming::tests::every_recorded_whole_reply_agrees_with_its_restatement_as_a_stream",
    ),
];

const PREAMBLE: &str = "\
# The cassette prune, written by `cargo xtask cassette prune`; review the rule
# and this list, not the deleted files. Rerunning the command gives the same
# list, and `--check` fails when the tree or this list disagrees with it.
#
# The rule. The candidates are the tests of the provider targets that record,
# replay or read a cassette, or name an effect golden. Every other test is kept
# (crate unit tests, conformance targets, the runtime target over the reply
# bank, verify, world_replay, and every test of a provider file another target
# compiles too), and so is every fixture or golden something outside the
# candidates names or crates/rig-cassette/coverage/prune-keep.txt lists with
# its reason. The kept candidates are chosen by greedy set
# cover over what the gate measures that the always-kept tests do not hold:
# the candidate covering the most such elements is taken, then the one with
# smaller fixtures, then the alphabetically first; then every taken test the
# others make redundant is dropped, latest first. A kept fixture keeps one of
# its recording tests. A fixture goes when every test naming it goes, with its
# `.requests.json` and `.clock.json`.
#
# The elements. Regions: every line and branch of crates/rig-cassette/coverage/
# lines.tsv, from each test's own coverage (`cargo xtask coverage --per-test`).
# The corpus sweeps (cassette_history_survival, cassette_usage_census,
# cassette_cache_prefix, chat_parity, world_replay, world_replay_world, the
# cassette-safety scans and the restatement sweeps) stay but are not credited,
# since what they cover depends on the files that exist. Shapes: every request
# fact and reply shape of crates/rig-cassette/coverage/shapes.tsv. Acceptance:
# every fact of fixtures/acceptance.toml keeps a live recording. The reply bank
# carries an entry forward when its source fixture goes. Mutants: the mutation
# baseline is measured against the crate unit tests and conformance targets
# only, never a cassette test or a golden, so no deletion here can lose a kill.
#
# Goldens. A golden goes with its producer. A kept producer's golden stays only
# when something other than its producer reads it, or it is the smallest golden
# of its corpus (agent or world) holding an effect kind no read golden holds:
# the one format pin for that kind. Every other kept producer checks its log in
# the test instead: the log replays record by record through a world, and a
# world log's programs restore and replay by id.
#
# Parity snapshots follow the same rule: fixtures/parity/<provider>.json pins
# one reply per reply shape and mode (the smallest, then the first in path
# order), and every other reply is checked in the test, its `call` and
# `stream().finish()` folding to the same response. A `parity` row is an entry
# so replaced; the entries of a deleted fixture go with the fixture.
#
# Columns: what was deleted, and the kept tests that cover what it covered
# beyond the always-kept tests (`-` when those cover all of it). A golden names
# its producer: `with` when the producer went too, `in-test` when it checks in
# the test now.
kind\titem\tcovered by
";

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let check = match args {
        [] => false,
        [flag] if flag == "--check" => true,
        _ => return Err("usage: cassette prune [--check]".into()),
    };
    let model = Model::load(root)?;
    let selection = select::select(&model.problem);
    let outcome = model.outcome(&selection);
    let path = root.join(MANIFEST);
    let committed = match std::fs::read_to_string(&path) {
        Ok(text) => parse(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let fresh = outcome.rows(&model, &selection);
    let mut rows: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut stale = Vec::new();
    for row in &committed {
        if model.exists(root, &row.kind, &row.item) {
            stale.push(format!(
                "{} {} is listed but still in the tree",
                row.kind, row.item
            ));
        } else {
            rows.insert((row.kind.clone(), row.item.clone()), row.covered.clone());
        }
    }
    for row in &fresh {
        rows.insert((row.kind.clone(), row.item.clone()), row.covered.clone());
    }
    let text = render(&rows);
    println!(
        "prune: {} candidate tests, {} kept, {} deleted; {} fixtures deleted, {} goldens deleted ({} replaced by in-test checks); {} elements, {} only sweeps reach",
        model.candidates.len(),
        selection.kept.len(),
        outcome.tests.len(),
        outcome.fixtures.len(),
        outcome.goldens.len(),
        outcome
            .goldens
            .values()
            .filter(|golden| golden.in_test)
            .count(),
        model.problem.elements,
        model.uncoverable
    );
    if check {
        let mut failures = Vec::new();
        for row in &fresh {
            failures.push(format!(
                "prune would delete {} {}; run `cargo xtask cassette prune`",
                row.kind, row.item
            ));
        }
        failures.extend(stale);
        if failures.is_empty() && std::fs::read_to_string(&path).ok().as_deref() != Some(&text) {
            failures.push(format!(
                "{MANIFEST} is not the manifest the tree gives; rewrite it with \
                 `cargo xtask cassette prune`"
            ));
        }
        return if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} prune failure(s):\n  {}",
                failures.len(),
                failures.join("\n  ")
            ))
        };
    }
    std::fs::write(&path, &text).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("wrote {}", path.display());
    let mut removed = 0;
    for file in outcome.files() {
        let full = root.join(&file);
        if full.is_file() {
            std::fs::remove_file(&full).map_err(|error| format!("{}: {error}", full.display()))?;
            removed += 1;
        }
    }
    println!("removed {removed} files");
    let mut by_file: BTreeMap<(&str, &str), BTreeSet<String>> = BTreeMap::new();
    for test in &outcome.tests {
        if let Some(what) = model.named_of(test) {
            by_file
                .entry((what.file.as_str(), what.module.as_str()))
                .or_default()
                .insert(test.1.clone());
        }
    }
    let mut taken = 0;
    for ((file, module), deleted) in by_file {
        let full = root.join(file);
        let source = std::fs::read_to_string(&full).map_err(|error| format!("{file}: {error}"))?;
        let (edited, removed) = edit::remove_tests(&source, module, &deleted)
            .map_err(|error| format!("{file}: {error}"))?;
        if removed != deleted.len() {
            println!("{file}: took out {removed} of {} tests", deleted.len());
        }
        taken += removed;
        std::fs::write(&full, edited).map_err(|error| format!("{file}: {error}"))?;
    }
    println!("took {taken} tests out of the source");
    Ok(())
}

/// One manifest row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) kind: String,
    pub(crate) item: String,
    pub(crate) covered: String,
}

/// Parse a manifest written by [`render`].
pub(crate) fn parse(text: &str) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    let mut header = false;
    for (number, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if !header {
            header = true;
            continue;
        }
        let mut columns = line.split('\t');
        match (
            columns.next(),
            columns.next(),
            columns.next(),
            columns.next(),
        ) {
            (Some(kind), Some(item), Some(covered), None) => rows.push(Row {
                kind: kind.to_owned(),
                item: item.to_owned(),
                covered: covered.to_owned(),
            }),
            _ => return Err(format!("{MANIFEST} line {}: {line:?}", number + 1)),
        }
    }
    Ok(rows)
}

/// The manifest's text: the preamble, then one row per deletion in kind
/// and item order.
pub(crate) fn render(rows: &BTreeMap<(String, String), String>) -> String {
    let mut out = PREAMBLE.to_owned();
    for ((kind, item), covered) in rows {
        let _ = writeln!(out, "{kind}\t{item}\t{covered}");
    }
    out
}

/// Whether a binary and test are a corpus sweep.
fn is_sweep(binary: &str, test: &str) -> bool {
    SWEEPS.iter().any(|(sweep_binary, prefix)| {
        let binary_matches = *sweep_binary == binary
            || (*sweep_binary == "*" && binary.starts_with("rig-cassette::"));
        let test_matches = if sweep_binary == &"*" {
            test.contains(prefix)
        } else {
            test.starts_with(prefix)
        };
        binary_matches && test_matches
    })
}

/// Whether a provider target's own test is gone from the source. Tests the
/// target compiles from shared modules are not the scan's.
fn is_stale((binary, test): &TestId, scanned: &BTreeSet<TestId>) -> bool {
    binary
        .strip_prefix("rig-cassette::")
        .is_some_and(|provider| {
            test.starts_with(&format!("{provider}::"))
                && !scanned.contains(&(binary.clone(), test.clone()))
        })
}

/// A covered region of the line baseline.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Region {
    Line(String, u32),
    Branch(String, Branch),
}

/// Everything the selection reads, indexed.
struct Model {
    /// The candidates, in name order: a candidate's index is its id.
    candidates: Vec<TestId>,
    problem: select::Problem,
    /// Every fixture, by id.
    fixtures: Vec<String>,
    fixture_ids: BTreeMap<String, usize>,
    /// What each candidate names.
    named: Vec<Named>,
    /// The goldens, and every effect kind each holds.
    goldens: BTreeMap<Golden, (u64, BTreeSet<String>)>,
    /// Goldens something other than a candidate names.
    protected_goldens: BTreeSet<Golden>,
    /// Each fixture's elements (shapes and acceptance facts), sorted.
    fixture_elements: Vec<Vec<usize>>,
    /// Elements only the sweeps reach.
    uncoverable: usize,
    /// Every test the scan finds in the provider sources.
    scanned: BTreeSet<TestId>,
}

impl Model {
    fn load(root: &Path) -> Result<Self, String> {
        let cassettes = root.join(CASSETTES);
        let corpus = shapes::corpus(&cassettes).map_err(|error| error.to_string())?;
        let goldens = read_goldens(root)?;
        let names = names::scan(
            root,
            &Corpus::new(
                corpus
                    .iter()
                    .map(|fixture| fixture.relative.clone())
                    .collect(),
                goldens.keys().cloned().collect(),
            ),
        )?;
        let eligible: BTreeSet<TestId> = names
            .tests
            .iter()
            .filter(|((binary, test), what)| {
                let names_something =
                    !what.owns.is_empty() || !what.reads.is_empty() || !what.goldens.is_empty();
                names_something
                    && !is_sweep(binary, test)
                    && !names.shared_files.contains(&what.file)
            })
            .map(|(test, _)| test.clone())
            .collect();
        let scanned: BTreeSet<TestId> = names.tests.keys().cloned().collect();
        let per_test = read_per_test(root, &eligible, &scanned)?;

        let fixtures: Vec<String> = corpus
            .iter()
            .map(|fixture| fixture.relative.clone())
            .collect();
        let fixture_ids: BTreeMap<String, usize> = fixtures
            .iter()
            .enumerate()
            .map(|(id, fixture)| (fixture.clone(), id))
            .collect();

        // A named test that never ran (ignored) is no candidate; what it
        // names stays.
        let mut candidates = Vec::new();
        let mut named = Vec::new();
        let mut protected_fixtures: BTreeSet<String> = names.fixtures.keys().cloned().collect();
        let mut protected_goldens: BTreeSet<Golden> = names.goldens.keys().cloned().collect();
        for item in read_keep(root)? {
            let golden = golden_of(&item);
            if fixture_ids.contains_key(&item) {
                protected_fixtures.insert(item);
            } else if goldens.contains_key(&golden) {
                protected_goldens.insert(golden);
            } else {
                return Err(format!(
                    "{KEEP} keeps {item}, which is neither a fixture nor a golden"
                ));
            }
        }
        for (test, what) in &names.tests {
            if eligible.contains(test) && per_test.regions.contains_key(test) {
                candidates.push(test.clone());
                named.push(what.clone());
            } else {
                protected_fixtures.extend(what.owns.iter().cloned());
                protected_fixtures.extend(what.reads.iter().cloned());
                protected_fixtures.extend(what.file_reads.iter().cloned());
                protected_goldens.extend(what.goldens.iter().cloned());
            }
        }
        // A fixture or golden no candidate names is not this command's.
        let candidate_fixtures: BTreeSet<&String> = named
            .iter()
            .flat_map(|what| what.owns.iter().chain(&what.reads).chain(&what.file_reads))
            .collect();
        for fixture in &fixtures {
            if !candidate_fixtures.contains(fixture) {
                protected_fixtures.insert(fixture.clone());
            }
        }
        let candidate_goldens: BTreeSet<&Golden> =
            named.iter().flat_map(|what| what.goldens.iter()).collect();
        for golden in goldens.keys() {
            if !candidate_goldens.contains(golden) {
                protected_goldens.insert(golden.clone());
            }
        }

        // Elements: regions first, then shapes, then acceptance facts.
        let mut elements: BTreeMap<Element, usize> = BTreeMap::new();
        let mut candidate_elements: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); candidates.len()];
        let mut coverable: BTreeSet<Region> = BTreeSet::new();
        for (index, test) in candidates.iter().enumerate() {
            if let (Some(regions), Some(set)) = (
                per_test.regions.get(test),
                candidate_elements.get_mut(index),
            ) {
                for region in regions {
                    if per_test.forced.contains(region) {
                        continue;
                    }
                    coverable.insert(region.clone());
                    let next = elements.len();
                    let id = *elements
                        .entry(Element::Region(region.clone()))
                        .or_insert(next);
                    set.insert(id);
                }
            }
        }
        let uncovered: Vec<&Region> = per_test
            .baseline
            .iter()
            .filter(|region| !per_test.forced.contains(*region) && !coverable.contains(*region))
            .collect();
        let uncoverable = uncovered.len();

        let shape_baseline: BTreeSet<ShapeKey> = shapes::parse(&read(root, SHAPES)?)
            .map_err(|error| error.to_string())?
            .into_keys()
            .collect();
        let index_text = read(root, acceptance::INDEX)?;
        let index = acceptance::parse(&index_text)?;
        let accepted: BTreeSet<acceptance::Cell> = index
            .facts
            .iter()
            .map(|entry| {
                (
                    entry.provider.clone(),
                    entry.encoder.clone(),
                    entry.fact.clone(),
                )
            })
            .collect();
        let observed = acceptance::observe_fixtures(&cassettes, &corpus)?;
        let cells = acceptance::recorded_cells(&observed, &index);

        let mut fixture_keys: Vec<BTreeSet<Element>> = vec![BTreeSet::new(); fixtures.len()];
        for (id, fixture) in corpus.iter().enumerate() {
            if let Some(keys) = fixture_keys.get_mut(id) {
                for (key, _) in shapes::fixture_shapes(fixture) {
                    if shape_baseline.contains(&key) {
                        keys.insert(Element::Shape(key));
                    }
                }
                for cell in cells.get(&fixture.relative).into_iter().flatten() {
                    if accepted.contains(cell) {
                        keys.insert(Element::Fact(cell.clone()));
                    }
                }
            }
        }
        let protected: BTreeSet<usize> = protected_fixtures
            .iter()
            .filter_map(|fixture| fixture_ids.get(fixture).copied())
            .collect();
        let held: BTreeSet<&Element> = protected
            .iter()
            .filter_map(|id| fixture_keys.get(*id))
            .flatten()
            .collect();
        let mut fixture_elements: Vec<Vec<usize>> = vec![Vec::new(); fixtures.len()];
        for (id, keys) in fixture_keys.iter().enumerate() {
            let mut ids = Vec::new();
            for key in keys {
                if held.contains(key) {
                    continue;
                }
                let next = elements.len();
                ids.push(*elements.entry(key.clone()).or_insert(next));
            }
            ids.sort_unstable();
            if let Some(slot) = fixture_elements.get_mut(id) {
                *slot = ids;
            }
        }

        let sizes: BTreeMap<&str, u64> = corpus
            .iter()
            .map(|fixture| (fixture.relative.as_str(), fixture.size as u64))
            .collect();
        let mut problem = select::Problem {
            candidates: Vec::new(),
            elements: elements.len(),
            fixtures: fixtures.len(),
            protected,
            forced: BTreeSet::new(),
        };
        for (index, what) in named.iter().enumerate() {
            let mut set = candidate_elements.get(index).cloned().unwrap_or_default();
            let owns: Vec<usize> = what
                .owns
                .iter()
                .filter_map(|fixture| fixture_ids.get(fixture).copied())
                .collect();
            for fixture in &owns {
                set.extend(
                    fixture_elements
                        .get(*fixture)
                        .into_iter()
                        .flatten()
                        .copied(),
                );
            }
            let reads: Vec<usize> = what
                .reads
                .iter()
                .chain(&what.file_reads)
                .filter_map(|fixture| fixture_ids.get(fixture).copied())
                .filter(|fixture| !owns.contains(fixture))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let cost = what
                .owns
                .iter()
                .filter_map(|fixture| sizes.get(fixture.as_str()))
                .sum();
            if what
                .goldens
                .iter()
                .any(|golden| protected_goldens.contains(golden))
            {
                problem.forced.insert(index);
            }
            problem.candidates.push(select::Candidate {
                elements: set.into_iter().collect(),
                cost,
                owns,
                reads,
            });
        }
        Ok(Self {
            candidates,
            problem,
            fixtures,
            fixture_ids,
            named,
            goldens,
            protected_goldens,
            fixture_elements,
            uncoverable,
            scanned: names.tests.keys().cloned().collect(),
        })
    }

    fn named_of(&self, test: &TestId) -> Option<&Named> {
        self.candidates
            .iter()
            .position(|candidate| candidate == test)
            .and_then(|index| self.named.get(index))
    }

    /// Whether a manifest item is still in the tree.
    fn exists(&self, root: &Path, kind: &str, item: &str) -> bool {
        match kind {
            "test" => item.split_once(' ').is_some_and(|(binary, test)| {
                self.scanned.contains(&(binary.to_owned(), test.to_owned()))
            }),
            "fixture" => sidecars(item)
                .iter()
                .any(|file| root.join(CASSETTES).join(file).is_file()),
            "golden" => golden_of(item)
                .files()
                .iter()
                .any(|file| root.join(file).is_file()),
            "parity" => parity_entry(root, item),
            _ => false,
        }
    }

    fn outcome(&self, selection: &select::Selection) -> Outcome {
        let kept: BTreeSet<usize> = selection.kept.iter().copied().collect();
        let mut outcome = Outcome::default();
        for (index, test) in self.candidates.iter().enumerate() {
            if !kept.contains(&index) {
                outcome.tests.push(test.clone());
            }
        }
        let mut live = vec![false; self.fixtures.len()];
        for id in &self.problem.protected {
            if let Some(slot) = live.get_mut(*id) {
                *slot = true;
            }
        }
        for index in &kept {
            if let Some(candidate) = self.problem.candidates.get(*index) {
                for id in candidate.owns.iter().chain(&candidate.reads) {
                    if let Some(slot) = live.get_mut(*id) {
                        *slot = true;
                    }
                }
            }
        }
        for (id, fixture) in self.fixtures.iter().enumerate() {
            if !live.get(id).copied().unwrap_or(true) {
                outcome.fixtures.push(fixture.clone());
            }
        }
        // Goldens: a producer's goes with it; a kept producer's stays when
        // read or a format pin.
        let mut producers: BTreeMap<&Golden, Vec<usize>> = BTreeMap::new();
        for (index, what) in self.named.iter().enumerate() {
            for golden in &what.goldens {
                producers.entry(golden).or_default().push(index);
            }
        }
        let mut remaining: Vec<&Golden> = Vec::new();
        for (golden, indices) in &producers {
            if self.protected_goldens.contains(*golden) {
                continue;
            }
            match indices.iter().find(|index| kept.contains(index)) {
                Some(_) => remaining.push(golden),
                None => {
                    outcome.goldens.insert(
                        (*golden).clone(),
                        GoldenFate {
                            producer: indices.first().copied(),
                            in_test: false,
                        },
                    );
                }
            }
        }
        let mut pinned: BTreeSet<(bool, String)> = BTreeSet::new();
        for golden in &self.protected_goldens {
            if let Some((_, kinds)) = self.goldens.get(golden) {
                for kind in kinds {
                    pinned.insert((golden.world, kind.clone()));
                }
            }
        }
        let mut pins: BTreeMap<(bool, String), (u64, &Golden)> = BTreeMap::new();
        for golden in &remaining {
            if let Some((size, kinds)) = self.goldens.get(*golden) {
                for kind in kinds {
                    let key = (golden.world, kind.clone());
                    if pinned.contains(&key) {
                        continue;
                    }
                    let better = pins
                        .get(&key)
                        .is_none_or(|(best, name)| (*size, *golden) < (*best, *name));
                    if better {
                        pins.insert(key, (*size, golden));
                    }
                }
            }
        }
        let pins: BTreeSet<&Golden> = pins.values().map(|(_, golden)| *golden).collect();
        for golden in remaining {
            if pins.contains(golden) {
                continue;
            }
            let producer = producers
                .get(golden)
                .and_then(|indices| indices.iter().find(|index| kept.contains(index)).copied());
            outcome.goldens.insert(
                golden.clone(),
                GoldenFate {
                    producer,
                    in_test: true,
                },
            );
        }
        outcome
    }
}

/// What one golden became.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GoldenFate {
    producer: Option<usize>,
    in_test: bool,
}

/// What a selection deletes.
#[derive(Debug, Default)]
struct Outcome {
    tests: Vec<TestId>,
    fixtures: Vec<String>,
    goldens: BTreeMap<Golden, GoldenFate>,
}

impl Outcome {
    fn rows(&self, model: &Model, selection: &select::Selection) -> Vec<Row> {
        let mut kept = selection.kept.clone();
        kept.sort_unstable();
        let label = |index: usize| {
            model
                .candidates
                .get(index)
                .map(|(binary, test)| format!("{binary} {test}"))
                .unwrap_or_default()
        };
        let covered = |elements: &[usize]| {
            let by = select::cover_with(elements, &kept, &model.problem);
            if by.is_empty() {
                "-".to_owned()
            } else {
                by.into_iter().map(label).collect::<Vec<_>>().join(", ")
            }
        };
        let mut rows = Vec::new();
        let ids: BTreeMap<&TestId, usize> = model
            .candidates
            .iter()
            .enumerate()
            .map(|(index, test)| (test, index))
            .collect();
        for test in &self.tests {
            let elements = ids
                .get(test)
                .and_then(|index| model.problem.candidates.get(*index))
                .map(|candidate| candidate.elements.clone())
                .unwrap_or_default();
            rows.push(Row {
                kind: "test".into(),
                item: format!("{} {}", test.0, test.1),
                covered: covered(&elements),
            });
        }
        for fixture in &self.fixtures {
            let elements = model
                .fixture_ids
                .get(fixture)
                .and_then(|id| model.fixture_elements.get(*id))
                .cloned()
                .unwrap_or_default();
            rows.push(Row {
                kind: "fixture".into(),
                item: fixture.clone(),
                covered: covered(&elements),
            });
        }
        for (golden, fate) in &self.goldens {
            let producer = fate.producer.map(label).unwrap_or_default();
            rows.push(Row {
                kind: "golden".into(),
                item: golden.label(),
                covered: if fate.in_test {
                    format!("in-test {producer}")
                } else {
                    format!("with {producer}")
                },
            });
        }
        rows
    }

    /// Every file the deletions remove, relative to the workspace root.
    fn files(&self) -> Vec<String> {
        let mut files = Vec::new();
        for fixture in &self.fixtures {
            for file in sidecars(fixture) {
                files.push(format!("{CASSETTES}/{file}"));
            }
        }
        for golden in self.goldens.keys() {
            files.extend(golden.files());
        }
        files
    }
}

/// Whether a parity snapshot (`<provider>/<scenario>.yaml#<n>`) holds the
/// entry.
fn parity_entry(root: &Path, item: &str) -> bool {
    let Some((provider, key)) = item.split_once('/') else {
        return false;
    };
    let path = root.join(format!("{PARITY}/{provider}.json"));
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .is_some_and(|snapshot| snapshot.get(key).is_some())
}

/// A fixture and the files beside it that belong to it.
fn sidecars(fixture: &str) -> Vec<String> {
    let stem = fixture.strip_suffix(".yaml").unwrap_or(fixture);
    vec![
        fixture.to_owned(),
        format!("{stem}.requests.json"),
        format!("{stem}.clock.json"),
    ]
}

/// The golden a manifest label names.
fn golden_of(label: &str) -> Golden {
    match label.strip_prefix("world/") {
        Some(name) => Golden {
            world: true,
            name: name.to_owned(),
        },
        None => Golden {
            world: false,
            name: label.to_owned(),
        },
    }
}

/// One selection element.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Element {
    Region(Region),
    Shape(ShapeKey),
    Fact(acceptance::Cell),
}

/// The items of the keep list: the first word of every line that is not
/// blank or a comment; none when the file is absent.
fn read_keep(root: &Path) -> Result<Vec<String>, String> {
    match std::fs::read_to_string(root.join(KEEP)) {
        Ok(text) => Ok(parse_keep(&text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(format!("{KEEP}: {error}")),
    }
}

pub(crate) fn parse_keep(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

fn read(root: &Path, relative: &str) -> Result<String, String> {
    std::fs::read_to_string(root.join(relative)).map_err(|error| format!("{relative}: {error}"))
}

/// Every golden with its size and the effect kinds its records hold.
fn read_goldens(root: &Path) -> Result<BTreeMap<Golden, (u64, BTreeSet<String>)>, String> {
    let mut goldens = BTreeMap::new();
    for (world, dir) in [
        (false, root.join(names::EFFECTS)),
        (true, root.join(names::EFFECTS).join("world")),
    ] {
        let entries =
            std::fs::read_dir(&dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        let mut files = Vec::new();
        for entry in entries {
            files.push(
                entry
                    .map_err(|error| format!("{}: {error}", dir.display()))?
                    .path(),
            );
        }
        files.sort();
        for file in files {
            let Some(name) = file
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_suffix(".effects.json"))
            else {
                continue;
            };
            let text = std::fs::read_to_string(&file)
                .map_err(|error| format!("{}: {error}", file.display()))?;
            let value: Value = serde_json::from_str(&text)
                .map_err(|error| format!("{}: {error}", file.display()))?;
            let kinds = value
                .get("records")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|record| record.pointer("/kind/effect").and_then(Value::as_str))
                .map(str::to_owned)
                .collect();
            goldens.insert(
                Golden {
                    world,
                    name: name.to_owned(),
                },
                (text.len() as u64, kinds),
            );
        }
    }
    Ok(goldens)
}

/// The per-test coverage, as the selection reads it.
struct PerTest {
    /// Every region of the line baseline.
    baseline: BTreeSet<Region>,
    /// Regions a credited test outside the candidates covers.
    forced: BTreeSet<Region>,
    /// Baseline regions each named test covers.
    regions: BTreeMap<TestId, BTreeSet<Region>>,
}

/// Read the per-test coverage; `eligible` are the tests that may be
/// candidates, and every other credited test's regions are always kept. A
/// provider test the source no longer holds (`scanned` lacks it) is a row
/// of a deleted test and is not read.
fn read_per_test(
    root: &Path,
    eligible: &BTreeSet<TestId>,
    scanned: &BTreeSet<TestId>,
) -> Result<PerTest, String> {
    let baseline_rows = lines::parse(&read(root, LINES)?).map_err(|error| error.to_string())?;
    let mut baseline = BTreeSet::new();
    for (file, row) in &baseline_rows {
        for line in &row.covered_lines {
            baseline.insert(Region::Line(file.clone(), *line));
        }
        for branch in &row.covered_branches {
            baseline.insert(Region::Branch(file.clone(), *branch));
        }
    }
    let path = crate::coverage::target_dir(root).join("coverage/per-test.tsv");
    let file = std::fs::File::open(&path).map_err(|error| {
        format!(
            "{}: {error}; run `cargo xtask coverage --per-test` first",
            path.display()
        )
    })?;
    let mut forced = BTreeSet::new();
    let mut regions: BTreeMap<TestId, BTreeSet<Region>> = BTreeMap::new();
    for (number, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("{}: {error}", path.display()))?;
        if number == 0 {
            continue;
        }
        let mut columns = line.split('\t');
        let (Some(binary), Some(test), Some(source), Some(covered), branches) = (
            columns.next(),
            columns.next(),
            columns.next(),
            columns.next(),
            columns.next().unwrap_or(""),
        ) else {
            return Err(format!("{} line {}: {line:?}", path.display(), number + 1));
        };
        let Some(row) = baseline_rows.get(source) else {
            continue;
        };
        let id = (binary.to_owned(), test.to_owned());
        let candidate = eligible.contains(&id);
        if !candidate && (is_sweep(binary, test) || is_stale(&id, scanned)) {
            continue;
        }
        let mut found = Vec::new();
        for number in lines::parse_ranges(covered).unwrap_or_default() {
            if row.covered_lines.contains(&number) {
                found.push(Region::Line(source.to_owned(), number));
            }
        }
        for branch in branches.split(',').filter_map(lines::parse_branch) {
            if row.covered_branches.contains(&branch) {
                found.push(Region::Branch(source.to_owned(), branch));
            }
        }
        if candidate {
            regions.entry(id).or_default().extend(found);
        } else {
            forced.extend(found);
        }
    }
    Ok(PerTest {
        baseline,
        forced,
        regions,
    })
}
