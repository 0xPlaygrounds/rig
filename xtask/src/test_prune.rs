//! `cargo xtask tests prune [--check]`: delete the unit and integration
//! tests whose coverage the kept tests already give, by a fixed rule, and
//! list each deletion in `crates/rig-cassette/coverage/pruned-tests.tsv`
//! with the kept tests that cover what it covered.
//!
//! It is the cassette prune's rule applied to the crates' own tests: the
//! same greedy set cover and redundancy pass (`cassette prune`'s `select`),
//! over the lines and branches of the coverage baseline and the killed
//! mutants of the mutation baseline, with the contract tests kept whatever
//! they cover. The rule is written at the head of the manifest.
//!
//! A rewrite needs `target/coverage/per-test.tsv` from
//! `cargo xtask coverage --per-test`, and lists the current tests with
//! `cargo nextest list`. It takes the deleted tests out of their sources;
//! the rows of earlier deletions carry forward. `--check` reruns the
//! selection and fails when it would delete more, when a listed test is
//! still there, or when the manifest is not the one a rewrite would write.

mod facts;
mod locate;
#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::fmt::Write as _;
use std::io::BufRead as _;
use std::path::Path;

use crate::cassette::prune::{self as cassette_prune, edit, select};
use crate::coverage::lines::{self, Branch};
use crate::coverage::mutants;
use crate::support::output;

use self::facts::Reason;
use self::locate::{Located, Locator};

pub(crate) const USAGE: &str = "\
  tests prune [--check]       delete the unit and integration tests the kept
                              tests already cover; list them in
                              crates/rig-cassette/coverage/pruned-tests.tsv, or
                              with --check fail when it would delete more
";

/// The manifest, relative to the workspace root.
pub(crate) const MANIFEST: &str = "crates/rig-cassette/coverage/pruned-tests.tsv";
const LINES: &str = "crates/rig-cassette/coverage/lines.tsv";
const MUTANTS: &str = "crates/rig-cassette/coverage/mutants.tsv";

/// The packages whose tests are candidates. Every other package's tests
/// stay; rig-cassette's belong to `cassette prune`, and xtask and the
/// test-support crates test code the line baseline does not measure.
const SCOPE: &[&str] = &[
    "rig",
    "rig-agent",
    "rig-bedrock",
    "rig-candle",
    "rig-core",
    "rig-ecs",
    "rig-gemini-grpc",
    "rig-memory",
    "rig-vertexai",
];

/// Packages the per-test run leaves out, and so the listing too.
const UNMEASURED: &[&str] = &["rig-cassette-minimal", "xtask"];

/// Module names whose tests test unmeasured helper code.
const HELPER_MODULES: &[&str] = &["test_utils", "test_fixtures", "test_support"];

const PREAMBLE: &str = "\
# The unit-test prune, written by `cargo xtask tests prune`; review the rule
# and this list, not the deleted tests. Rerunning the command gives the same
# list, and `--check` fails when the tree or this list disagrees with it. The
# cassette prune keeps its own manifest, pruned.tsv, since its candidates,
# elements and keep rules differ and each `--check` owns its file whole.
#
# The candidates are the tests of rig, rig-core, rig-agent, rig-ecs,
# rig-bedrock, rig-vertexai, rig-gemini-grpc, rig-candle and rig-memory that
# the per-test run covered and the source scan places as a `#[test]`-style
# function. Every other test stays: the conformance rows (any test whose binary
# or path names `conformance`), tests of helper modules (test_utils), tests a
# macro generates, every other package's tests, and the tests of xtask and the
# test-support crates, whose code the line baseline does not measure. A test
# is deleted only when all of these hold:
#
# (a) Regions. Every line and branch of crates/rig-cassette/coverage/lines.tsv
#     it covers (`cargo xtask coverage --per-test`) is covered by a kept test.
#     Only tests that stay whatever either prune selects are credited: the
#     cassette prune's candidates and the corpus sweeps are not, so deleting
#     a unit test never changes what the cassette prune must keep.
# (b) Mutants. No killed mutant of crates/rig-cassette/coverage/mutants.tsv
#     loses its last kept killer. A row names up to three killers; a mutant
#     killed by more keeps one of the three it names. `coverage --check
#     --mutants` confirms it after a deletion.
# (c) Contracts. It asserts no contract coverage cannot see, read from its
#     tokens: a wasm test (wasm_bindgen_test, or a wasm-only cfg on it or its
#     modules), a compile-fail or trybuild case, a public API shape (its name
#     holds api_surface, public_api, reexport or facade, or nothing in it can
#     fail at run time: no assertion, `?`, unwrap, expect or panic, so what it
#     checks is that its paths and types resolve), a security or scrub
#     test (its name holds scrub, redact, secret, credential, sanitiz,
#     password, api_key, leak, security, authoriz or authenticat, or its body
#     names scrub, redact or sanitiz), a serde round trip of a stored format
#     (its name holds round_trip, serde, serializ, persist, stored or golden,
#     its body names a golden or a path into fixtures/ or an effect log, or
#     it calls serde_json or serde_yaml both ways), an error message (an
#     asserted argument holds a string literal and the body names an error
#     and renders text), or rendered text (an asserted argument renders with
#     to_string, format or Display and holds a string literal). A test whose
#     name another tracked .rs or .md file cites (a findings registry, a
#     contract table, a doc comment) is kept too: the citation documents what
#     it asserts. An error-message or rendered-text test that no file cites
#     goes when each of those assertions appears, token for token, in an
#     earlier kept contract test; every other contract test always stays.
#     Doc examples are not tests here and always stay.
#
# The selection is cassette prune's: greedy set cover over the regions and
# mutants the credited tests do not hold, taking the candidate covering the
# most, then a table-driven test (a loop over a table of cases) before any
# other, then the shorter, then the alphabetically first; then every taken
# test the others make redundant is dropped, latest first.
#
# Columns: the deleted test, and the fewest kept tests that cover every region
# and mutant it covered, the one covering most first.
kind\titem\tcovered by
";

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), String> {
    let check = match args {
        [command] if command == "prune" => false,
        [command, flag] if command == "prune" && flag == "--check" => true,
        _ => return Err(format!("usage:\n{USAGE}")),
    };
    let model = Model::load(root)?;
    let selection = select::select(&model.problem);
    let deleted = model.deleted(&selection);
    let fresh = model.rows(&selection, &deleted)?;
    let path = root.join(MANIFEST);
    let committed = match std::fs::read_to_string(&path) {
        Ok(text) => cassette_prune::parse(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let mut rows: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut stale = Vec::new();
    for row in &committed {
        let listed = row
            .item
            .split_once(' ')
            .is_some_and(|(binary, test)| model.listed.contains(&(binary.into(), test.into())));
        if listed {
            stale.push(format!("{} is listed but still in the tree", row.item));
        } else {
            rows.insert((row.kind.clone(), row.item.clone()), row.covered.clone());
        }
    }
    for row in &fresh {
        rows.insert((row.kind.clone(), row.item.clone()), row.covered.clone());
    }
    let text = render(&rows);
    let forced = model.problem.forced.len();
    println!(
        "tests prune: {} candidate tests ({} contract tests kept), {} kept, {} deleted; {} elements",
        model.candidates.len(),
        forced,
        selection.kept.len(),
        deleted.len(),
        model.problem.elements
    );
    for (reason, count) in model.reasons() {
        println!("  contract {reason}: {count}");
    }
    if check {
        let mut failures: Vec<String> = fresh
            .iter()
            .map(|row| {
                format!(
                    "prune would delete {}; run `cargo xtask tests prune`",
                    row.item
                )
            })
            .collect();
        failures.extend(stale);
        if failures.is_empty() && std::fs::read_to_string(&path).ok().as_deref() != Some(&text) {
            failures.push(format!(
                "{MANIFEST} is not the manifest the tree gives; rewrite it with \
                 `cargo xtask tests prune`"
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
    let mut by_file: BTreeMap<(&str, &str), BTreeSet<String>> = BTreeMap::new();
    for index in &deleted {
        if let (Some((_, test)), Some(located)) =
            (model.candidates.get(*index), model.located.get(*index))
        {
            by_file
                .entry((located.file.as_str(), located.module.as_str()))
                .or_default()
                .insert(test.clone());
        }
    }
    let mut taken = 0;
    for ((file, module), names) in by_file {
        let full = root.join(file);
        let source = std::fs::read_to_string(&full).map_err(|error| format!("{file}: {error}"))?;
        let (edited, removed) = edit::remove_tests(&source, module, &names)
            .map_err(|error| format!("{file}: {error}"))?;
        if removed != names.len() {
            println!("{file}: took out {removed} of {} tests", names.len());
        }
        taken += removed;
        std::fs::write(&full, edited).map_err(|error| format!("{file}: {error}"))?;
    }
    println!("took {taken} tests out of the source");
    Ok(())
}

fn render(rows: &BTreeMap<(String, String), String>) -> String {
    let mut out = PREAMBLE.to_owned();
    for ((kind, item), covered) in rows {
        let _ = writeln!(out, "{kind}\t{item}\t{covered}");
    }
    out
}

/// A test: its nextest binary id and name.
type TestId = (String, String);

/// What the per-test coverage, the baselines and the source say.
struct Model {
    /// Every current test.
    listed: BTreeSet<TestId>,
    /// The candidates in name order: a candidate's index is its id.
    candidates: Vec<TestId>,
    located: Vec<Located>,
    problem: select::Problem,
    /// Every element each candidate covers, held ones included, sorted.
    candidate_all: Vec<Vec<u32>>,
    /// The credited tests that always stay, in name order, with every
    /// element each covers.
    credited: Vec<(TestId, Vec<u32>)>,
}

/// One element: a covered line or branch of the baseline, or a killed
/// mutant.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Element {
    Line(u32, u32),
    Branch(u32, Branch),
    Mutant(String),
}

#[derive(Default)]
struct Interner {
    ids: HashMap<Element, u32>,
}

impl Interner {
    fn id(&mut self, element: Element) -> u32 {
        let next = u32::try_from(self.ids.len()).unwrap_or(u32::MAX);
        *self.ids.entry(element).or_insert(next)
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Whether a test is a conformance row.
pub(crate) fn is_conformance(binary: &str, test: &str) -> bool {
    binary.contains("conformance") || test.contains("conformance")
}

/// Whether a test sits in a helper module the baseline does not measure.
pub(crate) fn in_helper_module(test: &str) -> bool {
    test.split("::")
        .any(|segment| HELPER_MODULES.contains(&segment))
}

/// The cost the selection orders keepers by: a table-driven test first,
/// then the shorter.
pub(crate) fn cost(located: &Located) -> u64 {
    let lines = located.facts.lines as u64;
    if located.facts.table {
        lines
    } else {
        (1 << 32) + lines
    }
}

impl Model {
    fn load(root: &Path) -> Result<Self, String> {
        let listed = list(root)?;
        let roots = locate::roots(root)?;
        let uncredited = cassette_prune::candidates(root)?;

        let baseline = lines::parse(&read(root, LINES)?).map_err(|error| error.to_string())?;
        let file_ids: BTreeMap<&str, u32> = baseline
            .keys()
            .enumerate()
            .map(|(id, file)| (file.as_str(), u32::try_from(id).unwrap_or(u32::MAX)))
            .collect();
        let mut interner = Interner::default();
        let mut covers: BTreeMap<TestId, Vec<u32>> = BTreeMap::new();
        let path = crate::coverage::target_dir(root).join("coverage/per-test.tsv");
        let file = std::fs::File::open(&path).map_err(|error| {
            format!(
                "{}: {error}; run `cargo xtask coverage --per-test` first",
                path.display()
            )
        })?;
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
            let id = (binary.to_owned(), test.to_owned());
            let (Some(row), Some(file_id)) = (baseline.get(source), file_ids.get(source)) else {
                continue;
            };
            if !listed.contains(&id) {
                continue;
            }
            let set = covers.entry(id).or_default();
            for number in lines::parse_ranges(covered).unwrap_or_default() {
                if row.covered_lines.contains(&number) {
                    set.push(interner.id(Element::Line(*file_id, number)));
                }
            }
            for branch in branches.split(',').filter_map(lines::parse_branch) {
                if row.covered_branches.contains(&branch) {
                    set.push(interner.id(Element::Branch(*file_id, branch)));
                }
            }
        }

        // Candidates, the credited always-kept tests, and the rest (kept,
        // not credited).
        let cited = citations(root)?;
        let mut locator = Locator::new(root);
        let mut candidates = Vec::new();
        let mut located = Vec::new();
        let mut credited: BTreeSet<TestId> = BTreeSet::new();
        for id in covers.keys() {
            let (binary, test) = id;
            if cassette_prune::is_sweep(binary, test) || uncredited.contains(id) {
                continue;
            }
            let package = binary.split("::").next().unwrap_or(binary);
            let place = (SCOPE.contains(&package)
                && !is_conformance(binary, test)
                && !in_helper_module(test))
            .then(|| roots.get(binary))
            .flatten()
            .and_then(|start| locator.locate(binary, start, test));
            match place {
                Some(mut place) => {
                    let name = test.rsplit("::").next().unwrap_or(test);
                    let cited = cited
                        .get(name)
                        .is_some_and(|files| files.iter().any(|file| *file != place.file));
                    let exemptable = place
                        .facts
                        .contract
                        .as_ref()
                        .is_none_or(|contract| !contract.assertions.is_empty());
                    if cited && exemptable {
                        place.facts.contract = Some(facts::Contract {
                            reason: Reason::Cited,
                            assertions: Vec::new(),
                        });
                    }
                    candidates.push(id.clone());
                    located.push(place);
                }
                None => {
                    credited.insert(id.clone());
                }
            }
        }

        // Mutants: each killed one is an element of its listed named
        // killers.
        let kills = mutants::parse(&read(root, MUTANTS)?).map_err(|error| error.to_string())?;
        let mut gone = Vec::new();
        for (identity, tested) in &kills.mutants {
            if !tested.outcome.killed() || tested.killers.is_empty() {
                continue;
            }
            let killers: Vec<TestId> = tested
                .killers
                .iter()
                .filter_map(|killer| killer.split_once(' '))
                .map(|(binary, test)| (binary.to_owned(), test.to_owned()))
                .filter(|killer| listed.contains(killer))
                .collect();
            if killers.is_empty() {
                gone.push(identity.clone());
                continue;
            }
            let element = interner.id(Element::Mutant(identity.clone()));
            for killer in killers {
                covers.entry(killer).or_default().push(element);
            }
        }
        if !gone.is_empty() {
            return Err(format!(
                "{MUTANTS} names only gone tests as the killers of {} mutants, such as {}; \
                 rewrite it with `cargo xtask coverage --only mutants --package <crate>`",
                gone.len(),
                gone.first().map_or("", String::as_str)
            ));
        }
        for elements in covers.values_mut() {
            elements.sort_unstable();
            elements.dedup();
        }

        let mut held = vec![false; interner.len()];
        let mut credited_rows = Vec::new();
        for id in &credited {
            let elements = covers.get(id).cloned().unwrap_or_default();
            for element in &elements {
                if let Some(slot) = held.get_mut(*element as usize) {
                    *slot = true;
                }
            }
            credited_rows.push((id.clone(), elements));
        }

        let candidate_all: Vec<Vec<u32>> = candidates
            .iter()
            .map(|id| covers.get(id).cloned().unwrap_or_default())
            .collect();
        let mut problem = select::Problem {
            candidates: Vec::new(),
            elements: interner.len(),
            fixtures: 0,
            protected: BTreeSet::new(),
            forced: forced(&located),
        };
        for (all, place) in candidate_all.iter().zip(&located) {
            let elements = all
                .iter()
                .filter(|element| !held.get(**element as usize).copied().unwrap_or(true))
                .map(|element| *element as usize)
                .collect();
            problem.candidates.push(select::Candidate {
                elements,
                cost: cost(place),
                owns: Vec::new(),
                reads: Vec::new(),
            });
        }
        Ok(Self {
            listed,
            candidates,
            located,
            problem,
            candidate_all,
            credited: credited_rows,
        })
    }

    fn deleted(&self, selection: &select::Selection) -> Vec<usize> {
        let kept: BTreeSet<usize> = selection.kept.iter().copied().collect();
        (0..self.candidates.len())
            .filter(|index| !kept.contains(index))
            .collect()
    }

    /// How many kept contract tests each reason holds.
    fn reasons(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for index in &self.problem.forced {
            if let Some(contract) = self
                .located
                .get(*index)
                .and_then(|place| place.facts.contract.as_ref())
            {
                *counts.entry(contract.reason.as_str()).or_default() += 1;
            }
        }
        counts
    }

    /// A row per deleted test, naming the fewest kept tests that cover all
    /// it covered.
    fn rows(
        &self,
        selection: &select::Selection,
        deleted: &[usize],
    ) -> Result<Vec<cassette_prune::Row>, String> {
        let mut keepers: Vec<(&TestId, &[u32])> = self
            .credited
            .iter()
            .map(|(id, elements)| (id, elements.as_slice()))
            .collect();
        for index in &selection.kept {
            if let (Some(id), Some(elements)) =
                (self.candidates.get(*index), self.candidate_all.get(*index))
            {
                keepers.push((id, elements.as_slice()));
            }
        }
        keepers.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut index: Vec<Vec<u32>> = vec![Vec::new(); self.problem.elements];
        for (keeper, (_, elements)) in keepers.iter().enumerate() {
            for element in *elements {
                if let Some(list) = index.get_mut(*element as usize) {
                    list.push(u32::try_from(keeper).unwrap_or(u32::MAX));
                }
            }
        }
        let lists: Vec<&[u32]> = keepers.iter().map(|(_, elements)| *elements).collect();
        let mut cover = Cover::new(self.problem.elements, keepers.len());
        let mut rows = Vec::new();
        for deleted in deleted {
            let (Some((binary, test)), Some(elements)) = (
                self.candidates.get(*deleted),
                self.candidate_all.get(*deleted),
            ) else {
                continue;
            };
            let chosen = cover.cover(elements, &index, &lists).ok_or_else(|| {
                format!("{binary} {test}: the kept tests do not cover all it covered")
            })?;
            let names: Vec<String> = chosen
                .into_iter()
                .filter_map(|keeper| keepers.get(keeper as usize))
                .map(|((binary, test), _)| format!("{binary} {test}"))
                .collect();
            rows.push(cassette_prune::Row {
                kind: "test".into(),
                item: format!("{binary} {test}"),
                covered: if names.is_empty() {
                    "-".into()
                } else {
                    names.join(", ")
                },
            });
        }
        Ok(rows)
    }
}

/// The contract tests kept whatever they cover: in name order, every one
/// but those whose contract assertions all appear in an earlier kept one.
pub(crate) fn forced(located: &[Located]) -> BTreeSet<usize> {
    let mut forced = BTreeSet::new();
    let mut kept_assertions: BTreeSet<&str> = BTreeSet::new();
    for (index, place) in located.iter().enumerate() {
        let Some(contract) = &place.facts.contract else {
            continue;
        };
        let restated = !contract.assertions.is_empty()
            && matches!(contract.reason, Reason::ErrorMessage | Reason::RenderedText)
            && contract
                .assertions
                .iter()
                .all(|assertion| kept_assertions.contains(assertion.as_str()));
        if !restated {
            forced.insert(index);
            kept_assertions.extend(place.facts.assertions.iter().map(String::as_str));
        }
    }
    forced
}

/// Lazy greedy set cover of one test's elements by the keepers: the
/// keeper covering the most uncovered elements, then the earlier.
struct Cover {
    /// The round each element was last marked uncovered in.
    uncovered: Vec<u32>,
    round: u32,
    gains: Vec<u32>,
}

impl Cover {
    fn new(elements: usize, keepers: usize) -> Self {
        Self {
            uncovered: vec![0; elements],
            round: 0,
            gains: vec![0; keepers],
        }
    }

    fn is_uncovered(&self, element: u32) -> bool {
        self.uncovered.get(element as usize) == Some(&self.round)
    }

    /// The chosen keepers in index order, or `None` when some element has
    /// no keeper.
    fn cover(
        &mut self,
        elements: &[u32],
        index: &[Vec<u32>],
        lists: &[&[u32]],
    ) -> Option<Vec<u32>> {
        self.round += 1;
        let mut remaining = 0usize;
        let mut touched = Vec::new();
        for element in elements {
            if let Some(slot) = self.uncovered.get_mut(*element as usize) {
                *slot = self.round;
                remaining += 1;
            }
            for keeper in index.get(*element as usize).into_iter().flatten() {
                if let Some(gain) = self.gains.get_mut(*keeper as usize) {
                    if *gain == 0 {
                        touched.push(*keeper);
                    }
                    *gain += 1;
                }
            }
        }
        let mut heap: BinaryHeap<(u32, Reverse<u32>)> = touched
            .iter()
            .map(|keeper| {
                (
                    self.gains.get(*keeper as usize).copied().unwrap_or(0),
                    Reverse(*keeper),
                )
            })
            .collect();
        for keeper in &touched {
            if let Some(gain) = self.gains.get_mut(*keeper as usize) {
                *gain = 0;
            }
        }
        let mut chosen = Vec::new();
        while remaining > 0 {
            let (bound, Reverse(keeper)) = heap.pop()?;
            let list = lists.get(keeper as usize).copied().unwrap_or(&[]);
            let gain = list
                .iter()
                .filter(|element| self.is_uncovered(**element))
                .count();
            let gain = u32::try_from(gain).unwrap_or(u32::MAX);
            if gain == 0 {
                continue;
            }
            if gain < bound {
                heap.push((gain, Reverse(keeper)));
                continue;
            }
            for element in list {
                if self.is_uncovered(*element)
                    && let Some(slot) = self.uncovered.get_mut(*element as usize)
                {
                    *slot = 0;
                    remaining -= 1;
                }
            }
            chosen.push(keeper);
        }
        chosen.sort_unstable();
        Some(chosen)
    }
}

/// Every word of the tracked Rust and Markdown files, with the files it is
/// in. A word right after `fn` is a definition, not a citation.
fn citations(root: &Path) -> Result<HashMap<String, BTreeSet<String>>, String> {
    let listed = output(root, "git", &["ls-files", "-z", "--", "*.rs", "*.md"])?;
    let mut words: HashMap<String, BTreeSet<String>> = HashMap::new();
    for file in listed.split('\0').filter(|file| !file.is_empty()) {
        let Ok(text) = std::fs::read_to_string(root.join(file)) else {
            continue;
        };
        let mut previous = "";
        for word in text
            .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .filter(|word| !word.is_empty())
        {
            if previous != "fn" {
                words
                    .entry(word.to_owned())
                    .or_default()
                    .insert(file.to_owned());
            }
            previous = word;
        }
    }
    Ok(words)
}

/// Every current test, ignored ones included, from `cargo nextest list`.
fn list(root: &Path) -> Result<BTreeSet<TestId>, String> {
    let mut args = vec![
        "nextest",
        "list",
        "--locked",
        "--workspace",
        "--all-features",
        "--run-ignored",
        "all",
        "--message-format",
        "oneline",
    ];
    for package in UNMEASURED {
        args.extend(["--exclude", package]);
    }
    let text = output(root, "cargo", &args)?;
    Ok(text
        .lines()
        .filter_map(|line| line.split_once(' '))
        .map(|(binary, test)| (binary.to_owned(), test.trim().to_owned()))
        .collect())
}

fn read(root: &Path, relative: &str) -> Result<String, String> {
    std::fs::read_to_string(root.join(relative)).map_err(|error| format!("{relative}: {error}"))
}
