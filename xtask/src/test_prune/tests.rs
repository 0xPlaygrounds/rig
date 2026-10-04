use super::facts::{Contract, Facts};
use super::*;

fn place(
    table: bool,
    lines: usize,
    contract: Option<(Reason, &[&str])>,
    assertions: &[&str],
) -> Located {
    Located {
        file: "src/a.rs".into(),
        module: "a".into(),
        facts: Facts {
            lines,
            table,
            contract: contract.map(|(reason, assertions)| Contract {
                reason,
                assertions: assertions.iter().map(|text| (*text).to_owned()).collect(),
            }),
            assertions: assertions.iter().map(|text| (*text).to_owned()).collect(),
        },
    }
}

#[test]
fn a_contract_test_goes_only_when_an_earlier_kept_one_restates_it() {
    let located = [
        place(false, 3, Some((Reason::ErrorMessage, &["a"])), &["a"]),
        place(false, 3, Some((Reason::ErrorMessage, &["a"])), &["a"]),
        place(false, 3, Some((Reason::ErrorMessage, &["b"])), &["b"]),
        place(false, 3, None, &["c"]),
        place(false, 3, Some((Reason::Wasm, &[])), &["a"]),
        place(false, 3, Some((Reason::Security, &[])), &[]),
        // Only an assertion-level contract is restated by its assertions.
        place(false, 3, Some((Reason::StoredFormat, &["a"])), &["a"]),
    ];
    assert_eq!(forced(&located), BTreeSet::from([0, 2, 4, 5, 6]));
}

#[test]
fn a_table_driven_keeper_comes_before_a_shorter_one() {
    let table = cost(&place(true, 40, None, &[]));
    let short = cost(&place(false, 5, None, &[]));
    let shorter = cost(&place(false, 4, None, &[]));
    assert!(table < short && shorter < short);
}

#[test]
fn conformance_rows_and_helper_tests_are_not_candidates() {
    assert!(is_conformance("rig-bedrock::history_conformance", "h1"));
    assert!(is_conformance(
        "rig::core",
        "core::streaming_conformance::x"
    ));
    assert!(!is_conformance("rig-core", "providers::openai::tests::x"));
    assert!(in_helper_module("test_utils::model_conformance::tests::x"));
    assert!(!in_helper_module("agent::tests::x"));
}

#[test]
fn the_cover_takes_the_keeper_covering_most_then_the_earlier() {
    // Keepers: 0 covers {0}, 1 covers {0, 1}, 2 covers {1, 2}, 3 covers {0, 1}.
    let lists: Vec<&[u32]> = vec![&[0], &[0, 1], &[1, 2], &[0, 1]];
    let mut index: Vec<Vec<u32>> = vec![Vec::new(); 4];
    for (keeper, list) in lists.iter().enumerate() {
        for element in *list {
            index[*element as usize].push(keeper as u32);
        }
    }
    let mut cover = Cover::new(4, lists.len());
    assert_eq!(cover.cover(&[0, 1], &index, &lists), Some(vec![1]));
    assert_eq!(cover.cover(&[0, 1, 2], &index, &lists), Some(vec![1, 2]));
    assert_eq!(
        cover.cover(&[2, 3], &index, &lists),
        None,
        "nothing covers 3"
    );
    assert_eq!(cover.cover(&[], &index, &lists), Some(vec![]));
}

#[test]
fn the_manifest_reads_back_as_written() {
    let mut rows = BTreeMap::new();
    rows.insert(
        ("test".to_owned(), "rig-core a::tests::x".to_owned()),
        "rig-core a::tests::y, rig::core core::z".to_owned(),
    );
    let text = render(&rows);
    assert!(text.starts_with(PREAMBLE));
    let parsed = cassette_prune::parse(&text).expect("the manifest parses");
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].covered, "rig-core a::tests::y, rig::core core::z");
}
