use super::*;

fn candidate(elements: &[usize], cost: u64, owns: &[usize], reads: &[usize]) -> Candidate {
    Candidate {
        elements: elements.to_vec(),
        cost,
        owns: owns.to_vec(),
        reads: reads.to_vec(),
    }
}

fn problem(candidates: Vec<Candidate>, elements: usize, fixtures: usize) -> Problem {
    Problem {
        candidates,
        elements,
        fixtures,
        protected: BTreeSet::new(),
        forced: BTreeSet::new(),
    }
}

fn kept(problem: &Problem) -> Vec<usize> {
    let mut kept = select(problem).kept;
    kept.sort_unstable();
    kept
}

#[test]
fn the_test_covering_most_is_taken_and_the_redundant_go() {
    // 0 covers {0,1}, 1 covers {1,2}, 2 covers {0,1,2}: 2 alone suffices.
    let problem = problem(
        vec![
            candidate(&[0, 1], 1, &[], &[]),
            candidate(&[1, 2], 1, &[], &[]),
            candidate(&[0, 1, 2], 1, &[], &[]),
        ],
        3,
        0,
    );
    assert_eq!(kept(&problem), [2]);
}

#[test]
fn ties_go_to_the_smaller_fixtures_then_the_earlier_name() {
    let cheaper = problem(
        vec![candidate(&[0], 9, &[], &[]), candidate(&[0], 3, &[], &[])],
        1,
        0,
    );
    assert_eq!(kept(&cheaper), [1]);
    let earlier = problem(
        vec![candidate(&[0], 3, &[], &[]), candidate(&[0], 3, &[], &[])],
        1,
        0,
    );
    assert_eq!(kept(&earlier), [0]);
}

#[test]
fn a_test_taken_early_goes_when_later_ones_cover_it() {
    // Greedy takes 0 (three elements) first; 1 and 2 are still needed for
    // elements 3 and 4, and together they cover 0's, so 0 goes.
    let problem = problem(
        vec![
            candidate(&[0, 1, 2], 1, &[], &[]),
            candidate(&[0, 1, 3], 1, &[], &[]),
            candidate(&[2, 4], 1, &[], &[]),
        ],
        5,
        0,
    );
    assert_eq!(kept(&problem), [1, 2]);
}

#[test]
fn a_forced_test_stays_and_its_elements_count() {
    let mut problem = problem(
        vec![candidate(&[0], 1, &[], &[]), candidate(&[0], 1, &[], &[])],
        1,
        0,
    );
    problem.forced.insert(1);
    assert_eq!(kept(&problem), [1]);
}

#[test]
fn a_kept_fixture_keeps_its_cheapest_owner() {
    // 0 reads fixture 0, which 1 and 2 own; nothing else needs them.
    let mut reads = problem(
        vec![
            candidate(&[0], 1, &[], &[0]),
            candidate(&[], 7, &[0], &[]),
            candidate(&[], 5, &[0], &[]),
        ],
        1,
        1,
    );
    assert_eq!(kept(&reads), [0, 2]);
    // A protected fixture keeps an owner with no reader at all.
    reads.candidates.clear();
    reads.candidates.push(candidate(&[], 1, &[0], &[]));
    reads.protected.insert(0);
    assert_eq!(kept(&reads), [0]);
}

#[test]
fn the_sole_owner_of_a_fixture_a_kept_test_reads_is_not_redundant() {
    // 1 owns fixture 0 and covers only what 2 covers, but 0 reads fixture 0.
    let problem = problem(
        vec![
            candidate(&[0], 1, &[], &[0]),
            candidate(&[1], 1, &[0], &[]),
            candidate(&[1, 2], 1, &[], &[]),
        ],
        3,
        1,
    );
    assert_eq!(kept(&problem), [0, 1, 2]);
}

#[test]
fn cover_with_names_the_fewest_kept_tests() {
    let problem = problem(
        vec![
            candidate(&[0, 1], 1, &[], &[]),
            candidate(&[1, 2], 1, &[], &[]),
            candidate(&[2], 1, &[], &[]),
        ],
        3,
        0,
    );
    assert_eq!(cover_with(&[0, 1, 2], &[0, 1, 2], &problem), [0, 1]);
    assert_eq!(cover_with(&[2], &[0, 2], &problem), [2]);
    assert!(cover_with(&[5], &[0], &problem).is_empty());
}
