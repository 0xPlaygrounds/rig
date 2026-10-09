use super::reaches;

#[test]
fn reaches_follows_edges_transitively() {
    let edges = [(1, 2), (2, 3), (4, 1)];
    assert!(reaches(&edges, 1, 2));
    assert!(reaches(&edges, 1, 3));
    assert!(reaches(&edges, 4, 3));
    assert!(!reaches(&edges, 3, 1));
    assert!(!reaches(&edges, 2, 4));
}

#[test]
fn reaches_finds_cycles_and_ends_on_them() {
    let edges = [(1, 2), (2, 1), (2, 3)];
    assert!(reaches(&edges, 1, 1));
    assert!(reaches(&edges, 2, 3));
    assert!(!reaches(&edges, 3, 1));
    assert!(!reaches(&[(5, 6)], 1, 1));
}
