use super::LogLines;

/// The complete lines of `bytes` and the torn tail after them.
fn read(bytes: &[u8]) -> (Vec<Vec<u8>>, Vec<u8>) {
    let log = LogLines::parse(bytes.to_vec());
    let lines = log.iter().map(|(_, _, line)| line.to_vec()).collect();
    (lines, log.tail().to_vec())
}

/// A torn tail a crash left is no line: bytes after the last newline, or a
/// last line that is not JSON. A blank last line is no tear.
#[test]
fn a_torn_tail_is_no_line() {
    let first = br#"{"a":1}"#.to_vec();
    assert_eq!(
        read(b"{\"a\":1}\n{\"b\":"),
        (vec![first.clone()], b"{\"b\":".to_vec())
    );
    assert_eq!(
        read(b"{\"a\":1}\n{\"b\":\n"),
        (vec![first.clone()], b"{\"b\":\n".to_vec())
    );
    assert_eq!(read(b"{\"a\":1}\n\n"), (vec![first], Vec::new()));
}
