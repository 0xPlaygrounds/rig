use super::*;

#[test]
fn a_stream_is_cut_after_every_blank_line() {
    let chunks = frames(b"data: 1\n\nevent: a\r\ndata: 2\r\n\r\ndata: [DONE]");
    let chunks: Vec<&[u8]> = chunks.iter().map(|chunk| chunk.as_ref()).collect();
    assert_eq!(
        chunks,
        [
            b"data: 1\n\n".as_slice(),
            b"event: a\r\ndata: 2\r\n\r\n".as_slice(),
            b"data: [DONE]".as_slice()
        ]
    );
}

#[test]
fn an_ending_is_read_off_the_values_a_reply_carries() {
    let of =
        |ends: &[&str]| Ending::of(&ends.iter().map(|end| (*end).to_owned()).collect::<Vec<_>>());
    assert_eq!(of(&["stop"]), Ending::Stop);
    assert_eq!(of(&["tool_use"]), Ending::Tool);
    assert_eq!(of(&["completed", "in_progress"]), Ending::Stop);
    assert_eq!(
        of(&["in_progress", "incomplete", "max_output_tokens"]),
        Ending::Length
    );
    assert_eq!(of(&["STOP", "tool_calls"]), Ending::Tool);
    assert_eq!(of(&[]), Ending::Other);
    assert_eq!(of(&["length", "stop"]), Ending::Other);
}

#[test]
fn every_bank_file_reads_and_every_script_resolves() {
    for provider in providers() {
        assert!(!entries(&provider).is_empty(), "{provider}");
    }
    for fixture in SCRIPTS.keys() {
        let (provider, scenario) = fixture
            .strip_suffix(".yaml")
            .and_then(|path| path.split_once('/'))
            .expect("a fixture path");
        if SCRIPTS[fixture].iter().all(Option::is_some) {
            assert!(!script(provider, scenario).is_empty(), "{fixture}");
        }
    }
}
