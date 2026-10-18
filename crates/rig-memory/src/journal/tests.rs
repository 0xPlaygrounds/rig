use std::path::PathBuf;

use rig_protocol::{EventRecord, Seq, SessionId};

use super::{FILE, Journal};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rig-journal-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn record(seq: u64) -> EventRecord {
    EventRecord {
        session: SessionId("s-1".into()),
        seq: Seq(seq),
        at_ms: seq,
        kind: "notice".into(),
        data: serde_json::json!({"level": "info", "text": format!("n{seq}")}),
        source: None,
    }
}

#[test]
fn a_torn_tail_is_cut_to_the_last_whole_record() {
    let dir = scratch("torn");
    let session = SessionId("s-1".into());
    let mut journal = Journal::open(&dir, &session).unwrap();
    journal.append(&[record(1), record(2), record(3)]).unwrap();
    drop(journal);

    // A power loss mid-write leaves half a line.
    let path = dir.join(FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    let whole = bytes.len();
    bytes.extend_from_slice(br#"{"session":"s-1","seq":4,"at_m"#);
    std::fs::write(&path, &bytes).unwrap();

    let mut journal = Journal::open(&dir, &session).unwrap();
    assert_eq!(journal.last(), Seq(3));
    assert_eq!(std::fs::metadata(&path).unwrap().len(), whole as u64);
    journal.append(&[record(4)]).unwrap();
    let seqs: Vec<u64> = journal
        .read_after(Seq(0), 100)
        .unwrap()
        .iter()
        .map(|r| r.seq.0)
        .collect();
    assert_eq!(seqs, [1, 2, 3, 4]);
    assert_eq!(journal.read_after(Seq(2), 1).unwrap()[0].seq, Seq(3));
    assert!(journal.read_after(Seq(4), 10).unwrap().is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_gap_another_session_or_format_is_refused() {
    let dir = scratch("refuse");
    let session = SessionId("s-1".into());
    let mut journal = Journal::open(&dir, &session).unwrap();
    assert!(journal.append(&[record(2)]).is_err());
    assert_eq!(journal.last(), Seq(0));
    drop(journal);
    assert!(Journal::open(&dir, &SessionId("s-2".into())).is_err());

    std::fs::write(
        dir.join(FILE),
        "{\"format\":\"rig-events/2\",\"session\":\"s-1\"}\n",
    )
    .unwrap();
    let refused = Journal::open(&dir, &session).err().unwrap();
    assert!(refused.to_string().contains("update rig"), "{refused}");
    std::fs::remove_dir_all(&dir).ok();
}
