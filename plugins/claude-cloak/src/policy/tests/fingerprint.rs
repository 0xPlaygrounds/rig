use super::*;
use rig_harness::rig_core::conversation::{Author, Conversation, Entry};

/// Saved-state migration uses authored history without provider or cassette access.
#[test]
fn legacy_suffix_migrates_to_the_first_retained_human() -> anyhow::Result<()> {
    let mut conversation = Conversation::default();
    conversation.push(Entry::new(Message::user("injected context")).by(Author::rig()));
    let human = conversation.push(Message::user("abcd😀XYabcdefghijklmnop"));
    let mut fingerprint = HumanFingerprint(Some("642".into()), Some(999));
    let mut seed = FingerprintSeed::default();
    refresh_fingerprint(&conversation, &mut fingerprint, &mut seed);
    anyhow::ensure!(fingerprint.0.as_deref() == Some("05c"));
    anyhow::ensure!(fingerprint.1 == Some(human.get()));
    anyhow::ensure!(seed.units == Some([0xd83d, u16::from(b'Y'), u16::from(b'm')]));
    anyhow::ensure!(seed.version == "2.1.307");
    Ok(())
}

/// A compacted legacy save cannot recover the old three-character hash input.
#[test]
fn compacted_legacy_suffix_starts_an_empty_seed_boundary_without_a_human() -> anyhow::Result<()> {
    let mut conversation = Conversation::default();
    conversation.push(Entry::new(Message::user("retained summary")).by(Author::rig()));
    let mut fingerprint = HumanFingerprint(Some("old".into()), Some(999));
    let mut seed = FingerprintSeed::default();
    refresh_fingerprint(&conversation, &mut fingerprint, &mut seed);
    let expected = crate::profile::suffix("");
    anyhow::ensure!(fingerprint.0.as_ref() == Some(&expected));
    anyhow::ensure!(fingerprint.1.is_none());
    anyhow::ensure!(seed.units == Some([u16::from(b'0'); 3]));
    conversation.push(Message::user("later human must not churn the prefix"));
    refresh_fingerprint(&conversation, &mut fingerprint, &mut seed);
    anyhow::ensure!(fingerprint.0.as_ref() == Some(&expected));
    Ok(())
}

/// A saved seed supports a version update after the original human has been compacted.
#[test]
fn version_refresh_uses_the_saved_utf16_seed_after_compaction() -> anyhow::Result<()> {
    let conversation = Conversation::from_messages([Message::user("different retained human")]);
    let mut fingerprint = HumanFingerprint(Some("old".into()), Some(999));
    let mut seed = FingerprintSeed {
        units: Some([0xd83d, u16::from(b'Y'), u16::from(b'm')]),
        version: "2.1.280".into(),
    };
    refresh_fingerprint(&conversation, &mut fingerprint, &mut seed);
    anyhow::ensure!(fingerprint.0.as_deref() == Some("05c"));
    anyhow::ensure!(fingerprint.1 == Some(999));
    anyhow::ensure!(seed.version == "2.1.307");
    Ok(())
}
