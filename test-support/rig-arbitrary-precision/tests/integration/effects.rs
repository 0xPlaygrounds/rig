use std::path::Path;

use rig_cassette::effect_log::EffectLog;

/// The checked-in effect logs hold requests' temperatures, costs,
/// embeddings, retrieval scores and rerank scores.
#[test]
fn every_golden_effect_log_loads_as_it_was_written() -> anyhow::Result<()> {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/rig-cassette/fixtures/effects");
    let mut read = 0;
    for file in std::fs::read_dir(&root)? {
        let path = file?.path();
        if !path.to_string_lossy().ends_with(".effects.json") {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let log: EffectLog = serde_json::from_str(&text)
            .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
        let written: serde_json::Value = serde_json::from_str(&text)?;
        anyhow::ensure!(
            serde_json::to_value(&log)? == written,
            "{} does not read back as written",
            path.display()
        );
        read += 1;
    }
    anyhow::ensure!(read > 100, "{read} golden logs under {}", root.display());
    Ok(())
}

/// No golden retrieval sets a threshold, so this record does.
#[test]
fn a_retrieval_keeps_its_threshold() -> anyhow::Result<()> {
    let text = r#"{"effect":"retrieve","query":{"query":"top_n","req":{"query":"q","samples":1,"threshold":0.5,"filter":null}}}"#;
    let read: rig_core::effect::EffectKind = serde_json::from_str(text)?;
    let written = serde_json::to_value(&read)?;
    anyhow::ensure!(
        written == serde_json::from_str::<serde_json::Value>(text)?,
        "{written}"
    );
    Ok(())
}
