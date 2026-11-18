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

/// No golden retrieval sets a threshold, so one is set here.
#[test]
fn a_retrieval_keeps_its_threshold() -> anyhow::Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/rig-cassette/fixtures/effects/gemini_retrieval_context_and_tools.effects.json");
    let log: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mut kind = log["records"][0]["kind"].clone();
    kind["query"]["req"]["threshold"] = serde_json::json!(0.5);
    let read: rig_core::effect::EffectKind = serde_json::from_str(&kind.to_string())?;
    anyhow::ensure!(serde_json::to_value(&read)? == kind, "{read:?}");
    Ok(())
}
