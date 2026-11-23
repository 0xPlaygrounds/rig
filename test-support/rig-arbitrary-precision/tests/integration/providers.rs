use rig_core::providers::openai::transcription::TranscriptionUsage;
use rig_core::providers::openrouter::extension::{LatencyThreshold, ThroughputThreshold};
use rig_typesafeai::types::Answer;

#[test]
fn openai_duration_usage_keeps_its_seconds() -> anyhow::Result<()> {
    let usage: TranscriptionUsage = serde_json::from_str(r#"{"type":"duration","seconds":2.50}"#)?;
    anyhow::ensure!(
        matches!(usage, TranscriptionUsage::Duration { seconds } if seconds == 2.5),
        "{usage:?}"
    );
    Ok(())
}

#[test]
fn openrouter_thresholds_read_a_number_or_percentiles() -> anyhow::Result<()> {
    let simple: ThroughputThreshold = serde_json::from_str("12.5")?;
    anyhow::ensure!(simple == ThroughputThreshold::Simple(12.5), "{simple:?}");
    let throughput: ThroughputThreshold = serde_json::from_str(r#"{"p75":1.25}"#)?;
    let ThroughputThreshold::Percentile(percentiles) = throughput else {
        anyhow::bail!("{throughput:?}");
    };
    anyhow::ensure!(percentiles.p75 == Some(1.25), "{percentiles:?}");
    for input in ["[2.5]", r#"{"p50":1.0,"p50":2.0}"#] {
        anyhow::ensure!(
            serde_json::from_str::<ThroughputThreshold>(input).is_err(),
            "{input} reads"
        );
    }
    let latency: LatencyThreshold = serde_json::from_str(r#"{"p50":0.50,"p99":1e1}"#)?;
    let LatencyThreshold::Percentile(percentiles) = latency else {
        anyhow::bail!("{latency:?}");
    };
    anyhow::ensure!(
        (percentiles.p50, percentiles.p99) == (Some(0.5), Some(10.0)),
        "{percentiles:?}"
    );
    Ok(())
}

#[test]
fn typesafe_answers_keep_their_numbers() -> anyhow::Result<()> {
    let noul: Answer = serde_json::from_str(r#"{"type":"noul","noul":0.75}"#)?;
    anyhow::ensure!(
        matches!(noul, Answer::Noul { noul } if noul == 0.75),
        "{noul:?}"
    );
    let choice: Answer = serde_json::from_str(
        r#"{"type":"choice","choice":"yes","probabilities":{"yes":0.6},"confidence":0.5}"#,
    )?;
    let Answer::Choice {
        probabilities,
        confidence,
        ..
    } = choice
    else {
        anyhow::bail!("{choice:?}");
    };
    anyhow::ensure!(
        (probabilities.get("yes").copied(), confidence) == (Some(0.6), 0.5),
        "{probabilities:?} {confidence}"
    );
    let score: Answer = serde_json::from_str(
        r#"{"type":"score","score":3.0,"probabilities":{"3":0.9},"legend":{},"confidence":0.8}"#,
    )?;
    let Answer::Score {
        score,
        probabilities,
        confidence,
        ..
    } = score
    else {
        anyhow::bail!("{score:?}");
    };
    anyhow::ensure!(
        (score, probabilities.get("3").copied(), confidence) == (3.0, Some(0.9), 0.8),
        "{score} {probabilities:?} {confidence}"
    );
    Ok(())
}
