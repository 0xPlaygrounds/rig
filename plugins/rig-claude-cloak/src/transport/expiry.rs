//! Opt-in expiry observations use three independent prefixes so the six-minute
//! long-retention read cannot refresh the prefix observed after 65 minutes.

use super::live::{MODEL, ProbeResult, complete, handler, request};
use rig_harness::rig_core::completion::CacheRetention;
use std::time::{Duration, Instant};

// This probe's evidence threshold is not a provider or model cache minimum.
const ELIGIBILITY_READS: u64 = 4096;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
enum ObservationError {
    #[error("UNKNOWN: substantial cache reuse was not established; expiry is inconclusive")]
    Ineligible,
    #[error("UNKNOWN: cache read usage was absent; expiry is inconclusive")]
    MissingUsage,
    #[error("UNKNOWN: observation exceeded timing tolerance; expiry is inconclusive")]
    Timing,
    #[error("FAIL: observed cache retention differs from expected expiry window")]
    Mismatch,
}

fn eligibility(reads: Option<u64>) -> Result<u64, ObservationError> {
    let reads = reads.ok_or(ObservationError::MissingUsage)?;
    if reads < ELIGIBILITY_READS {
        return Err(ObservationError::Ineligible);
    }
    Ok(reads)
}

fn classify(
    baseline: u64,
    reads: Option<u64>,
    expect_warm: bool,
    elapsed: Duration,
    minutes: u64,
) -> Result<(), ObservationError> {
    eligibility(Some(baseline))?;
    let reads = reads.ok_or(ObservationError::MissingUsage)?;
    if elapsed > Duration::from_secs((minutes + 2) * 60) {
        return Err(ObservationError::Timing);
    }
    if (expect_warm && reads >= baseline) || (!expect_warm && reads < ELIGIBILITY_READS) {
        Ok(())
    } else {
        Err(ObservationError::Mismatch)
    }
}

#[test]
#[ignore = "requires separately authorized stored-credential use, nine requests and 70 minutes"]
fn real_cache_expiry() -> ProbeResult<()> {
    if std::env::var("CLAUDE_CLOAK_LIVE_EXPIRY").as_deref() != Ok("1")
        || std::env::var("CLAUDE_CLOAK_LIVE_MODEL").as_deref() != Ok(MODEL)
    {
        return Err("set CLAUDE_CLOAK_LIVE_EXPIRY=1 and CLAUDE_CLOAK_LIVE_MODEL=claude-cloak/claude-opus-5-5 only after approving charges".into());
    }
    bevy_tasks::block_on(async {
        rig_http::wasm_compat::timeout(Duration::from_secs(4200), probe())
            .await
            .map_err(|_| "expiry probe exceeded 70 minutes")?
    })
}

#[cfg(test)]
mod tests;

async fn probe() -> ProbeResult<()> {
    let handler = handler()?;
    let mut prefixes = Vec::new();
    let mut id = 0;
    for (name, retention, minutes, expect_warm) in [
        ("short-expiry", CacheRetention::Short, 6, false),
        ("long-window", CacheRetention::Long, 6, true),
        ("long-expiry", CacheRetention::Long, 65, false),
    ] {
        let nonce = uuid::Uuid::new_v4();
        let mut prompt = format!("Synthetic expiry experiment {nonce} {name}.\n");
        for line in 0..2048 {
            prompt.push_str(&format!(
                "Synthetic row {line}: amber cedar quartz meadow.\n"
            ));
        }
        prompt.push_str("Reply only OK.");
        let request = request(&prompt).cache(retention).max_tokens(16);
        for _ in 0..2 {
            let response = rig_http::wasm_compat::timeout(
                Duration::from_secs(60),
                complete(&handler, request.clone(), false, id),
            )
            .await
            .map_err(|_| "expiry request timed out")??;
            id += 1;
            if response.text().trim() != "OK" || response.model() != Some("claude-opus-5-5") {
                return Err("expiry request did not complete the synthetic instruction".into());
            }
            if id % 2 == 0 {
                let reads =
                    eligibility(response.usage.cached_input_tokens).inspect_err(|error| {
                        eprintln!("{error} {name} probe_threshold={ELIGIBILITY_READS}");
                    })?;
                eprintln!(
                    "PASS {name} eligibility reads={reads} probe_threshold={ELIGIBILITY_READS}"
                );
                prefixes.push((
                    name,
                    request.clone(),
                    Instant::now(),
                    minutes,
                    expect_warm,
                    reads,
                ));
            }
        }
    }
    for (name, request, written, minutes, expect_warm, baseline) in prefixes {
        let delay = Duration::from_secs(minutes * 60).saturating_sub(written.elapsed());
        rig_http::wasm_compat::sleep(delay).await;
        let response = rig_http::wasm_compat::timeout(
            Duration::from_secs(60),
            complete(&handler, request, false, id),
        )
        .await
        .map_err(|_| "expiry request timed out")??;
        id += 1;
        if response.text().trim() != "OK" || response.model() != Some("claude-opus-5-5") {
            return Err("expiry read did not complete the synthetic instruction".into());
        }
        let reads = response.usage.cached_input_tokens;
        let result = classify(baseline, reads, expect_warm, written.elapsed(), minutes);
        let classification = match &result {
            Ok(()) => "PASS".to_owned(),
            Err(error) => error.to_string(),
        };
        eprintln!(
            "{classification} {name} elapsed_seconds={} baseline_reads={baseline} reads={reads:?} writes={:?}",
            written.elapsed().as_secs(),
            response.usage.cache_creation_input_tokens
        );
        result?;
    }
    Ok(())
}
