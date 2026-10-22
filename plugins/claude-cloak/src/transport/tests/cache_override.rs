use super::*;

/// Synthetic wire shaping exercises TTL combinations without account access.
#[test]
fn raw_root_control_preserves_explicit_blocks_without_automatic_markers() -> anyhow::Result<()> {
    for retention in ["none", "short", "long"] {
        for ttl in ["5m", "1h"] {
            let root = json!({"type":"ephemeral","ttl":ttl});
            let head = json!({"type":"ephemeral","ttl":"1h"});
            let value = shaped(json!({
                "_omp":{"cache_retention":retention,"explicit_cache_control":true},
                "cache_control":root,
                "tools":[{"name":"read","cache_control":head}],
                "system":"stable",
                "messages":[{"role":"user","content":"tail"}]
            }))?;
            anyhow::ensure!(value.get("cache_control") == Some(&root));
            anyhow::ensure!(value.pointer("/tools/0/cache_control") == Some(&head));
            let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
            anyhow::ensure!(
                cache_blocks(object)
                    .filter(|b| b.get("cache_control").is_some())
                    .count()
                    == 1
            );
            validate_caching(object)?;
        }
    }
    Ok(())
}

/// Synthetic invalid bodies verify rejection before an HTTP transport is used.
#[test]
fn raw_root_and_final_explicit_control_require_matching_ttls() -> anyhow::Result<()> {
    for (root, block, rejected) in [
        (
            json!({"type":"ephemeral","ttl":"5m"}),
            json!({"type":"ephemeral","ttl":"1h"}),
            true,
        ),
        (
            json!({"type":"ephemeral"}),
            json!({"type":"ephemeral","ttl":"5m"}),
            false,
        ),
        (
            json!({"type":"ephemeral","ttl":"1h"}),
            json!({"type":"ephemeral","ttl":"1h"}),
            false,
        ),
    ] {
        let source = json!({
            "_omp":{"cache_retention":"long","explicit_cache_control":true},
            "cache_control":root,
            "messages":[{"role":"user","content":[
                {"type":"text","text":"tail","cache_control":block},
                {"type":"thinking","thinking":"signed reasoning","signature":"synthetic"}
            ]}]
        });
        let outcome = body(&serde_json::to_vec(&source)?, "device", "session", None);
        if rejected {
            anyhow::ensure!(matches!(outcome, Err(Error::CacheTtl)));
        } else {
            outcome?;
        }
    }
    Ok(())
}

/// Synthetic controls cover the final shaped marker budget without live calls.
#[test]
fn raw_root_reserves_budget_and_keeps_all_explicit_points() -> anyhow::Result<()> {
    for count in [3, 4] {
        let control = json!({"type":"ephemeral","ttl":"5m"});
        let messages: Vec<_> = (0..count).map(|i| json!({
            "role":"user","content":[{"type":"text","text":format!("turn {i}"),"cache_control":control}]
        })).collect();
        let source = json!({
            "_omp":{"cache_retention":"long","explicit_cache_control":true},
            "cache_control":control,"messages":messages
        });
        let outcome = body(&serde_json::to_vec(&source)?, "device", "session", None);
        if count == 4 {
            anyhow::ensure!(matches!(outcome, Err(Error::CacheBudget)));
        } else {
            let value: Value = serde_json::from_slice(&outcome?)?;
            let object = value.as_object().ok_or_else(|| anyhow::anyhow!("object"))?;
            anyhow::ensure!(validate_caching(object)?.0 == 4);
        }
    }
    Ok(())
}
