use serde_json::json;

use super::{parse_claude_usage, parse_codex_usage, parse_grok_usage, parse_muse_usage};

#[test]
fn claude_usage_formats_subscription_windows() {
    let msg = json!({
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": "rig-usage",
            "response": {
                "subscription_type": "max",
                "rate_limits_available": true,
                "rate_limits": {
                    "five_hour": { "utilization": 12.4, "resets_at": "2026-11-22T14:00:00Z" },
                    "seven_day": { "utilization": 18, "resets_at": "2026-11-23T04:59:59Z" }
                }
            }
        }
    });
    let text = parse_claude_usage(&msg);
    assert!(text.contains("plan max"));
    assert!(text.contains("5h 12% used"));
    assert!(text.contains("weekly 18% used"));
}

#[test]
fn codex_usage_formats_rate_limits() {
    let msg = json!({
        "id": 2,
        "result": {
            "rateLimits": {
                "planType": "promax",
                "primary": { "usedPercent": 64, "windowDurationMins": 10080, "resetsAt": 1795376764 },
                "secondary": null
            }
        }
    });
    let text = parse_codex_usage(&msg);
    assert!(text.contains("plan promax"));
    assert!(text.contains("weekly 64% used"));
}

#[test]
fn muse_usage_formats_tier_and_windows() {
    let val = json!({
        "allowance": {
            "tier": "Muse Code Max Usage",
            "window": { "usedPercent": 5, "resetsAtMs": 1795358157000u64 },
            "weekly": { "usedPercent": 22, "resetsAtMs": 1795392000000u64 }
        }
    });
    let text = parse_muse_usage(&val);
    assert!(text.contains("Muse Code Max Usage"));
    assert!(text.contains("window 5% used"));
    assert!(text.contains("weekly 22% used"));
}

#[test]
fn grok_usage_handles_percent_fields() {
    let val = json!({ "weekly": { "usedPercent": 35 } });
    assert_eq!(parse_grok_usage(&val), "weekly 35% used");
}
