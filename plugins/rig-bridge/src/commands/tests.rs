use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::ModelSpec;
use rig_harness::rig_core::effect::{EffectKind, FamilyDescriptor, HandlerDescriptor};
use rig_harness::rig_core::error::{ErrorKind, ErrorReport};
use rig_harness::rig_core::providers::registry::ProviderId;
use rig_harness::rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use serde_json::json;
use tokio::sync::mpsc;

use super::{
    BridgeSignal, SignalInbox, drain_signals, format_grok_usage_error, parse_claude_usage,
    parse_codex_usage, parse_grok_usage, parse_muse_usage,
};
use crate::harness::Vendor;

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
    assert_eq!(
        format_grok_usage_error(
            "Error: `grok usage` can't show your plan's usage here yet. Run /usage inside grok instead."
        ),
        "unavailable (run `/usage` inside grok)"
    );
}

#[test]
fn drain_signals_updates_catalog_connection_and_refreshes_on_last_usage() {
    struct Stub;
    impl Serve for Stub {
        type Family = rig_harness::rig_core::effect::family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            HandlerDescriptor {
                key: rig_harness::rig_core::effect::model_key("grok-build/grok-4.7"),
                family: FamilyDescriptor::Completion {
                    model: "grok-build/grok-4.7".into(),
                    capabilities: Default::default(),
                },
                layers: Vec::new(),
            }
        }
        async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
            Reply::Outcome(Err(ErrorReport::new(ErrorKind::Internal, "stub")))
        }
    }

    let Ok(pid) = ProviderId::declare(Vendor::GrokBuild.as_str()) else {
        return;
    };
    let initial = ModelSpec::new(pid, "grok-4.7").with_context_window(500_000);
    let (tx, rx) = mpsc::unbounded_channel();
    let inserts = Arc::new(AtomicUsize::new(0));
    let seen_inserts = inserts.clone();

    let mut app = App::new();
    let mut models = Models::default();
    models.0.catalog_mut().insert(initial.clone());
    app.insert_resource(models)
        .insert_resource(SignalInbox::new(rx))
        .add_observer(move |_: On<Insert<Connection>>| {
            seen_inserts.fetch_add(1, Ordering::SeqCst);
        })
        .add_systems(Update, drain_signals);

    let agent = app
        .world_mut()
        .spawn((
            Connection {
                spec: Arc::new(initial),
                handler: rig_harness::rig_ecs::effects::Handler(ErasedHandler::new(Stub)),
            },
            LastUsage(None),
        ))
        .id();
    inserts.store(0, Ordering::SeqCst);

    let _ = tx.send(BridgeSignal::Window {
        vendor: Vendor::GrokBuild,
        model: "grok-4.7".to_owned(),
        limit: 256_000,
    });
    app.update();

    let conn_limit = app
        .world()
        .get::<Connection>(agent)
        .and_then(|c| c.spec.input_limit());
    let cat_limit = app
        .world()
        .resource::<Models>()
        .0
        .catalog()
        .get_exact(pid, "grok-4.7")
        .and_then(ModelSpec::input_limit);
    assert_eq!(conn_limit, Some(256_000));
    assert_eq!(cat_limit, Some(256_000));
    assert_eq!(inserts.load(Ordering::SeqCst), 1);

    // Simulate Turn 1 completing (changing LastUsage): drain_signals must
    // re-insert Connection once so rig-basics clears any stale Turn-1 InputLimit.
    if let Some(mut last) = app.world_mut().get_mut::<LastUsage>(agent) {
        last.0 = None;
    }
    app.update();
    assert_eq!(inserts.load(Ordering::SeqCst), 2);
    assert_eq!(
        app.world()
            .get::<Connection>(agent)
            .and_then(|c| c.spec.input_limit()),
        Some(256_000)
    );
}

#[test]
fn drain_signals_emits_notice_and_releases_keep_running_holder() {
    use bevy_ecs::message::Messages;

    let (tx, rx) = mpsc::unbounded_channel();
    let mut app = App::new();
    app.init_resource::<Messages<Notice>>()
        .insert_resource(SignalInbox::new(rx))
        .add_systems(Update, drain_signals);

    let agent = app.world_mut().spawn_empty().id();
    let holder = app.world_mut().spawn(KeepRunning).id();
    let _ = tx.send(BridgeSignal::Notice {
        agent: Some(agent),
        warn: false,
        text: "usage report".to_owned(),
        release: Some(holder),
    });
    app.update();

    assert!(app.world().get_entity(holder).is_err());
    let notices = app.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let texts: Vec<_> = cursor
        .read(notices)
        .map(|notice| (notice.agent, notice.text.as_str()))
        .collect();
    assert_eq!(texts, vec![(Some(agent), "usage report")]);
}
