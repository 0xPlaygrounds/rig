//! Host assembly and transactional execution restoration. Provider configuration
//! never enters the world; only its advertised execution contract is saved.

use crate::bus_support;
use bevy_reflect::TypePath;
use rig_core::{
    driver::Bind,
    effect::HandlerKey,
    providers::openai::wire::OpenAI,
    serve::{ErasedHandler, adapters::CompletionAdapter},
    test_utils::RecordingHttpClient,
};
use rig_ecs::{
    bus::{Bound, EffectOutcome, Handlers, PendingEffect},
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
};

const KEY: &str = "host/model";
const BODY: &str = r#"{"id":"c","object":"chat.completion","created":0,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;

fn handler(label: &str, token: &str, endpoint: &str) -> (ErasedHandler, RecordingHttpClient) {
    let http = RecordingHttpClient::new(BODY);
    let model = OpenAI::new(token)
        .with_base_url(endpoint)
        .bind(http.clone())
        .chat("model-x");
    (
        ErasedHandler::new(CompletionAdapter::new(label, model)),
        http,
    )
}

// Forward real execution while varying only the advertised contract.
struct AdvertisedHandler {
    descriptor: rig_core::effect::HandlerDescriptor,
    inner: ErasedHandler,
}

impl rig_core::serve::Serve for AdvertisedHandler {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        self.descriptor.clone()
    }

    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        self.inner.handle(kind, dispatch).await
    }
}

fn saved(keys: &[&str]) -> Checkpoint {
    let mut app = bus_support::app();
    for key in keys {
        let (handler, _) = handler("saved", "old-secret", "https://old.invalid/v1");
        Handlers::with(app.world_mut(), |h| h.register_erased(*key, handler))
            .unwrap()
            .unwrap();
    }
    save_world(app.world_mut()).unwrap()
}

fn descriptor(app: &mut bevy_app::App, key: &str) -> rig_core::effect::HandlerDescriptor {
    Handlers::with(app.world_mut(), |h| h.descriptor(&HandlerKey::from(key)))
        .unwrap()
        .unwrap()
}

fn dispatch(app: &mut bevy_app::App) {
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, bus_support::completion()))
        .id();
    bus_support::tick_until(app, "host model answers", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        bus_support::text_of(&app.world().get::<EffectOutcome>(effect).unwrap().0),
        "hi"
    );
}

#[test]
fn direct_and_declarative_host_assembly_send_identical_requests() {
    use rig_core::providers::{openai::wire::Route, registry::ProviderConfig};
    let endpoint = "https://explicit.invalid/v1";
    let (direct, direct_http) = handler("saved", "host-token", endpoint);
    let config = ProviderConfig::OpenAi(
        OpenAI::new("discarded-credential")
            .with_base_url(endpoint)
            .with_route(Route::Chat),
    );
    let stored = serde_json::to_string(&config).unwrap();
    assert!(!stored.contains("discarded-credential"));
    let config: ProviderConfig = serde_json::from_str(&stored).unwrap();
    let declarative_http = RecordingHttpClient::new(BODY);
    let declarative = config.with_credential("host-token").completion_handler(
        "saved",
        "model-x",
        rig_core::http_client::BoxedHttpClient::new(declarative_http.clone()),
    );
    for handler in [direct, declarative] {
        let mut app = bus_support::app();
        Handlers::with(app.world_mut(), |h| h.register_erased(KEY, handler))
            .unwrap()
            .unwrap();
        dispatch(&mut app);
    }
    assert_eq!(direct_http.requests().len(), 1);
    assert_eq!(direct_http.requests(), declarative_http.requests());
}

#[test]
fn host_constructed_model_restores_without_provider_data_or_credential_lookup() {
    let checkpoint = saved(&[KEY]);
    let json = checkpoint.to_json().unwrap();
    assert!(!json.contains("old-secret"));
    assert!(!json.contains("old.invalid"));
    assert!(!json.contains("ProviderBinding"));
    assert_eq!(checkpoint.requirements().unwrap().len(), 1);
    let mut app = bus_support::app();
    checkpoint.validate(app.world()).unwrap();
    let (handler, http) = handler("saved", "rotated-secret", "https://gateway.invalid/v1");
    load_world(
        &checkpoint,
        app.world_mut(),
        RestoreMode::Strict,
        [(KEY.into(), handler)],
    )
    .unwrap();
    dispatch(&mut app);
    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].uri.starts_with("https://gateway.invalid/v1"));
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer rotated-secret"
    );
}

#[test]
fn a_matching_supplied_handler_really_rotates_the_live_connection() {
    let checkpoint = saved(&[KEY]);
    let mut app = bus_support::app();
    let (old, old_http) = handler("saved", "old", "https://old.invalid/v1");
    let entity = Handlers::with(app.world_mut(), |h| h.register_erased(KEY, old))
        .unwrap()
        .unwrap();
    let (new, new_http) = handler("saved", "new", "https://new.invalid/v1");
    let loaded = load_world(
        &checkpoint,
        app.world_mut(),
        RestoreMode::Strict,
        [(KEY.into(), new)],
    )
    .unwrap();
    assert!(loaded.entities.contains(&entity));
    dispatch(&mut app);
    assert!(old_http.requests().is_empty());
    let requests = new_http.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].uri.starts_with("https://new.invalid/v1"));
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer new"
    );
}

#[test]
fn omitted_handlers_keep_matching_preinstalled_implementations() {
    let checkpoint = saved(&[KEY]);
    let mut app = bus_support::app();
    let (live, http) = handler("saved", "live", "https://live.invalid/v1");
    Handlers::with(app.world_mut(), |h| h.register_erased(KEY, live))
        .unwrap()
        .unwrap();
    load_world(&checkpoint, app.world_mut(), RestoreMode::Strict, []).unwrap();
    dispatch(&mut app);
    assert_eq!(http.requests().len(), 1);
}

#[test]
fn supplied_advertised_key_is_normalized_to_the_saved_dispatch_key() {
    let checkpoint = saved(&[KEY]);
    assert!(
        checkpoint
            .entities
            .iter()
            .any(|row| row.contains_key(Bound::type_path()))
    );
    let saved_descriptor = checkpoint.requirements().unwrap().remove(0);
    let mut app = bus_support::app();
    let (live, http) = handler("saved", "rotated", "https://rebound.invalid/v1");
    let mut advertised = live.descriptor();
    advertised.key = "another/advertised-key".into();
    assert_ne!(advertised.key, saved_descriptor.key);
    let supplied = ErasedHandler::new(AdvertisedHandler {
        descriptor: advertised,
        inner: live,
    });
    load_world(
        &checkpoint,
        app.world_mut(),
        RestoreMode::Strict,
        [(KEY.into(), supplied)],
    )
    .unwrap();
    assert_eq!(descriptor(&mut app, KEY), saved_descriptor);
    assert_eq!(
        Handlers::with(app.world_mut(), |h| h.keys()).unwrap(),
        vec![HandlerKey::from(KEY)]
    );
    dispatch(&mut app);
    assert_eq!(http.requests().len(), 1);
    assert_eq!(
        http.requests()[0].headers.get("authorization").unwrap(),
        "Bearer rotated"
    );
}

#[test]
fn key_normalization_preserves_every_strict_descriptor_comparison() {
    use rig_core::effect::FamilyDescriptor;

    for difference in [
        "family",
        "model",
        "capabilities",
        "added-layer",
        "removed-layer",
        "layer-order",
    ] {
        for mode in [RestoreMode::Strict, RestoreMode::Replace] {
            let mut checkpoint = saved(&[KEY]);
            let mut original = checkpoint.requirements().unwrap().remove(0);
            original.layers = vec!["outer".into(), "inner".into()];
            checkpoint.entities[0].get_mut(Bound::type_path()).unwrap()["descriptor"] =
                serde_json::to_value(&original).unwrap();
            let (live, http) = handler("saved", "new", "https://new.invalid/v1");
            let mut advertised = original.clone();
            advertised.key = "different/advertised-key".into();
            match difference {
                "family" => advertised.family = FamilyDescriptor::Memory {},
                "model" => {
                    let FamilyDescriptor::Completion { model, .. } = &mut advertised.family else {
                        panic!("completion descriptor")
                    };
                    *model = "other-model".into();
                }
                "capabilities" => {
                    let FamilyDescriptor::Completion { capabilities, .. } = &mut advertised.family
                    else {
                        panic!("completion descriptor")
                    };
                    capabilities.composes_native_output_with_tools =
                        !capabilities.composes_native_output_with_tools;
                }
                "added-layer" => advertised.layers.push("additional".into()),
                "removed-layer" => {
                    advertised.layers.pop();
                }
                "layer-order" => advertised.layers.reverse(),
                _ => unreachable!(),
            }
            let mut app = bus_support::app();
            let (old, old_http) = handler("saved", "old", "https://old.invalid/v1");
            let entity = Handlers::with(app.world_mut(), |h| h.register_erased(KEY, old))
                .unwrap()
                .unwrap();
            let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
            let count = app.world().entities().len();
            let supplied = ErasedHandler::new(AdvertisedHandler {
                descriptor: advertised.clone(),
                inner: live,
            });
            let result = load_world(&checkpoint, app.world_mut(), mode, [(KEY.into(), supplied)]);
            if mode == RestoreMode::Strict || difference == "family" {
                let error = result.unwrap_err().to_string();
                assert!(
                    error.contains(if difference == "family" {
                        "changed effect family"
                    } else {
                        "original saved descriptor"
                    }),
                    "{difference}: {error}"
                );
                assert_eq!(app.world().entities().len(), count);
                assert_eq!(
                    save_world(app.world_mut()).unwrap().to_json().unwrap(),
                    before,
                    "{difference}"
                );
                dispatch(&mut app);
                assert_eq!(old_http.requests().len(), 1);
                assert!(http.requests().is_empty());
            } else {
                assert!(result.unwrap().entities.contains(&entity));
                advertised.key = KEY.into();
                assert_eq!(descriptor(&mut app, KEY), advertised);
                dispatch(&mut app);
                assert!(old_http.requests().is_empty());
                assert_eq!(http.requests().len(), 1);
            }
        }
    }
}

#[test]
fn strict_restore_checks_original_descriptor_before_aliasing() {
    let checkpoint = saved(&[KEY]);
    let mut app = bus_support::app();
    let (live, _) = handler("different", "live", "https://live.invalid/v1");
    Handlers::with(app.world_mut(), |h| h.register_erased(KEY, live))
        .unwrap()
        .unwrap();
    let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
    let error = load_world(&checkpoint, app.world_mut(), RestoreMode::Strict, []).unwrap_err();
    assert!(error.to_string().contains("original saved descriptor"));
    assert_eq!(
        save_world(app.world_mut()).unwrap().to_json().unwrap(),
        before
    );
    load_world(&checkpoint, app.world_mut(), RestoreMode::Replace, []).unwrap();
    assert_ne!(
        descriptor(&mut app, KEY),
        checkpoint.requirements().unwrap()[0]
    );
}

#[test]
fn intentional_replacement_updates_new_dispatches_not_an_in_flight_operation() {
    use std::sync::{Arc, atomic::Ordering};
    let counters = Arc::new(bus_support::Counters::default());
    counters.hold.hold();
    let mut app = bus_support::app();
    bus_support::register(
        &mut app,
        KEY,
        bus_support::MockModel::saying(&counters, "old-flight"),
    );
    let checkpoint = save_world(app.world_mut()).unwrap();
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new(KEY, bus_support::completion()))
        .id();
    bus_support::tick_until(&mut app, "old operation started", |_| {
        counters.unary_started.load(Ordering::SeqCst) == 1
    });
    let (new, http) = handler("replacement", "new", "https://replacement.invalid/v1");
    load_world(
        &checkpoint,
        app.world_mut(),
        RestoreMode::Replace,
        [(KEY.into(), new)],
    )
    .unwrap();
    assert_ne!(
        descriptor(&mut app, KEY),
        checkpoint.requirements().unwrap()[0]
    );
    counters.hold.release();
    bus_support::tick_until(&mut app, "captured operation finishes", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(
        bus_support::text_of(&app.world().get::<EffectOutcome>(effect).unwrap().0),
        "old-flight"
    );
    dispatch(&mut app);
    assert_eq!(http.requests().len(), 1);
    assert_eq!(counters.unary_started.load(Ordering::SeqCst), 1);
}

#[test]
fn an_incomplete_batch_installs_neither_state_nor_handlers() {
    let checkpoint = saved(&[KEY, "host/other"]);
    let mut app = bus_support::app();
    let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
    let (live, _) = handler("saved", "live", "https://live.invalid/v1");
    let error = load_world(
        &checkpoint,
        app.world_mut(),
        RestoreMode::Strict,
        [(KEY.into(), live)],
    )
    .unwrap_err();
    assert!(error.to_string().contains("no implementation supplied"));
    assert_eq!(
        save_world(app.world_mut()).unwrap().to_json().unwrap(),
        before
    );
    assert!(
        Handlers::with(app.world_mut(), |h| h.keys())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn duplicate_or_extraneous_supplied_keys_are_refused_atomically() {
    for duplicate in [true, false] {
        let checkpoint = saved(&[KEY]);
        let mut app = bus_support::app();
        let (one, _) = handler("saved", "a", "https://one.invalid/v1");
        let (two, _) = handler("saved", "b", "https://two.invalid/v1");
        let other = if duplicate { KEY } else { "host/extra" };
        let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
        assert!(
            load_world(
                &checkpoint,
                app.world_mut(),
                RestoreMode::Strict,
                [(KEY.into(), one), (other.into(), two)]
            )
            .is_err()
        );
        assert_eq!(
            save_world(app.world_mut()).unwrap().to_json().unwrap(),
            before
        );
    }
}

#[test]
fn malformed_state_cannot_replace_a_live_handler() {
    let mut checkpoint = saved(&[KEY]);
    checkpoint.entities.push(serde_json::Map::from_iter([(
        "not.a.component".into(),
        serde_json::Value::Null,
    )]));
    let mut app = bus_support::app();
    let (old, old_http) = handler("saved", "old", "https://old.invalid/v1");
    Handlers::with(app.world_mut(), |h| h.register_erased(KEY, old))
        .unwrap()
        .unwrap();
    let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
    let (new, new_http) = handler("saved", "new", "https://new.invalid/v1");
    assert!(
        load_world(
            &checkpoint,
            app.world_mut(),
            RestoreMode::Strict,
            [(KEY.into(), new)]
        )
        .is_err()
    );
    assert_eq!(
        save_world(app.world_mut()).unwrap().to_json().unwrap(),
        before
    );
    dispatch(&mut app);
    assert_eq!(old_http.requests().len(), 1);
    assert!(new_http.requests().is_empty());
}

#[test]
fn old_provider_bearing_checkpoints_are_refused_not_silently_stripped() {
    // These are frozen historical wire paths, not identities of live types.
    for path in [
        "rig_ecs::bus::binding::ProviderBinding",
        "rig_ecs::bus::binding::CredentialRef",
    ] {
        let mut checkpoint = saved(&[KEY]);
        checkpoint.entities[0].insert(path.into(), serde_json::json!({"provider":"openai:model"}));
        let mut app = bus_support::app();
        let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
        let error = checkpoint.validate(app.world()).unwrap_err().to_string();
        assert!(error.contains(path));
        assert!(error.contains("format-2"));
        assert!(error.contains("migrate provider launch settings to the host"));
        for mode in [RestoreMode::Strict, RestoreMode::Replace] {
            let (live, http) = handler("saved", "new", "https://unused.invalid/v1");
            let error = load_world(&checkpoint, app.world_mut(), mode, [(KEY.into(), live)])
                .unwrap_err()
                .to_string();
            assert!(error.contains(path));
            assert_eq!(
                save_world(app.world_mut()).unwrap().to_json().unwrap(),
                before
            );
            assert!(
                Handlers::with(app.world_mut(), |h| h.keys())
                    .unwrap()
                    .is_empty()
            );
            assert!(http.requests().is_empty());
        }
    }
}

#[test]
fn inconsistent_or_duplicate_saved_keys_are_refused_atomically() {
    for duplicate in [false, true] {
        for mode in [RestoreMode::Strict, RestoreMode::Replace] {
            let mut checkpoint = saved(&[KEY]);
            let expected = if duplicate {
                checkpoint.entities.push(checkpoint.entities[0].clone());
                "duplicate saved"
            } else {
                checkpoint.entities[0].get_mut(Bound::type_path()).unwrap()["descriptor"]["key"] =
                    serde_json::json!("different");
                "descriptor key"
            };
            assert!(
                checkpoint
                    .requirements()
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
            let mut app = bus_support::app();
            let (old, old_http) = handler("saved", "old", "https://old.invalid/v1");
            Handlers::with(app.world_mut(), |h| h.register_erased(KEY, old))
                .unwrap()
                .unwrap();
            let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
            let count = app.world().entities().len();
            assert!(
                checkpoint
                    .validate(app.world())
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
            let (new, new_http) = handler("saved", "new", "https://new.invalid/v1");
            let error =
                load_world(&checkpoint, app.world_mut(), mode, [(KEY.into(), new)]).unwrap_err();
            assert!(error.to_string().contains(expected));
            assert_eq!(app.world().entities().len(), count);
            assert_eq!(
                save_world(app.world_mut()).unwrap().to_json().unwrap(),
                before
            );
            dispatch(&mut app);
            assert_eq!(old_http.requests().len(), 1);
            assert!(new_http.requests().is_empty());
        }
    }
}

#[test]
fn missing_execution_counters_refuse_before_any_installation() {
    for remove_ids in [false, true] {
        let checkpoint = saved(&[KEY]);
        let mut app = bus_support::app();
        if remove_ids {
            app.world_mut().remove_resource::<rig_ecs::bus::IdCounter>();
        } else {
            app.world_mut()
                .remove_resource::<rig_ecs::bus::SeqCounter>();
        }
        let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
        let count = app.world().entities().len();
        let (handler, http) = handler("saved", "new-secret", "https://new.invalid");
        let error = load_world(
            &checkpoint,
            app.world_mut(),
            RestoreMode::Strict,
            [(KEY.into(), handler)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("execution counters"));
        assert_eq!(app.world().entities().len(), count);
        assert_eq!(
            save_world(app.world_mut()).unwrap().to_json().unwrap(),
            before
        );
        assert!(http.requests().is_empty());
    }
}

#[test]
fn open_world_handlers_accept_arbitrary_input_kinds_across_strict_resume() {
    let mut source = bus_support::app();
    let mut destination = bus_support::app();
    for app in [&mut source, &mut destination] {
        Handlers::with(app.world_mut(), |h| {
            h.register_open(
                KEY,
                rig_core::effect::FamilyDescriptor::Custom {
                    kind: "host-dispatch".into(),
                },
            )
        })
        .unwrap()
        .unwrap();
    }
    source
        .world_mut()
        .spawn(PendingEffect::new(KEY, bus_support::completion()));
    let checkpoint = save_world(source.world_mut()).unwrap();
    let loaded = load_world(
        &checkpoint,
        destination.world_mut(),
        RestoreMode::Strict,
        [],
    )
    .unwrap();
    assert_eq!(loaded.with::<PendingEffect>(destination.world()).len(), 1);
    // A supplied completion task cannot change the saved advertised family,
    // even though this open world implementation accepts completion input.
    let before = save_world(destination.world_mut())
        .unwrap()
        .to_json()
        .unwrap();
    for mode in [RestoreMode::Strict, RestoreMode::Replace] {
        let (task, http) = handler("task", "secret", "https://unused.invalid");
        let error = load_world(
            &checkpoint,
            destination.world_mut(),
            mode,
            [(KEY.into(), task)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("changed effect family"));
        assert_eq!(
            save_world(destination.world_mut())
                .unwrap()
                .to_json()
                .unwrap(),
            before
        );
        assert!(http.requests().is_empty());
    }
}

#[test]
fn destination_cannot_invent_an_unfinished_effects_original_contract() {
    let mut source = bus_support::app();
    source
        .world_mut()
        .spawn(PendingEffect::new(KEY, bus_support::completion()));
    let checkpoint = save_world(source.world_mut()).unwrap();
    let mut app = bus_support::app();
    let (handler, http) = handler("saved", "secret", "https://new.invalid");
    Handlers::with(app.world_mut(), |h| h.register_erased(KEY, handler))
        .unwrap()
        .unwrap();
    let before = save_world(app.world_mut()).unwrap().to_json().unwrap();
    for mode in [RestoreMode::Strict, RestoreMode::Replace] {
        let error = load_world(&checkpoint, app.world_mut(), mode, []).unwrap_err();
        assert!(error.to_string().contains("missing saved handler"));
        assert_eq!(
            save_world(app.world_mut()).unwrap().to_json().unwrap(),
            before
        );
    }
    assert!(http.requests().is_empty());
}
