use super::*;
use rig_harness::rig_core::message::ToolChoice;
use serde_json::json;

mod fingerprint;

#[derive(Resource, Default)]
struct FitObservations(Vec<bool>);

fn observe_fit(
    mut calls: Query<(&mut ModelRequest, &Adapted), Added<Admitted>>,
    mut agents: Query<&mut Conversation>,
    mut observations: ResMut<FitObservations>,
) {
    for (_, adapted) in &mut calls {
        let layers = adapted.0.handler.0.descriptor().layers;
        observations.0.push(
            layers
                .iter()
                .any(|layer| layer == "claude-cloak-agent-identity"),
        );
    }
    // Hold the same mutable component access as a compaction system.
    for conversation in &mut agents {
        let _ = conversation.entries().count();
    }
}

#[test]
fn agent_loop_sets_profile_default_before_explicit_edits() -> anyhow::Result<()> {
    use rig_harness::rig_core::test_utils::MockCompletionModel;
    use rig_harness_test_support::{self as support, MemoryStore};
    for explicit in [
        None,
        Some(CacheRetention::None),
        Some(CacheRetention::Short),
        Some(CacheRetention::Long),
    ] {
        let model = MockCompletionModel::from_stream_turns([
            support::calls("c1", "unavailable", json!({})),
            support::reply("one"),
            support::reply("two"),
        ]);
        let store = MemoryStore::default();
        let (mut app, wakes) = support::app_on(&store);
        app.insert_resource(support::scripted_models([("claude-cloak/test", &model)]))
            .init_resource::<FitObservations>()
            .add_systems(AgentPass, observe_fit.in_set(PrepareSystems::Fit))
            .add_plugins(AgentPlugin)
            .register_required_components::<Agent, FingerprintSeed>()
            .add_systems(
                AgentPass,
                (
                    default_cache.in_set(PrepareSystems::Input),
                    identify_requests.in_set(PrepareSystems::Adapt),
                ),
            )
            .add_systems(
                AgentPass,
                (move |mut calls: Query<&mut ModelRequest, Added<Admitted>>| {
                    if let Some(retention) = explicit {
                        for mut call in &mut calls {
                            call.request_mut().options.cache = Some(retention);
                        }
                    }
                })
                .in_set(PrepareSystems::Edit),
            );
        app.finish();
        app.update();
        let agent = support::first_agent(app.world_mut())
            .ok_or_else(|| anyhow::anyhow!("missing agent"))?;
        app.world_mut()
            .entity_mut(agent)
            .insert((OAuthSession::default(), HumanFingerprint::default()));
        app.world_mut().trigger(SetModel {
            entity: agent,
            model: "claude-cloak/test".into(),
            kind: ChoiceKind::ForThisRun,
        });
        for (index, text) in ["first human", "second human"].into_iter().enumerate() {
            if index == 1 {
                app.world_mut()
                    .entity_mut(agent)
                    .insert(Conversation::default());
            }
            app.world_mut()
                .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            support::run_until(&mut app, &wakes, |world| {
                model.request_count() >= index + 2 && world.get::<ActiveTurn>(agent).is_none()
            });
        }
        let requests = model.requests();
        anyhow::ensure!(requests.len() == 3);
        let observed = &app.world().resource::<FitObservations>().0;
        anyhow::ensure!(observed.len() == 3 && observed.iter().all(|ready| *ready));
        for request in requests {
            anyhow::ensure!(
                request.options.cache == Some(explicit.unwrap_or(crate::profile::CACHE))
            );
            anyhow::ensure!(
                request
                    .additional_params
                    .as_ref()
                    .and_then(|params| params.pointer("/_omp/human_suffix"))
                    .and_then(Value::as_str)
                    == Some(crate::profile::suffix("first human").as_str())
            );
        }
    }
    Ok(())
}

#[test]
fn started_fallback_recovers_omission_without_overriding_explicit_short() -> anyhow::Result<()> {
    use rig_harness::rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
    use rig_harness_test_support::{self as support, MemoryStore};
    for (explicit, edit) in [(false, false), (true, false), (false, true)] {
        let first = MockCompletionModel::from_stream_turns([vec![MockStreamEvent::Error(
            MockError::provider("failed"),
        )]]);
        let next = MockCompletionModel::from_stream_turns([support::reply("fallback")]);
        let mut destination =
            support::connect(&next).ok_or_else(|| anyhow::anyhow!("missing destination"))?;
        let mut spec = rig_harness::rig_core::catalog::Catalog::builtin()
            .resolve("anthropic/claude-sonnet-4-6")?
            .shared()
            .as_ref()
            .clone();
        spec.provider =
            rig_harness::rig_core::providers::registry::ProviderId::declare("claude-cloak")?;
        destination.spec = Arc::new(spec);
        let (mut app, wakes) = support::app_on(&MemoryStore::default());
        app.insert_resource(support::scripted_models([("claude-cloak/test", &first)]))
            .add_plugins(AgentPlugin)
            .register_required_components::<Agent, FingerprintSeed>()
            .register_required_components::<Agent, OAuthSession>()
            .register_required_components::<Agent, HumanFingerprint>()
            .add_systems(
                AgentPass,
                (
                    default_cache.in_set(PrepareSystems::Input),
                    identify_requests.in_set(PrepareSystems::Adapt),
                    (move |failed: Query<
                        (Entity, &ModelRequest, &Outcome, &CallOf, &ForAgent),
                        (With<Started>, Without<SupersededBy>),
                    >,
                           mut commands: Commands| {
                        for (entity, request, outcome, turn, agent) in &failed {
                            if outcome.0.is_ok() {
                                continue;
                            }
                            let mut fallback = request.clone().on(destination.clone());
                            if explicit {
                                fallback.request_mut().options.cache = Some(CacheRetention::Short);
                            }
                            commands
                                .spawn((fallback, TurnCall, CallOf(turn.0), ForAgent(agent.0)))
                                .add_related::<SupersededBy>(&[entity]);
                        }
                    })
                    .in_set(ReviewSystems::Edit),
                    (move |mut calls: Query<
                        &mut ModelRequest,
                        (Added<Admitted>, With<Supersedes>),
                    >| {
                        if edit {
                            for mut call in &mut calls {
                                call.request_mut().options.cache = Some(CacheRetention::Short);
                            }
                        }
                    })
                    .in_set(PrepareSystems::Edit),
                ),
            );
        app.finish();
        app.update();
        let agent = support::first_agent(app.world_mut())
            .ok_or_else(|| anyhow::anyhow!("missing agent"))?;
        app.world_mut().trigger(SetModel {
            entity: agent,
            model: "claude-cloak/test".into(),
            kind: ChoiceKind::ForThisRun,
        });
        app.world_mut()
            .trigger(Deliver::new(agent, "human", DeliveryMode::Steer));
        support::run_until(&mut app, &wakes, |world| {
            next.request_count() == 1 && world.get::<ActiveTurn>(agent).is_none()
        });
        let requests = next.requests();
        let request = requests
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing fallback request"))?;
        anyhow::ensure!(
            request.options.cache
                == Some(if explicit || edit {
                    CacheRetention::Short
                } else {
                    CacheRetention::Long
                })
        );
    }
    Ok(())
}

#[derive(Resource, Default)]
struct PreparedRequests(Vec<CompletionRequest>);

fn capture_prepared(
    calls: Query<(&ModelRequest, &Adapted), Added<Admitted>>,
    mut prepared: ResMut<PreparedRequests>,
) {
    for (model, adapted) in &calls {
        prepared.0.push(adapted.sent(model));
    }
}

#[test]
fn routing_and_cloned_fallbacks_do_not_carry_private_policy() -> anyhow::Result<()> {
    use rig_harness::rig_core::test_utils::MockCompletionModel;
    use rig_harness_test_support::{self as support, MemoryStore};
    for (toward_cloak, explicit) in [
        (false, None),
        (false, Some(CacheRetention::Long)),
        (true, None),
        (true, Some(CacheRetention::Short)),
    ] {
        let model = MockCompletionModel::from_stream_turns([support::reply("routed")]);
        let mut destination =
            support::connect(&model).ok_or_else(|| anyhow::anyhow!("missing destination"))?;
        destination.spec = rig_harness::rig_core::catalog::Catalog::builtin()
            .resolve("anthropic/claude-sonnet-4-6")?
            .shared();
        let source = if toward_cloak {
            "deepseek/test"
        } else {
            "claude-cloak/test"
        };
        if toward_cloak {
            let mut spec = (*destination.spec).clone();
            spec.provider =
                rig_harness::rig_core::providers::registry::ProviderId::declare("claude-cloak")?;
            destination.spec = Arc::new(spec);
        }
        let (mut app, wakes) = support::app_on(&MemoryStore::default());
        app.insert_resource(support::scripted_models([(source, &model)]))
            .init_resource::<PreparedRequests>()
            .add_systems(AgentPass, capture_prepared.in_set(Stage::Gate))
            .add_plugins(AgentPlugin)
            .register_required_components::<Agent, FingerprintSeed>()
            .register_required_components::<Agent, OAuthSession>()
            .register_required_components::<Agent, HumanFingerprint>()
            .add_systems(
                AgentPass,
                (
                    default_cache.in_set(PrepareSystems::Input),
                    identify_requests.in_set(PrepareSystems::Adapt),
                    (move |mut calls: Query<&mut ModelRequest, Added<Admitted>>| {
                        for mut call in &mut calls {
                            let mut fallback = call.clone();
                            fallback.connection = Some(destination.clone());
                            fallback.request_mut().options.cache = explicit;
                            *call = fallback;
                        }
                    })
                    .in_set(PrepareSystems::Edit),
                ),
            );
        app.finish();
        app.update();
        let agent = support::first_agent(app.world_mut())
            .ok_or_else(|| anyhow::anyhow!("missing agent"))?;
        app.world_mut().trigger(SetModel {
            entity: agent,
            model: source.into(),
            kind: ChoiceKind::ForThisRun,
        });
        app.world_mut()
            .trigger(Deliver::new(agent, "human", DeliveryMode::Steer));
        support::run_until(&mut app, &wakes, |world| {
            model.request_count() == 1 && world.get::<ActiveTurn>(agent).is_none()
        });
        let requests = model.requests();
        let request = requests
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing request"))?;
        let prepared = app.world().resource::<PreparedRequests>();
        anyhow::ensure!(!prepared.0.is_empty());
        anyhow::ensure!(prepared.0.iter().all(|request| {
            request
                .additional_params
                .as_ref()
                .and_then(|params| params.get("_omp"))
                .is_none()
        }));
        let default = if toward_cloak {
            CacheRetention::Long
        } else {
            CacheRetention::Short
        };
        anyhow::ensure!(
            request.options.cache == Some(explicit.unwrap_or(default)),
            "toward={toward_cloak} explicit={explicit:?} actual={:?}",
            request.options.cache
        );
        anyhow::ensure!(
            request
                .additional_params
                .as_ref()
                .and_then(|params| params.get("_omp"))
                .is_some()
                == toward_cloak
        );
    }
    Ok(())
}

#[test]
fn saved_fingerprints_restore_and_forks_inherit_without_crossing_agents() -> anyhow::Result<()> {
    use rig_harness_test_support::{self as support, MemoryStore};
    let store = MemoryStore::default();
    let model = rig_harness::rig_core::test_utils::MockCompletionModel::from_stream_turns([
        support::reply("child"),
    ]);
    let setup = || {
        let (mut app, wakes) = support::app_on(&store);
        app.insert_resource(support::scripted_models([("claude-cloak/test", &model)]))
            .add_systems(
                AgentPass,
                (
                    default_cache.in_set(PrepareSystems::Input),
                    identify_requests.in_set(PrepareSystems::Adapt),
                ),
            );
        app.add_plugins((AgentPlugin, SessionLogPlugin))
            .register_required_components::<Agent, FingerprintSeed>()
            .register_required_components::<Agent, HumanFingerprint>()
            .keeps_saved::<HumanFingerprint>()
            .keeps_saved::<FingerprintSeed>()
            .register_required_components::<Agent, OAuthSession>()
            .keeps_saved::<OAuthSession>()
            .add_observer(fork_fingerprint);
        app.finish();
        app.update();
        (app, wakes)
    };
    let (mut app, wakes) = setup();
    let first =
        support::first_agent(app.world_mut()).ok_or_else(|| anyhow::anyhow!("missing agent"))?;
    let initial = crate::profile::suffix("first human original");
    let other = crate::profile::suffix("second agent distinct");
    anyhow::ensure!(initial != other);
    let (before_human, human) = {
        use rig_harness::rig_core::conversation::{Author, Entry};
        let mut conversation = app
            .world_mut()
            .get_mut::<Conversation>(first)
            .ok_or_else(|| anyhow::anyhow!("missing conversation"))?;
        let before = conversation.push(Entry::new(Message::user("environment")).by(Author::rig()));
        let human = conversation.push(Message::user("first human original"));
        (before, human)
    };
    app.world_mut().entity_mut(first).insert((
        HumanFingerprint(Some(initial.clone()), Some(human.get())),
        FingerprintSeed {
            units: Some(crate::profile::seed("first human original")),
            version: crate::profile::VERSION.into(),
        },
    ));
    let second = app
        .world_mut()
        .spawn((
            Agent,
            HumanFingerprint(Some(other.clone()), Some(0)),
            FingerprintSeed {
                units: Some(crate::profile::seed("second agent distinct")),
                version: crate::profile::VERSION.into(),
            },
        ))
        .id();
    support::commit_message(
        app.world_mut(),
        second,
        Message::user("second agent distinct"),
    )
    .map_err(anyhow::Error::msg)?;
    app.update();
    app.world_mut().trigger(ForkAgent {
        agent: first,
        through: None,
    });
    app.update();
    for through in [before_human, human] {
        app.world_mut().trigger(ForkAgent {
            agent: first,
            through: Some(through),
        });
        app.update();
    }
    let mut fingerprints = app.world_mut().query::<&HumanFingerprint>();
    anyhow::ensure!(
        fingerprints
            .iter(app.world())
            .filter(|value| value.0.is_none())
            .count()
            == 2
    );
    let mut children = app.world_mut().query::<(Entity, &HumanFingerprint)>();
    let child = children
        .iter(app.world())
        .find(|(_, fingerprint)| fingerprint.0.is_none())
        .map(|(entity, _)| entity)
        .ok_or_else(|| anyhow::anyhow!("missing prefix fork"))?;
    app.world_mut().trigger(SetModel {
        entity: child,
        model: "claude-cloak/test".into(),
        kind: ChoiceKind::ForThisRun,
    });
    app.world_mut().trigger(Deliver::new(
        child,
        "new child human prompt",
        DeliveryMode::Steer,
    ));
    support::run_until(&mut app, &wakes, |world| {
        model.request_count() == 1 && world.get::<ActiveTurn>(child).is_none()
    });
    let child_suffix = crate::profile::suffix("new child human prompt");
    anyhow::ensure!(
        app.world()
            .get::<HumanFingerprint>(child)
            .and_then(|value| value.0.as_ref())
            == Some(&child_suffix)
    );
    anyhow::ensure!(child_suffix != initial);
    anyhow::ensure!(
        app.world()
            .get::<FingerprintSeed>(child)
            .is_some_and(|seed| {
                seed.units == Some(crate::profile::seed("new child human prompt"))
                    && seed.version == crate::profile::VERSION
            })
    );
    let mut sessions = app.world_mut().query::<&OAuthSession>();
    let mut before: Vec<_> = sessions
        .iter(app.world())
        .map(|session| session.0.clone())
        .collect();
    before.sort();
    before.dedup();
    anyhow::ensure!(before.len() == 5);
    drop(app);
    let (mut restored, _) = setup();
    let mut sessions = restored.world_mut().query::<&OAuthSession>();
    let mut after: Vec<_> = sessions
        .iter(restored.world())
        .map(|session| session.0.clone())
        .collect();
    after.sort();
    anyhow::ensure!(before == after);
    let mut query = restored.world_mut().query::<&HumanFingerprint>();
    let fingerprints: Vec<_> = query
        .iter(restored.world())
        .filter_map(|value| value.0.clone())
        .collect();
    anyhow::ensure!(
        fingerprints
            .iter()
            .filter(|value| **value == initial)
            .count()
            == 2
    );
    anyhow::ensure!(fingerprints.iter().filter(|value| **value == other).count() == 1);
    anyhow::ensure!(
        fingerprints
            .iter()
            .filter(|value| **value == child_suffix)
            .count()
            == 1
    );
    let mut seeds = restored.world_mut().query::<&FingerprintSeed>();
    anyhow::ensure!(
        seeds
            .iter(restored.world())
            .filter(|seed| {
                seed.units == Some(crate::profile::seed("first human original"))
                    && seed.version == crate::profile::VERSION
            })
            .count()
            == 2
    );
    anyhow::ensure!(
        seeds
            .iter(restored.world())
            .filter(|seed| {
                seed.units == Some(crate::profile::seed("new child human prompt"))
                    && seed.version == crate::profile::VERSION
            })
            .count()
            == 1
    );
    Ok(())
}

#[test]
fn human_provenance_uses_authorship_and_survives_restore() -> anyhow::Result<()> {
    use rig_harness::rig_core::conversation::{Author, Conversation, Entry};
    let mut conversation = Conversation::default();
    conversation.push(Entry::new(Message::user("injected context")).by(Author::plugin("context")));
    conversation.push(Message::user("first human"));
    conversation.push(Message::assistant("answer"));
    conversation.push(Message::user("second human"));
    conversation.push(Entry::new(Message::user("side call")).by(Author::rig()));
    anyhow::ensure!(first_human_text(&conversation) == "first human");
    let restored: Conversation = serde_json::from_value(serde_json::to_value(&conversation)?)?;
    anyhow::ensure!(first_human_text(&restored) == "first human");
    let empty_first = Conversation::from_messages([Message::user(""), Message::user("later text")]);
    anyhow::ensure!(first_human_text(&empty_first).is_empty());
    let image_first = Conversation::from_messages([
        Message::User {
            content: vec![UserContent::Image(rig_harness::rig_core::message::Image {
                data: rig_harness::rig_core::message::DocumentSourceKind::Unknown,
                media_type: None,
                detail: None,
                native: None,
            })],
        },
        Message::user("later text"),
    ]);
    anyhow::ensure!(first_human_text(&image_first).is_empty());
    let other = Conversation::from_messages([Message::user("another agent")]);
    anyhow::ensure!(first_human_text(&other) == "another agent");
    Ok(())
}

#[tokio::test]
async fn unauthored_human_identity_keeps_wire_fallback_available() -> anyhow::Result<()> {
    use rig_harness::rig_core::conversation::{Author, Conversation, Entry};
    let mut conversation = Conversation::default();
    conversation.push(
        Entry::new(Message::user("delegated instructions")).by(Author::agent("parent", None)),
    );
    anyhow::ensure!(!conversation.entries().any(|entry| entry.is_users_own()));
    for suffix in [None, Some(crate::profile::suffix(""))] {
        let identity = AgentIdentity {
            session: "session".into(),
            suffix: suffix.clone(),
            default_cache: false,
        };
        let kind = EffectKind::Completion {
            request: Arc::new(CompletionRequest::new("delegated instructions")),
            stream: true,
        };
        let BeforeServe::Patch(EffectKind::Completion { request, .. }) =
            identity.before(EffectId::from_raw(0), &kind).await
        else {
            anyhow::bail!("identity should patch completion");
        };
        let actual = request
            .additional_params
            .as_ref()
            .and_then(|params| params.pointer("/_omp/human_suffix"))
            .and_then(Value::as_str);
        anyhow::ensure!(actual == suffix.as_deref());
    }
    Ok(())
}

#[test]
fn implicit_and_explicit_auto_remain_distinguishable_after_encoding() -> anyhow::Result<()> {
    let mut implicit = CompletionRequest::new("hello");
    let mut explicit = implicit.clone();
    explicit.tool_choice = Some(ToolChoice::Auto);
    anyhow::ensure!(annotate(&mut implicit));
    anyhow::ensure!(annotate(&mut explicit));
    anyhow::ensure!(
        (implicit
            .additional_params
            .as_ref()
            .and_then(|v| v.pointer("/_omp/tool_choice_explicit")))
            == (Some(&json!(false)))
    );
    anyhow::ensure!(
        (explicit
            .additional_params
            .as_ref()
            .and_then(|v| v.pointer("/_omp/tool_choice_explicit")))
            == (Some(&json!(true)))
    );
    Ok(())
}

#[test]
fn cache_policy_preserves_explicit_retention_and_session() -> anyhow::Result<()> {
    for (retention, expected) in [
        (None, "long"),
        (Some(CacheRetention::None), "none"),
        (Some(CacheRetention::Short), "short"),
        (Some(CacheRetention::Long), "long"),
    ] {
        let mut request = CompletionRequest::new("hello");
        request.options.cache = retention;
        request.additional_params =
            Some(json!({"_omp":{"session_id":"session-fixture"},"temperature":0.4}));
        anyhow::ensure!(annotate(&mut request));
        let Some(params) = request.additional_params else {
            anyhow::bail!("annotation supplies parameters")
        };
        anyhow::ensure!((params.pointer("/_omp/cache_retention")) == (Some(&json!(expected))));
        anyhow::ensure!((params.pointer("/_omp/session_id")) == (Some(&json!("session-fixture"))));
        anyhow::ensure!((params.get("temperature")) == (Some(&json!(0.4))));
    }
    Ok(())
}

#[test]
fn raw_tool_choice_is_explicit_and_invalid_parameters_are_rejected() -> anyhow::Result<()> {
    let mut request = CompletionRequest::new("hello");
    request.additional_params = Some(json!({"tool_choice":{"type":"auto"}}));
    anyhow::ensure!(annotate(&mut request));
    anyhow::ensure!(
        (request
            .additional_params
            .as_ref()
            .and_then(|v| v.pointer("/_omp/tool_choice_explicit")))
            == (Some(&json!(true)))
    );
    request.additional_params = Some(json!([1, 2]));
    anyhow::ensure!(!annotate(&mut request));
    Ok(())
}

#[test]
fn thinking_intent_survives_forced_tools_and_raw_overrides_win() -> anyhow::Result<()> {
    let mut request = CompletionRequest::new("hello");
    request.tool_choice = Some(ToolChoice::Required);
    request.options.reasoning = Some(Reasoning::Budget { tokens: 1024 });
    anyhow::ensure!(annotate(&mut request));
    anyhow::ensure!(
        (request
            .additional_params
            .as_ref()
            .and_then(|p| p.pointer("/_omp/thinking_requested")))
            == (Some(&json!(true)))
    );
    request.additional_params = Some(json!({"thinking":{"type":"disabled"}}));
    anyhow::ensure!(annotate(&mut request));
    anyhow::ensure!(
        (request
            .additional_params
            .as_ref()
            .and_then(|p| p.pointer("/_omp/thinking_requested")))
            == (Some(&json!(false)))
    );
    Ok(())
}
