use bevy_app::App;
use bevy_ecs::prelude::*;
use rig_protocol::{Method, RpcError, codes};
use serde::{Deserialize, Serialize};

use super::{Call, Caller, ChannelAppExt, call};
use crate::server::Principal;

#[derive(Serialize, Deserialize)]
struct Halve {
    n: u32,
}

impl Method for Halve {
    const NAME: &'static str = "x/test/halve";
    type Params = Halve;
    type Result = u32;
}

#[derive(Resource, Default)]
struct Calls(u32);

fn halve(call: In<Call<Halve>>, mut calls: ResMut<Calls>) -> Result<u32, RpcError> {
    calls.0 += 1;
    if call.params.n % 2 == 1 {
        return Err(RpcError::new(codes::INVALID_PARAMS, "odd"));
    }
    Ok(call.params.n / 2)
}

fn owner() -> Caller {
    Caller {
        client: Entity::PLACEHOLDER,
        principal: Principal::Owner,
    }
}

#[test]
fn a_method_runs_as_a_system_and_its_errors_pass_through() {
    let mut app = App::new();
    app.init_resource::<Calls>()
        .add_method::<Halve, _, _>(halve);
    let world = app.world_mut();

    let even = call(world, Halve::NAME, owner(), serde_json::json!({"n": 4}));
    assert_eq!(even.and_then(Result::ok), Some(serde_json::json!(2)));

    let odd = call(world, Halve::NAME, owner(), serde_json::json!({"n": 3}));
    assert_eq!(
        odd.and_then(Result::err).map(|e| e.message),
        Some("odd".to_owned())
    );

    // Params that don't parse never reach the system.
    let bad = call(world, Halve::NAME, owner(), serde_json::json!({"m": 1}));
    assert_eq!(
        bad.and_then(Result::err).map(|e| e.code),
        Some(codes::INVALID_PARAMS)
    );
    assert_eq!(world.resource::<Calls>().0, 2);

    assert!(call(world, "x/test/none", owner(), serde_json::json!({})).is_none());
}

#[derive(Serialize, Deserialize)]
struct Flaky {}

impl Method for Flaky {
    const NAME: &'static str = "x/test/flaky";
    type Params = Flaky;
    type Result = u32;
}

#[allow(clippy::panic_in_result_fn)]
fn flaky(_call: In<Call<Flaky>>, mut calls: ResMut<Calls>) -> Result<u32, RpcError> {
    calls.0 += 1;
    assert!(calls.0 > 1, "the first call fails");
    Ok(calls.0)
}

#[test]
fn a_panicking_method_answers_an_error_and_stays_registered() {
    let mut app = App::new();
    app.init_resource::<Calls>()
        .insert_resource(crate::server::CatchPanics)
        .add_method::<Flaky, _, _>(flaky);
    let world = app.world_mut();
    let first = call(world, Flaky::NAME, owner(), serde_json::json!({}));
    assert_eq!(first.and_then(Result::err).map(|e| e.code), Some(codes::INTERNAL));
    let second = call(world, Flaky::NAME, owner(), serde_json::json!({}));
    assert_eq!(second.and_then(Result::ok), Some(serde_json::json!(2)));
}
