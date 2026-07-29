#![cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]

use rig::wasm_compat::{
    BoxFuture as FacadeBoxFuture, MaybeSend as FacadeMaybeSend, MaybeSync as FacadeMaybeSync,
};
use rig_core::wasm_compat::{BoxFuture, MaybeSend, MaybeSync};

fn core_markers_imply_thread_safety<T: MaybeSend + MaybeSync>() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<T>();
}

fn facade_markers_imply_thread_safety<T: FacadeMaybeSend + FacadeMaybeSync>() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<T>();
}

#[test]
fn native_core_and_facade_exports_are_sendable() {
    fn assert_send<T: Send>() {}

    core_markers_imply_thread_safety::<String>();
    facade_markers_imply_thread_safety::<String>();
    assert_send::<BoxFuture<'static, ()>>();
    assert_send::<FacadeBoxFuture<'static, ()>>();
}
