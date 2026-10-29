//! What runs and what is left: a session process killed while its tool
//! runs leaves nothing and the others go on; the router's end, by EOF or a
//! signal, ends every session and its tools. Linux, which shows each
//! process's environment.

use super::*;

/// The process running `session`.
fn process_of(place: &Place, session: &acp::SessionId) -> Option<u32> {
    let mark = format!("RIG_SESSION={session}");
    processes_of(place).into_iter().find(|pid| {
        std::fs::read(format!("/proc/{pid}/environ")).is_ok_and(|environ| {
            environ
                .split(|byte| *byte == 0)
                .any(|entry| entry == mark.as_bytes())
        })
    })
}

/// Whether an MCP server of `place` runs.
fn serving(place: &Place) -> bool {
    processes_of(place).into_iter().any(|pid| {
        std::fs::read(format!("/proc/{pid}/cmdline"))
            .is_ok_and(|line| String::from_utf8_lossy(&line).contains("mcp_stub"))
    })
}

#[test]
fn a_killed_session_leaves_no_tool_and_the_others_go_on_until_the_client_leaves() {
    let place = Place::new("killed");
    let place = &place;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        // The first's MCP server outlives the end of its input.
        let stays = vec![acp::EnvVariable::new(STAYS, "1")];
        let (first, second) = (
            driven.new_session(vec![stub_with("files", stays)]).await,
            driven.new_session(Vec::new()).await,
        );
        assert!(first.is_some() && second.is_some(), "no sessions");
        let (Some(first), Some(second)) = (first, second) else {
            return;
        };
        let (first, second) = (first.session_id, second.session_id);
        assert!(driven.model(&first, "test/wait").await);
        let request = acp::PromptRequest::new(first.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        assert!(driven.until(|_| sleeping(place) && serving(place)).await);
        let pid = process_of(place, &first);
        assert!(pid.is_some(), "no process runs {first}");
        let Some(pid) = pid else { return };
        std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .ok();
        assert!(prompt.block_task().await.is_err());
        assert!(eventually(|| !sleeping(place) && !serving(place)));
        assert!(driven.model(&second, "test/reply").await);
        assert_eq!(
            driven.prompt(&second, "Hello.").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        let prompt = acp::PromptRequest::new(first.clone(), vec!["Hello.".into()]);
        let gone = driven.cx.send_request(prompt).block_task().await;
        let code = gone.err().map(|error| i32::from(error.code));
        assert_eq!(code, Some(-32002), "a session not open is not found");
    });
    assert!(router.is_some(), "rig acp did not run");
    let Some(mut router) = router else { return };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let status = runtime
        .ok()
        .and_then(|runtime| runtime.block_on(router.process.wait()).ok());
    assert!(status.is_some_and(|status| status.success()));
    assert!(eventually(|| processes_of(place).is_empty()));
}

#[test]
fn sigterm_ends_the_router_its_sessions_and_their_tools() {
    let place = Place::new("sigterm");
    let place = &place;
    let mut pid = None;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let stays = vec![acp::EnvVariable::new(STAYS, "1")];
        let new = driven.new_session(vec![stub_with("files", stays)]).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/wait").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        assert!(driven.until(|_| sleeping(place) && serving(place)).await);
        pid = driven.router;
        if let Some(pid) = pid {
            std::process::Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .status()
                .ok();
        }
        // The router ends the session, which answers the prompt or closes.
        prompt.block_task().await.ok();
    });
    assert!(pid.is_some() && router.is_some());
    assert!(eventually(|| processes_of(place).is_empty()));
}

#[test]
fn a_stopped_session_process_closes_within_its_grace() {
    let place = Place::new("stopped");
    let place = &place;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        let pid = process_of(place, &id);
        assert!(pid.is_some(), "no process runs {id}");
        let Some(pid) = pid else { return };
        // A process stopped by a signal reads and answers nothing.
        std::process::Command::new("kill")
            .args(["-STOP", &pid.to_string()])
            .status()
            .ok();
        let close = driven
            .cx
            .send_request(acp::CloseSessionRequest::new(id.clone()));
        let closed = tokio::time::timeout(Duration::from_secs(20), close.block_task()).await;
        assert!(closed.is_ok_and(|closed| closed.is_ok()));
        assert!(eventually(|| process_of(place, &id).is_none()));
    });
    assert!(router.is_some());
}
