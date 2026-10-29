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
        // The client leaves while a session's tool and MCP server run.
        let stays = vec![acp::EnvVariable::new(STAYS, "1")];
        let third = driven.new_session(vec![stub_with("files", stays)]).await;
        assert!(third.is_some(), "no session");
        let Some(third) = third else { return };
        assert!(driven.model(&third.session_id, "test/wait").await);
        let working = tokio::spawn(driven.send(&third.session_id, "Wait."));
        assert!(driven.until(|_| sleeping(place) && serving(place)).await);
        drop(working);
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
        let Some(router) = pid else { return };
        let sent = std::process::Command::new("kill")
            .args(["-TERM", &router.to_string()])
            .status();
        assert!(sent.is_ok_and(|sent| sent.success()));
        // It ends, its session and their work with it, while the client
        // still holds its connection.
        let deadline = Instant::now() + Duration::from_secs(20);
        while (alive(router) || !processes_of(place).is_empty()) && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(!alive(router) && processes_of(place).is_empty());
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
        // A process stopped by a signal reads and answers nothing, and a
        // prompt larger than its pipe holds fills it.
        let stopped = std::process::Command::new("kill")
            .args(["-STOP", &pid.to_string()])
            .status();
        assert!(stopped.is_ok_and(|stopped| stopped.success()));
        let large = "x".repeat(1 << 20);
        let filling = tokio::spawn(driven.send(&id, &large));
        let close = driven
            .cx
            .send_request(acp::CloseSessionRequest::new(id.clone()));
        let closed = tokio::time::timeout(Duration::from_secs(20), close.block_task()).await;
        assert!(closed.is_ok_and(|closed| closed.is_ok()));
        assert!(eventually(|| process_of(place, &id).is_none()));
        drop(filling);
    });
    assert!(router.is_some());
}

#[test]
fn a_late_answer_for_a_killed_session_leaves_the_others_working() {
    let place = Place::new("late");
    let place = &place;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let (first, second) = (
            driven.new_session(Vec::new()).await,
            driven.new_session(Vec::new()).await,
        );
        assert!(first.is_some() && second.is_some(), "no sessions");
        let (Some(first), Some(second)) = (first, second) else {
            return;
        };
        let (first, second) = (first.session_id, second.session_id);
        driven.answering(Answering::Later);
        assert!(driven.model(&first, "test/guarded").await);
        let asking = tokio::spawn(driven.send(&first, "Read it."));
        assert!(driven.until(|seen| !seen.later.is_empty()).await);
        let pid = process_of(place, &first);
        assert!(pid.is_some(), "no process runs {first}");
        let Some(pid) = pid else { return };
        std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .ok();
        assert!(asking.await.ok().is_some_and(|answer| answer.is_err()));
        // The editor answers the dead session's question anyway.
        let late = driven.seen().later.pop();
        let allow = acp::SelectedPermissionOutcome::new(rig_basics::approval::ALLOW);
        let answer =
            acp::RequestPermissionResponse::new(acp::RequestPermissionOutcome::Selected(allow));
        late.map(|late| late.respond(answer));
        assert!(driven.model(&second, "test/reply").await);
        assert_eq!(
            driven.prompt(&second, "Hello.").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
    });
    assert!(router.is_some());
}
