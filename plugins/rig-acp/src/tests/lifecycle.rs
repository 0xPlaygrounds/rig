//! What runs and what is left: a session process killed while its tool
//! runs leaves nothing and the others go on; the router's end, by EOF or a
//! signal, ends every session and its tools. Linux, which shows each
//! process's environment.

#![allow(clippy::disallowed_methods, reason = "test fixtures")]

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
fn a_loaded_session_shows_each_call_under_its_live_id() {
    let place = Place::new("ids");
    let place = &place;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        driven.answering(Answering::Allow);
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        // Two calls the provider calls `r1`.
        for (model, prompt) in [
            ("test/read", "Read the fixture."),
            ("test/again", "Read the missing file."),
        ] {
            assert!(driven.model(&id, model).await);
            assert!(driven.prompt(&id, prompt).await.is_ok());
        }
        // A subagent's call, shown by the permission it asks.
        assert!(driven.model(&id, "test/delegate").await);
        assert!(driven.prompt(&id, "Delegate the read.").await.is_ok());
        assert!(driven.until(|seen| !seen.asked.is_empty()).await);
        let asked = driven
            .seen()
            .asked
            .first()
            .map(|call| call.tool_call_id.to_string());
        let asked = asked.unwrap_or_default();
        let ended = |seen: &Seen| {
            seen.events.iter().any(|event| {
                matches!(event, Event::Update(_, acp::SessionUpdate::ToolCallUpdate(update))
                    if update.tool_call_id.to_string() == asked
                        && update.fields.status == Some(acp::ToolCallStatus::Completed))
            })
        };
        assert!(driven.until(ended).await, "{:?}", tool_steps(driven, &id));
        // Its report ends a turn of the main agent too.
        let noted = |seen: &Seen| {
            seen.events.iter().any(|event| {
                matches!(event, Event::Update(_, acp::SessionUpdate::AgentMessageChunk(chunk))
                    if text(&chunk.content).is_some_and(|said| said.contains("noted")))
            })
        };
        assert!(driven.until(noted).await, "{}", driven.said(&id));
        // A call the session's process is killed in.
        assert!(driven.model(&id, "test/wait").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Wait.".into()]);
        let announced = |from: usize| -> Vec<String> {
            let updates = driven.updates(&id).into_iter().skip(from);
            let calls = updates.filter_map(|update| match update {
                acp::SessionUpdate::ToolCall(call) => Some(call.tool_call_id.to_string()),
                _ => None,
            });
            calls.collect()
        };
        let shown = announced(0).len();
        let waiting = driven.cx.send_request(request);
        // Its call runs, and the client was told of it.
        let told = |seen: &Seen| {
            let calls = seen.events.iter().filter(|event| {
                matches!(event, Event::Update(of, acp::SessionUpdate::ToolCall(_)) if *of == id)
            });
            calls.count() > shown
        };
        assert!(driven.until(|seen| told(seen) && sleeping(place)).await);
        let live = announced(0);
        let unique: std::collections::HashSet<_> = live.iter().collect();
        assert!(
            unique.len() == live.len() && !unique.contains(&asked),
            "{live:?} {asked}"
        );
        let pid = process_of(place, &id);
        assert!(pid.is_some(), "no process runs {id}");
        let Some(pid) = pid else { return };
        std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .ok();
        assert!(waiting.block_task().await.is_err());
        let before = driven.updates(&id).len();
        // Loaded with a server slow to start: the turn the session carries
        // on calls `w1` again before the session opens.
        let slow = stub_with("files", vec![acp::EnvVariable::new(SLOW, "1500")]);
        let mut loaded = false;
        for _ in 0..100 {
            let load = acp::LoadSessionRequest::new(id.clone(), &driven.project)
                .mcp_servers(vec![slow.clone()]);
            match driven.cx.send_request(load).block_task().await {
                Ok(_) => loaded = true,
                // Until the router cleaned up after the killed process.
                Err(error) if error.message.contains("already open") => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
                Err(_) => {}
            }
            break;
        }
        assert!(loaded);
        let again = |seen: &Seen| {
            let calls = seen.events.iter().filter(|event| {
                matches!(event, Event::Update(of, acp::SessionUpdate::ToolCall(_)) if *of == id)
            });
            calls.count() > live.len() * 2
        };
        assert!(driven.until(again).await, "{:?}", tool_steps(driven, &id));
        let interrupted = live.last().cloned().unwrap_or_default();
        let failed = |seen: &Seen| {
            seen.events.iter().any(|event| {
                matches!(event, Event::Update(_, acp::SessionUpdate::ToolCallUpdate(update))
                    if update.tool_call_id.to_string() == interrupted
                        && update.fields.status == Some(acp::ToolCallStatus::Failed))
            })
        };
        assert!(driven.until(failed).await, "{:?}", tool_steps(driven, &id));
        let close = acp::CloseSessionRequest::new(id.clone());
        assert!(driven.cx.send_request(close).block_task().await.is_ok());
        // Counted once the close answered, after every update of the session:
        // the history first, then each call once, the interrupted one failed
        // once.
        let failures = driven.updates(&id).into_iter().filter(|update| {
            matches!(update, acp::SessionUpdate::ToolCallUpdate(update)
                if update.tool_call_id.to_string() == interrupted
                    && update.fields.status == Some(acp::ToolCallStatus::Failed))
        });
        assert_eq!(failures.count(), 1);
        let replayed = announced(before);
        assert_eq!(replayed.get(..live.len()), Some(live.as_slice()));
        let unique: std::collections::HashSet<_> = replayed.iter().collect();
        assert_eq!(unique.len(), replayed.len(), "{replayed:?}");
    });
    assert!(router.is_some());
}

/// SIGTERM ends the router, its sessions and their tools, and first
/// answers the open prompt `cancelled`, after its tool's closing update.
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
        let prompt = driven.send(&id, "Wait.");
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
        let answer = prompt.await;
        assert!(
            matches!(answer, Ok(acp::StopReason::Cancelled)),
            "{answer:?}"
        );
        let steps = tool_steps(driven, &id);
        let closed = steps.last().is_some_and(|(_, status)| {
            matches!(
                status,
                acp::ToolCallStatus::Failed | acp::ToolCallStatus::Completed
            )
        });
        assert!(closed, "{steps:?}");
        // The closing update comes before the answer.
        let seen = driven.seen();
        let answered = seen
            .events
            .iter()
            .position(|event| matches!(event, Event::Answered(session) if *session == id));
        let updated = seen.events.iter().rposition(
            |event| matches!(event, Event::Update(session, acp::SessionUpdate::ToolCallUpdate(_)) if *session == id),
        );
        assert!(
            updated
                .zip(answered)
                .is_some_and(|(updated, answered)| updated < answered)
        );
    });
    assert!(pid.is_some() && router.is_some());
    assert!(eventually(|| processes_of(place).is_empty()));
}

/// A line past the limit from the editor, while a prompt runs, ends what
/// the router reads as a closed stdin does: the running prompt is still
/// answered `cancelled`, after its tool's closing update, and the router
/// ends.
#[test]
fn a_line_past_the_limit_ends_the_input_and_the_open_prompt_is_answered() {
    let mut place = Place::new("oversized");
    place.limit = Some(16 * 1024);
    let place = &place;
    let mut pid = None;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/wait").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        assert!(driven.until(|_| sleeping(place)).await);
        pid = driven.router;
        let large = acp::PromptRequest::new(id.clone(), vec!["x".repeat(32 * 1024).into()]);
        let refused = driven.cx.send_request(large);
        let answer = prompt.block_task().await.map(|answer| answer.stop_reason);
        assert!(
            matches!(answer, Ok(acp::StopReason::Cancelled)),
            "{answer:?}"
        );
        let steps = tool_steps(driven, &id);
        let closed = steps.last().is_some_and(|(_, status)| {
            matches!(
                status,
                acp::ToolCallStatus::Failed | acp::ToolCallStatus::Completed
            )
        });
        assert!(closed, "{steps:?}");
        drop(refused);
    });
    assert!(router.is_some());
    assert!(pid.is_some_and(|pid| eventually(|| !alive(pid))));
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
        // Its answer's grace, then its connection's, then it is killed.
        let bound = crate::router::GRACE * 3;
        let closed = tokio::time::timeout(bound, close.block_task()).await;
        assert!(closed.is_ok_and(|closed| closed.is_ok()));
        assert!(eventually(|| process_of(place, &id).is_none()));
        drop(filling);
    });
    assert!(router.is_some());
}

#[test]
fn a_close_sent_while_its_session_loads_closes_it() {
    let place = Place::new("closing");
    let place = &place;
    let router = drive(place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/reply").await);
        assert!(driven.prompt(&id, "Hello.").await.is_ok());
        let close = || {
            let close = acp::CloseSessionRequest::new(id.clone());
            driven.cx.send_request(close).block_task()
        };
        let load = || {
            let load = acp::LoadSessionRequest::new(id.clone(), &driven.project);
            driven.cx.send_request(load).block_task()
        };
        assert!(close().await.is_ok());
        // Sent together: the close comes before the session's process runs.
        let (loaded, closed) = tokio::join!(load(), close());
        assert!(loaded.is_ok() && closed.is_ok());
        assert!(process_of(place, &id).is_none());
        // A second load while the first opens is refused; the close still
        // waits for the first.
        let (first, second, closed) = tokio::join!(load(), load(), close());
        assert!(first.is_ok() && second.is_err() && closed.is_ok());
        assert!(process_of(place, &id).is_none());
        // A process stopped while it opens never answers: the close ends it
        // after its graces.
        let slow = stub_with("files", vec![acp::EnvVariable::new(SLOW, "10000")]);
        let load_slowly =
            acp::LoadSessionRequest::new(id.clone(), &driven.project).mcp_servers(vec![slow]);
        let loading = driven.cx.send_request(load_slowly);
        assert!(driven.until(|_| process_of(place, &id).is_some()).await);
        let pid = process_of(place, &id).unwrap_or_default();
        let stopped = std::process::Command::new("kill")
            .args(["-STOP", &pid.to_string()])
            .status();
        assert!(stopped.is_ok_and(|stopped| stopped.success()));
        let bound = crate::router::GRACE * 4;
        let closed = tokio::time::timeout(bound, close()).await;
        assert!(closed.is_ok_and(|closed| closed.is_ok()));
        assert!(loading.block_task().await.is_err());
        assert!(eventually(|| process_of(place, &id).is_none()));
        assert!(load().await.is_ok());
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
