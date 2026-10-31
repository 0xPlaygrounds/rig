//! What a client sees of a session: the protocol, prompts, tool calls,
//! questions, cancelling, history and MCP servers.

use std::io::{BufRead, BufReader, Write};

use super::*;

/// What `session` told the user: its agent text and its notices.
fn told(driven: &Driven, session: &acp::SessionId) -> String {
    let notices = driven
        .updates(session)
        .into_iter()
        .filter_map(|update| match update {
            acp::SessionUpdate::Notice(notice) => Some(notice.title),
            _ => None,
        });
    driven.said(session) + &notices.collect::<String>()
}

#[test]
fn the_agent_speaks_v1_to_a_client_that_offers_2_and_says_what_it_takes() {
    let place = Place::new("negotiation");
    let router = std::process::Command::new(std::env::current_exe().unwrap_or_default())
        .args(libtest("agent"))
        .env(AGENT, "1")
        .env("RIG_HOME", &place.home)
        .env_remove("RIG_RUN")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok();
    assert!(router.is_some(), "rig acp did not start");
    let Some(mut router) = router else { return };
    let (Some(mut input), Some(output)) = (router.stdin.take(), router.stdout.take()) else {
        return;
    };
    let mut lines = BufReader::new(output).lines().map_while(Result::ok);
    // stdout carries only ACP, but for libtest's lines about the test.
    let mut stray = Vec::new();
    let mut ask = |request: Value| {
        writeln!(input, "{request}").ok();
        let id = request.get("id").cloned();
        for line in lines.by_ref() {
            match serde_json::from_str::<Value>(&line) {
                Ok(answer) if answer.get("id") == id.as_ref() => return Some(answer),
                Ok(_) => {}
                Err(_) => stray.push(line),
            }
        }
        None
    };
    // As a client that speaks 2 first, such as Zed with its beta flag on.
    let offered = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": 2, "info": { "name": "zed", "version": "1" }, "capabilities": {},
    }});
    let answer = ask(offered).unwrap_or_default();
    assert_eq!(
        answer.pointer("/result/protocolVersion"),
        Some(&json!(1)),
        "{answer}"
    );
    let capabilities = answer
        .pointer("/result/agentCapabilities")
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        capabilities,
        json!({
            "loadSession": true,
            "promptCapabilities": { "image": true, "audio": false, "embeddedContext": true },
            "mcpCapabilities": { "http": true, "sse": false },
            "sessionCapabilities": { "list": {}, "resume": {}, "close": {} },
            "auth": {},
        })
    );
    let cwd = place.project.to_string_lossy();
    let new = json!({ "jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {
        "cwd": cwd, "mcpServers": [],
    }});
    let answer = ask(new).unwrap_or_default();
    assert!(
        answer
            .pointer("/result/sessionId")
            .is_some_and(Value::is_string),
        "{answer}"
    );
    let options = answer
        .pointer("/result/configOptions")
        .and_then(Value::as_array);
    let ids: Vec<_> = options
        .into_iter()
        .flatten()
        .filter_map(|option| option.get("id"))
        .collect();
    assert_eq!(ids, [&json!("model"), &json!("reasoning")], "{answer}");
    drop(input);
    stray.extend(lines.filter(|line| serde_json::from_str::<Value>(line).is_err()));
    stray.retain(|line| {
        let line = line.trim();
        !line.is_empty() && line != "running 1 test" && !line.starts_with("test tests::agent")
    });
    assert!(stray.is_empty(), "{stray:?}");
    assert!(router.wait().is_ok_and(|status| status.success()));
}

#[test]
fn a_prompt_streams_its_tool_call_and_text_before_its_answer_and_commands_answer_too() {
    let place = Place::new("prompt");
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        // Two changes sent at once are both answered, the second after the
        // first took effect.
        let change = |config: &str, value: &str| {
            let value = acp::SessionConfigOptionValue::value_id(value.to_owned());
            let request =
                acp::SetSessionConfigOptionRequest::new(id.clone(), config.to_owned(), value);
            driven.cx.send_request(request)
        };
        let (model, reasoning) = (change("model", "test/read"), change("reasoning", "default"));
        let (model, reasoning) = (model.block_task().await, reasoning.block_task().await);
        assert!(
            model.is_ok() && reasoning.is_ok(),
            "{model:?} {reasoning:?}"
        );
        let answer = driven.prompt(&id, "Read the fixture.").await;
        assert_eq!(answer.ok(), Some(acp::StopReason::EndTurn));
        let steps = tool_steps(driven, &id);
        let expected = [
            acp::ToolCallStatus::Pending,
            acp::ToolCallStatus::InProgress,
            acp::ToolCallStatus::Completed,
        ];
        let statuses: Vec<_> = steps.iter().map(|(_, status)| *status).collect();
        assert_eq!(statuses, expected);
        let first = steps.first().map(|(call, _)| call.clone());
        assert!(steps.iter().all(|(call, _)| Some(call) == first.as_ref()));
        let call = driven
            .updates(&id)
            .into_iter()
            .find_map(|update| match update {
                acp::SessionUpdate::ToolCall(call) => Some(call),
                _ => None,
            });
        let call = call.map(|call| {
            (
                call.kind,
                call.locations
                    .into_iter()
                    .map(|at| at.path)
                    .collect::<Vec<_>>(),
            )
        });
        assert_eq!(
            call,
            Some((
                acp::ToolKind::Read,
                vec![project(&place).join("fixture.txt")]
            ))
        );
        assert!(driven.said(&id).contains("read it"));
        let output = driven.updates(&id).into_iter().any(|update| matches!(update,
            acp::SessionUpdate::ToolCallUpdate(update) if format!("{:?}", update.fields.content).contains("fixture text")));
        assert!(output);
        assert_eq!(
            driven.prompt(&id, "/mcp").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        assert!(
            driven.said(&id).contains("No MCP servers"),
            "{}",
            driven.said(&id)
        );
        assert_eq!(
            driven.prompt(&id, "/nosuch").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        assert!(
            driven.said(&id).contains("Unknown command /nosuch"),
            "{}",
            driven.said(&id)
        );
        // Every update of the first turn came before its answer, which the
        // later answers show it did not just outrun.
        let events = driven.seen().events.clone();
        let answered = events
            .iter()
            .position(|event| matches!(event, Event::Answered(of) if *of == id));
        let last = events.iter().rposition(|event| match event {
            Event::Update(of, acp::SessionUpdate::AgentMessageChunk(chunk)) => {
                *of == id && text(&chunk.content).is_some_and(|said| said.contains("read it"))
            }
            Event::Update(of, acp::SessionUpdate::ToolCallUpdate(_)) => *of == id,
            _ => false,
        });
        assert!(
            answered.is_some() && last < answered,
            "{answered:?} {last:?}"
        );
        // A turn a plugin stops for a reason, as `/compact`'s, is no cancel.
        assert!(driven.model(&id, "test/reply").await);
        assert_eq!(
            driven.prompt(&id, "Hello.").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        assert_eq!(
            driven.prompt(&id, "/compact").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        // `/quit` is the terminal's: the editor closes a session.
        for quit in ["/quit", "/quit now"] {
            assert!(driven.prompt(&id, quit).await.is_err(), "{quit}");
        }
        assert_eq!(
            driven.prompt(&id, "/mcp").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
    });
    assert!(router.is_some());
}

#[test]
fn a_model_the_editor_or_a_typed_command_picks_is_never_the_default() {
    let place = Place::new("defaults");
    let saved = r#"{"model":"test/reply","effort":null}"#;
    std::fs::write(place.home.join("defaults.json"), saved).ok();
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/read").await);
        let answer = driven.prompt(&id, "Read the fixture.").await;
        assert_eq!(answer.ok(), Some(acp::StopReason::EndTurn));
        assert!(driven.said(&id).contains("read it"));
        let typed = driven.prompt(&id, "/model test/again").await;
        assert_eq!(typed.ok(), Some(acp::StopReason::EndTurn));
        let answer = driven.prompt(&id, "Read the other one.").await;
        assert_eq!(answer.ok(), Some(acp::StopReason::EndTurn));
        assert!(driven.said(&id).contains("not there"), "{}", driven.said(&id));
    });
    assert!(router.is_some());
    let kept = std::fs::read_to_string(place.home.join("defaults.json"));
    assert_eq!(kept.ok().as_deref(), Some(saved));
}

#[test]
fn an_allowed_call_runs_a_denied_one_does_not_and_cancel_answers_an_open_one() {
    let place = Place::new("permission");
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        for (answering, status) in [
            (Answering::Allow, acp::ToolCallStatus::Completed),
            (Answering::Deny, acp::ToolCallStatus::Failed),
        ] {
            driven.answering(answering);
            let new = driven.new_session(Vec::new()).await;
            assert!(new.is_some(), "no session");
            let Some(new) = new else { return };
            let id = new.session_id;
            assert!(driven.model(&id, "test/guarded").await);
            assert_eq!(
                driven.prompt(&id, "Read it.").await.ok(),
                Some(acp::StopReason::EndTurn)
            );
            let last = tool_steps(driven, &id).last().map(|(_, status)| *status);
            assert_eq!(last, Some(status), "{answering:?}");
        }
        // The question shows the call whole, and why it is asked.
        let asked = driven.seen().asked.last().cloned();
        assert!(asked.is_some(), "no permission asked");
        let Some(asked) = asked else { return };
        let fields = asked.fields;
        // The call as it was announced.
        let announced = driven.seen().events.iter().any(|event| {
            matches!(event, Event::Update(_, acp::SessionUpdate::ToolCall(call))
                if call.tool_call_id == asked.tool_call_id)
        });
        assert!(announced, "{}", asked.tool_call_id);
        assert_eq!(fields.title.as_deref(), Some("read guarded.txt"));
        assert_eq!(fields.kind, Some(acp::ToolKind::Read));
        let path = fields
            .raw_input
            .as_ref()
            .and_then(|input| input.get("path").cloned());
        assert_eq!(path, Some(json!("guarded.txt")));
        let at: Vec<_> = fields
            .locations
            .into_iter()
            .flatten()
            .map(|at| at.path)
            .collect();
        assert_eq!(at, [project(&place).join("guarded.txt")]);
        let why = format!("{:?}", fields.content);
        assert!(why.contains("It reads the guarded file."), "{why}");
        driven.answering(Answering::Later);
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/guarded").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Read it.".into()]);
        let prompt = driven.cx.send_request(request);
        assert!(driven.until(|seen| !seen.later.is_empty()).await);
        let open = driven.seen().later.pop();
        let Some(open) = open else { return };
        driven
            .cx
            .send_notification(acp::CancelNotification::new(id.clone()))
            .ok();
        let answer = prompt.block_task().await.map(|answer| answer.stop_reason);
        assert_eq!(answer.ok(), Some(acp::StopReason::Cancelled));
        // The question the cancel took away is cancelled at the client too.
        let cancelled = open.cancellation();
        let told = tokio::time::timeout(Duration::from_secs(10), cancelled.cancelled()).await;
        assert!(told.is_ok(), "the open question was not cancelled");
    });
    assert!(router.is_some());
}

#[test]
fn cancel_stops_a_running_tool_and_one_sent_with_the_prompt() {
    let place = Place::new("cancel");
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/wait").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        let running = |seen: &Seen| {
            seen.events.iter().any(|event| {
                matches!(event, Event::Update(_, acp::SessionUpdate::ToolCallUpdate(update))
                if update.fields.status == Some(acp::ToolCallStatus::InProgress))
            })
        };
        assert!(driven.until(running).await);
        #[cfg(target_os = "linux")]
        assert!(driven.until(|_| sleeping(&place)).await);
        driven
            .cx
            .send_notification(acp::CancelNotification::new(id.clone()))
            .ok();
        let answer = prompt.block_task().await.map(|answer| answer.stop_reason);
        assert_eq!(answer.ok(), Some(acp::StopReason::Cancelled));
        #[cfg(target_os = "linux")]
        assert!(eventually(|| !sleeping(&place)));
        // Closing a session cancels its prompt first, and frees it to load.
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/wait").await);
        let prompt = tokio::spawn(driven.send(&id, "Wait."));
        let running = |seen: &Seen| {
            seen.events.iter().any(|event| {
                matches!(event, Event::Update(of, acp::SessionUpdate::ToolCallUpdate(update))
                if *of == id && update.fields.status == Some(acp::ToolCallStatus::InProgress))
            })
        };
        assert!(driven.until(running).await);
        // Two closes at once are both answered.
        let close = || {
            driven
                .cx
                .send_request(acp::CloseSessionRequest::new(id.clone()))
        };
        let (first, second) = (close(), close());
        let (first, second) = (first.block_task().await, second.block_task().await);
        assert!(first.is_ok() && second.is_ok(), "{first:?} {second:?}");
        let answer = prompt.await.ok().and_then(Result::ok);
        assert_eq!(answer, Some(acp::StopReason::Cancelled));
        #[cfg(target_os = "linux")]
        assert!(eventually(|| !sleeping(&place)));
        let load = acp::LoadSessionRequest::new(id.clone(), &driven.project);
        assert!(driven.cx.send_request(load).block_task().await.is_ok());
        // Cancel stops work that is no turn too, such as a sign-in.
        let prompt = tokio::spawn(driven.send(&id, "/hold"));
        let held = driven.project.join("held");
        assert!(driven.until(|_| held.exists()).await);
        driven
            .cx
            .send_notification(acp::CancelNotification::new(id.clone()))
            .ok();
        let answer = tokio::time::timeout(Duration::from_secs(10), prompt).await;
        let answer = answer.ok().and_then(Result::ok).and_then(Result::ok);
        assert_eq!(answer, Some(acp::StopReason::Cancelled));
        // A prompt cancelled as it is sent is cancelled too.
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/wait").await);
        let request = acp::PromptRequest::new(id.clone(), vec!["Wait.".into()]);
        let prompt = driven.cx.send_request(request);
        driven
            .cx
            .send_notification(acp::CancelNotification::new(id.clone()))
            .ok();
        let answer = prompt.block_task().await.map(|answer| answer.stop_reason);
        assert_eq!(answer.ok(), Some(acp::StopReason::Cancelled));
    });
    assert!(router.is_some());
}

#[test]
fn a_loaded_session_replays_its_history_as_it_went_and_a_resumed_one_does_not() {
    let place = Place::new("history");
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/read").await);
        assert!(driven.prompt(&id, "Read the fixture.").await.is_ok());
        assert!(driven.model(&id, "test/again").await);
        assert!(driven.prompt(&id, "Read the missing file.").await.is_ok());
        // Both calls are `r1` to the provider, two calls to the client.
        let live = tool_steps(driven, &id);
        let done: Vec<_> = live
            .iter()
            .filter(|(_, status)| *status != acp::ToolCallStatus::InProgress)
            .filter(|(_, status)| *status != acp::ToolCallStatus::Pending)
            .collect();
        assert!(
            matches!(done.as_slice(), [(first, acp::ToolCallStatus::Completed), (second, acp::ToolCallStatus::Failed)] if first != second),
            "{live:?}"
        );
        let close = acp::CloseSessionRequest::new(id.clone());
        assert!(driven.cx.send_request(close).block_task().await.is_ok());
        let before = driven.updates(&id).len();
        let load = acp::LoadSessionRequest::new(id.clone(), &driven.project);
        assert!(driven.cx.send_request(load).block_task().await.is_ok());
        let replayed: Vec<_> = driven.updates(&id).into_iter().skip(before).collect();
        let user = replayed.iter().find_map(|update| match update {
            acp::SessionUpdate::UserMessageChunk(chunk) => text(&chunk.content),
            _ => None,
        });
        assert_eq!(user.as_deref(), Some("Read the fixture."));
        let steps: Vec<_> = tool_steps(driven, &id)
            .into_iter()
            .skip(live.len())
            .collect();
        let finished = |steps: &[(String, acp::ToolCallStatus)]| {
            steps
                .iter()
                .filter(|(_, status)| *status != acp::ToolCallStatus::InProgress)
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(finished(&steps), finished(&live));
        assert!(driven.said(&id).matches("read it").count() == 2);
        let close = acp::CloseSessionRequest::new(id.clone());
        assert!(driven.cx.send_request(close).block_task().await.is_ok());
        let before = driven.updates(&id).len();
        let resume = acp::ResumeSessionRequest::new(id.clone(), &driven.project);
        assert!(driven.cx.send_request(resume).block_task().await.is_ok());
        let replayed = driven.updates(&id).into_iter().skip(before).any(|update| {
            matches!(
                update,
                acp::SessionUpdate::UserMessageChunk(_) | acp::SessionUpdate::ToolCall(_)
            )
        });
        assert!(!replayed);
        let sessions = driven
            .cx
            .send_request(acp::ListSessionsRequest::new())
            .block_task()
            .await;
        let listed = sessions.map(|list| {
            list.sessions
                .into_iter()
                .map(|info| info.session_id)
                .collect::<Vec<_>>()
        });
        assert_eq!(listed.ok(), Some(vec![id.clone()]));
    });
    assert!(router.is_some());
}

#[test]
fn a_client_mcp_server_serves_the_model_and_a_failing_one_is_told_after_the_session_opens() {
    let place = Place::new("mcp");
    let broken = acp::McpServer::Stdio(acp::McpServerStdio::new("broken", "/no/such/mcp-server"));
    // Slow to start: the session opens only once it connected.
    let files = stub_with("files", vec![acp::EnvVariable::new(SLOW, "1500")]);
    let router = drive(&place, full_client(), async |driven| {
        let new = driven.new_session(vec![files, broken]).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/mcp").await);
        assert_eq!(
            driven.prompt(&id, "Add.").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        let last = tool_steps(driven, &id).last().map(|(_, status)| *status);
        assert_eq!(last, Some(acp::ToolCallStatus::Completed));
        let failed = |seen: &Seen| {
            seen.events.iter().any(|event| {
                matches!(event, Event::Update(_, acp::SessionUpdate::Notice(notice))
                if notice.title.contains("`broken` could not connect"))
            })
        };
        assert!(driven.until(failed).await, "{}", told(driven, &id));
        // Nothing of the session came before its answer, which a client
        // needs to know the session by.
        let events = driven.seen().events.clone();
        let opened = events
            .iter()
            .position(|event| matches!(event, Event::Opened(opened) if *opened == id));
        let first = events
            .iter()
            .position(|event| matches!(event, Event::Update(of, _) if *of == id));
        assert!(opened.is_some() && first > opened, "{opened:?} {first:?}");
        assert!(driven.prompt(&id, "/mcp").await.is_ok());
        // A call a plugin starts before its reply ended is not shown, even
        // with the id of a call the conversation has.
        let shown = tool_steps(driven, &id).len();
        assert!(driven.prompt(&id, "/early m1").await.is_ok());
        assert_eq!(tool_steps(driven, &id).len(), shown);
        let said = driven.said(&id);
        assert!(said.contains("files (client): connected, 1 tool"), "{said}");
    });
    assert!(router.is_some());
}

#[test]
fn a_failed_turn_tells_the_client_its_message_was_taken_out() {
    let place = Place::new("failed");
    let router = drive(&place, full_client(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/fail").await);
        assert!(driven.prompt(&id, "Summarize the fixture.").await.is_err());
        // In the reply, or as a notice once it was answered.
        let taken = |seen: &Seen| {
            seen.events.iter().any(|event| match event {
                Event::Update(of, acp::SessionUpdate::AgentMessageChunk(chunk)) => {
                    *of == id
                        && text(&chunk.content).is_some_and(|said| {
                            said.contains("taken out of the conversation")
                                && said.contains("Summarize the fixture.")
                        })
                }
                Event::Update(of, acp::SessionUpdate::Notice(notice)) => {
                    *of == id
                        && notice.title.contains("taken out of the conversation")
                        && notice.title.contains("Summarize the fixture.")
                }
                _ => false,
            })
        };
        assert!(driven.until(taken).await, "{}", told(driven, &id));
    });
    assert!(router.is_some());
}

#[test]
fn a_question_is_a_form_or_closed_with_why_for_a_client_without_forms() {
    for (client, told_text) in [
        (full_client(), "answered: Some([\"b\"])"),
        (
            acp::ClientCapabilities::new(),
            "not answered: The client cannot show this question",
        ),
    ] {
        let place = Place::new("question");
        let router = drive(&place, client, async |driven| {
            let new = driven.new_session(Vec::new()).await;
            assert!(new.is_some(), "no session");
            let Some(new) = new else { return };
            let id = new.session_id;
            assert!(driven.prompt(&id, "/ask").await.is_ok());
            // The answer may come after the prompt's: an open question is no
            // work the prompt waits for.
            let deadline = Instant::now() + Duration::from_secs(30);
            while !told(driven, &id).contains(told_text) && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(
                told(driven, &id).contains(told_text),
                "{}",
                told(driven, &id)
            );
        });
        assert!(router.is_some());
    }
}

/// The model the last options `session` was told of show.
fn advertised_model(seen: &Seen, session: &acp::SessionId) -> Option<String> {
    let updates = seen.events.iter().filter_map(|event| match event {
        Event::Update(of, acp::SessionUpdate::ConfigOptionUpdate(update)) if of == session => {
            Some(update)
        }
        _ => None,
    });
    let mut models = updates.filter_map(|update| {
        update
            .config_options
            .iter()
            .find_map(|option| match &option.kind {
                acp::SessionConfigKind::Select(select) if option.id.to_string() == "model" => {
                    Some(select.current_value.to_string())
                }
                _ => None,
            })
    });
    models.next_back()
}

/// Showing another agent of the session, such as with `/agents`, makes it
/// the one prompts go to, and the options show its model.
#[test]
fn the_options_follow_the_agent_shown() {
    let place = Place::new("shown");
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        assert!(driven.model(&id, "test/reply").await);
        assert!(driven.prompt(&id, "Hello.").await.is_ok());
        assert!(driven.prompt(&id, "/fork").await.is_ok());
        let forked = "Forked into a new agent";
        let confirmed = driven.until(|seen| {
            seen.events.iter().any(|event| match event {
                Event::Update(of, acp::SessionUpdate::AgentMessageChunk(chunk)) => {
                    *of == id && text(&chunk.content).is_some_and(|said| said.contains(forked))
                }
                _ => false,
            })
        });
        let confirmed = confirmed.await;
        assert!(confirmed, "{}", driven.said(&id));
        assert!(driven.prompt(&id, "/model test/read").await.is_ok());
        let read = |seen: &Seen| advertised_model(seen, &id).as_deref() == Some("test/read");
        assert!(driven.until(read).await, "{}", driven.said(&id));
        // The agent forked from, back on `test/reply`.
        let logs = std::fs::read_dir(place.home.join("sessions"))
            .into_iter()
            .flatten();
        let logs = logs.flatten().flat_map(|session| {
            std::fs::read_dir(session.path().join("logs"))
                .into_iter()
                .flatten()
                .flatten()
        });
        let source = logs.map(|log| log.path()).find_map(|log| {
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            let name = log.file_stem()?.to_str()?.to_owned();
            (name != "session" && !text.contains("ForkedFrom")).then_some(name)
        });
        assert!(source.is_some(), "no source log");
        let Some(source) = source else { return };
        assert!(
            driven
                .prompt(&id, &format!("/agents {source}"))
                .await
                .is_ok()
        );
        let reply = |seen: &Seen| advertised_model(seen, &id).as_deref() == Some("test/reply");
        assert!(driven.until(reply).await, "{}", driven.said(&id));
    });
    assert!(router.is_some());
}

#[test]
fn a_refused_model_change_says_why_and_shows_the_options_as_they_are() {
    let place = Place::new("refused");
    let router = drive(&place, acp::ClientCapabilities::new(), async |driven| {
        let new = driven.new_session(Vec::new()).await;
        assert!(new.is_some(), "no session");
        let Some(new) = new else { return };
        let id = new.session_id;
        let value = acp::SessionConfigOptionValue::value_id(UNREACHABLE.to_owned());
        let change = acp::SetSessionConfigOptionRequest::new(id.clone(), "model", value);
        let refused = driven.cx.send_request(change).block_task().await;
        let why = refused.err().map(|error| error.message);
        assert!(
            why.as_ref()
                .is_some_and(|why| !why.is_empty() && why != "rig did not change it"),
            "{why:?}"
        );
        // The options the client is told after it show the model rig has.
        let told = |seen: &Seen| {
            seen.events.iter().any(|event| match event {
                Event::Update(of, acp::SessionUpdate::ConfigOptionUpdate(update)) => {
                    *of == id && format!("{:?}", update.config_options).contains(UNREACHABLE)
                }
                _ => false,
            })
        };
        assert!(driven.until(told).await);
        // A model that can be reached takes its place.
        assert!(driven.model(&id, "test/reply").await);
        assert_eq!(
            driven.prompt(&id, "Hello.").await.ok(),
            Some(acp::StopReason::EndTurn)
        );
        // `/model` changes it too, and the options show it.
        assert!(driven.prompt(&id, "/model test/read").await.is_ok());
        let chosen = |seen: &Seen| {
            seen.events.iter().any(|event| match event {
                Event::Update(of, acp::SessionUpdate::ConfigOptionUpdate(update)) => {
                    *of == id
                        && update.config_options.iter().any(|option| {
                            matches!(&option.kind, acp::SessionConfigKind::Select(select)
                                if option.id.to_string() == "model"
                                    && select.current_value.to_string() == "test/read")
                        })
                }
                _ => false,
            })
        };
        assert!(driven.until(chosen).await, "{}", driven.said(&id));
    });
    assert!(router.is_some());
}
