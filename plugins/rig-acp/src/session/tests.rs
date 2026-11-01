use super::*;

#[test]
fn a_clients_mcp_servers_go_to_rig_as_written_and_an_sse_one_holds_its_name_refused() {
    let stdio = acp::McpServerStdio::new("files", "/bin/files-mcp")
        .args(vec!["${HOME}".to_owned()])
        .env(vec![
            acp::EnvVariable::new("TOKEN", "first"),
            acp::EnvVariable::new("TOKEN", "${SECRET}"),
        ]);
    // Header names are alike in any case: the last one stands.
    let http = acp::McpServerHttp::new("web", "https://example.com/mcp?k=${KEY}").headers(vec![
        acp::HttpHeader::new("authorization", "Bearer old"),
        acp::HttpHeader::new("Authorization", "Bearer ${TOKEN}"),
    ]);
    let sse = acp::McpServerSse::new("old", "https://example.com/sse");
    let servers = [
        acp::McpServer::Stdio(stdio),
        acp::McpServer::Http(http),
        acp::McpServer::Sse(sse),
    ];
    let added: Vec<_> = servers
        .iter()
        .filter_map(mcp_server)
        .map(|(_, server)| server)
        .collect();
    let named: Vec<_> = added.iter().map(|server| server.name.as_str()).collect();
    assert_eq!(named, ["files", "web", "old"]);
    let specs: Vec<_> = added.into_iter().map(|server| server.spec).collect();
    let expected_stdio = ServerSpec::Stdio {
        command: "/bin/files-mcp".to_owned(),
        args: vec!["${HOME}".to_owned()],
        // A later entry of a name wins, as in an environment.
        env: BTreeMap::from([("TOKEN".to_owned(), "${SECRET}".to_owned())]),
        cwd: None,
    };
    let expected_http = ServerSpec::Http {
        url: "https://example.com/mcp?k=${KEY}".to_owned(),
        headers: BTreeMap::from([("Authorization".to_owned(), "Bearer ${TOKEN}".to_owned())]),
    };
    assert_eq!(specs.first(), Some(&Ok(expected_stdio)));
    assert_eq!(specs.get(1), Some(&Ok(expected_http)));
    assert!(
        specs
            .get(2)
            .is_some_and(|spec| spec.as_ref().is_err_and(|why| why.contains("SSE")))
    );
}

#[test]
fn a_server_name_for_people_becomes_one_for_tools() {
    for (given, name) in [
        ("files", "files"),
        ("GitHub MCP", "GitHub-MCP"),
        ("files.local", "files-local"),
        ("a__b_", "a_b"),
        ("2fa", "_2fa"),
        ("", "server"),
    ] {
        assert_eq!(server_name(given), name, "{given}");
    }
}

#[test]
fn a_prompt_ends_as_its_last_turn_did_and_a_plugins_stop_is_no_cancel() {
    use rig_harness::rig_core::completion::message::{AssistantMessage, StopReason};

    let answered = |stop| {
        let mut reply = AssistantMessage::new(Vec::new());
        reply.stop = stop;
        TurnOutcome::Answered(Said::Assistant(reply))
    };
    let stop = |cancelled, ended: Option<TurnOutcome>| stop_reason(cancelled, ended.as_ref()).ok();
    assert_eq!(
        stop(false, Some(answered(None))),
        Some(acp::StopReason::EndTurn)
    );
    let cut = answered(Some(StopReason::Length));
    assert_eq!(
        stop(false, Some(cut.clone())),
        Some(acp::StopReason::MaxTokens)
    );
    assert_eq!(stop(true, Some(cut)), Some(acp::StopReason::Cancelled));
    let compacted = StopCause::Plugin {
        by: "compaction".to_owned(),
        why: "compacted".to_owned(),
    };
    assert_eq!(stopped(&compacted), acp::StopReason::EndTurn);
    assert_eq!(stopped(&StopCause::User), acp::StopReason::Cancelled);
    let exit = StopCause::Exit("the run ended".to_owned());
    assert_eq!(stopped(&exit), acp::StopReason::Cancelled);
    assert_eq!(stop(false, None), Some(acp::StopReason::EndTurn));
    assert!(stop_reason(false, Some(&TurnOutcome::Failed("why".to_owned()))).is_err());
}
