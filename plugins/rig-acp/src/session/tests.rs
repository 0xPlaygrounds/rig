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
    let added: Vec<_> = servers.iter().filter_map(mcp_server).collect();
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
