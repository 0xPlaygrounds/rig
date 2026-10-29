use super::*;
use agent_client_protocol::schema::ProtocolVersion;

fn backend() -> Arc<Backend> {
    Arc::new(Backend {
        home: Home::at(std::env::temp_dir().join(uuid::Uuid::new_v4().to_string())),
        workers: Arc::default(),
    })
}

/// Exercises SDK negotiation and request dispatch without provider calls.
#[tokio::test]
async fn v1_negotiates_and_rejects_unknown_sessions() -> Result<()> {
    Client
        .builder()
        .connect_with(router(backend()), async |cx| {
            let response = cx
                .send_request(v1::InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            if response.protocol_version != ProtocolVersion::V1
                || !response.agent_capabilities.load_session
            {
                return Err(internal("incorrect v1 capabilities"));
            }
            let request = v1::PromptRequest::new("missing", vec!["hello".into()]);
            if cx.send_request(request).block_task().await.is_ok() {
                return Err(internal("unknown session was accepted"));
            }
            Ok(())
        })
        .await
}

/// Exercises v2's distinct initialization schema and baseline session methods.
#[tokio::test]
async fn v2_negotiates_and_lists_without_starting_a_model() -> Result<()> {
    Client
        .v2()
        .connect_with(router(backend()), async |cx| {
            let response = cx
                .send_request(v2::InitializeRequest::new(
                    ProtocolVersion::V2,
                    v2::Implementation::new("test", "1"),
                ))
                .block_task()
                .await?;
            if response.protocol_version != ProtocolVersion::V2
                || response.capabilities.session.is_none()
            {
                return Err(internal("incorrect v2 capabilities"));
            }
            let sessions = cx
                .send_request(v2::ListSessionsRequest::new())
                .block_task()
                .await?;
            if !sessions.sessions.is_empty() {
                return Err(internal("unexpected saved sessions"));
            }
            Ok(())
        })
        .await
}
