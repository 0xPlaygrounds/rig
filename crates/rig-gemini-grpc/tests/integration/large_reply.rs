//! A reply larger than tonic's default 4 MB decoding limit, such as a
//! generated image, reaches the caller.

use std::convert::Infallible;

use rig_core::completion::CompletionRequest;
use rig_core::providers::gemini::GeminiConfig;
use rig_gemini_grpc::GeminiGrpc;
use rig_gemini_grpc::proto::{self, GenerateContentRequest, GenerateContentResponse};
use tonic::codegen::{BoxFuture, Service, http};
use tonic::transport::server::TcpIncoming;
use tonic::transport::{Endpoint, Server};

/// Over tonic's 4 MB default.
const REPLY_BYTES: usize = 5 * 1024 * 1024;

/// A `GenerateContent` endpoint answering every call with one text part
/// [`REPLY_BYTES`] long.
#[derive(Clone)]
struct LargeReply;

impl tonic::server::NamedService for LargeReply {
    const NAME: &'static str = "google.ai.generativelanguage.v1beta.GenerativeService";
}

impl tonic::server::UnaryService<GenerateContentRequest> for LargeReply {
    type Response = GenerateContentResponse;
    type Future = std::future::Ready<Result<tonic::Response<Self::Response>, tonic::Status>>;

    fn call(&mut self, _request: tonic::Request<GenerateContentRequest>) -> Self::Future {
        let content = proto::Content {
            parts: vec![proto::Part {
                data: Some(proto::part::Data::Text("a".repeat(REPLY_BYTES))),
                ..proto::Part::default()
            }],
            role: "model".to_owned(),
        };
        std::future::ready(Ok(tonic::Response::new(GenerateContentResponse {
            candidates: vec![proto::Candidate {
                content: Some(content),
                finish_reason: proto::candidate::FinishReason::Stop as i32,
                ..proto::Candidate::default()
            }],
            ..GenerateContentResponse::default()
        })))
    }
}

impl<B> Service<http::Request<B>> for LargeReply
where
    B: tonic::codegen::Body + Send + 'static,
    B::Error: Into<tonic::codegen::StdError> + Send,
{
    type Response = http::Response<tonic::body::Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default())
                .max_encoding_message_size(usize::MAX);
            Ok(grpc.unary(LargeReply, request).await)
        })
    }
}

#[tokio::test]
async fn a_reply_over_four_megabytes_decodes() -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(
        Server::builder()
            .add_service(LargeReply)
            .serve_with_incoming(TcpIncoming::from(listener)),
    );

    let channel = Endpoint::from_shared(format!("http://{address}"))?
        .connect()
        .await?;
    let transport = GeminiGrpc::with_channel(&GeminiConfig::new("key"), channel)?;
    let reply = transport
        .completion("gemini-2.5-flash")
        .call(CompletionRequest::new("draw something large"))
        .await;
    server.abort();

    let text = reply?.text();
    anyhow::ensure!(text.len() == REPLY_BYTES, "{} bytes", text.len());
    Ok(())
}
