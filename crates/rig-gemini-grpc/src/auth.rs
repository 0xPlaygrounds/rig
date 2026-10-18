//! The API key every Gemini gRPC request carries, as request metadata.

use tonic::metadata::{Ascii, MetadataValue};
use tonic::service::Interceptor;
use tonic::{Request, Status};

/// Adds API-key and client-identification metadata to outgoing requests.
#[derive(Clone)]
pub struct ApiKeyInterceptor {
    pub(crate) api_key: MetadataValue<Ascii>,
    pub(crate) client_id: MetadataValue<Ascii>,
}

impl Interceptor for ApiKeyInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert("x-goog-api-key", self.api_key.clone());
        request
            .metadata_mut()
            .insert("x-goog-api-client", self.client_id.clone());
        Ok(request)
    }
}
