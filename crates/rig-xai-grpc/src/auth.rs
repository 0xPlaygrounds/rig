//! The API key every xAI gRPC request carries, as request metadata.

use tonic::metadata::{Ascii, MetadataValue};
use tonic::service::Interceptor;
use tonic::{Request, Status};

/// Adds the `authorization` bearer token to outgoing requests.
#[derive(Clone)]
pub struct ApiKeyInterceptor {
    pub(crate) authorization: MetadataValue<Ascii>,
}

impl Interceptor for ApiKeyInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert("authorization", self.authorization.clone());
        Ok(request)
    }
}
