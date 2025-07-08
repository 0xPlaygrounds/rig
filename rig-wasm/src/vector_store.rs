use send_wrapper::SendWrapper;
use serde::Deserialize;
use serde_wasm_bindgen::Deserializer;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{js_sys::{self, Array}, JsFuture};

use crate::JsVectorStoreShim;

#[wasm_bindgen]
pub struct JsVectorStore {
    inner: SendWrapper<JsVectorStoreShim>,
}

#[wasm_bindgen]
impl JsVectorStore {
    #[wasm_bindgen(constructor)]
    pub fn new(shim: JsVectorStoreShim) -> Self {
        let inner = SendWrapper::new(shim);
        Self { inner }
    }
}

impl rig::vector_store::VectorStoreIndex for JsVectorStore {
    fn top_n<T: for<'a> Deserialize<'a> + Send>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<
        Output = Result<Vec<(f64, String, T)>, rig::vector_store::VectorStoreError>,
    > + Send {
        async {
            let promise = self
                .inner
                .top_n(query, n as u32)
                .unchecked_into::<js_sys::Promise>();

            let promise = JsFuture::from(promise).await.unwrap();

            let arr = Array::from(&promise).into_iter().map(|x| {
                let x = Array::from(&x);

            })

            Ok(promise)
        }
    }

    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<
        Output = Result<Vec<(f64, String)>, rig::vector_store::VectorStoreError>,
    > + Send {
        todo!()
    }
}
