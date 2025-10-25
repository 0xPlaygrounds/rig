use crate::agent::prompt_request::streaming::StreamingError;
use crate::agent::{Agent, MultiTurnStreamItem, Text};
use crate::completion::CompletionModel;
use crate::pipeline::try_stream_ops::TryStreamingOp;
use crate::wasm_compat::*;
use futures::{Stream, StreamExt};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

/// A trait for streamed pipeline operations.
pub trait StreamingOp: WasmCompatSend + WasmCompatSync {
    type Input: WasmCompatSend + WasmCompatSync;
    type Output: WasmCompatSend + WasmCompatSync;

    /// Calls the stream
    fn call_stream(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Pin<Box<dyn Stream<Item = Self::Output> + Send>>> + WasmCompatSend;

    /// Map over each item in the stream
    fn map_stream<F, U>(self, f: F) -> StreamMap<Self, F>
    where
        F: Fn(Self::Output) -> U + WasmCompatSend + WasmCompatSync,
        U: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        StreamMap::new(self, f)
    }

    /// Async map over each item in the stream
    fn then_stream<F, Fut>(self, f: F) -> StreamThen<Self, F>
    where
        F: Fn(Self::Output) -> Fut + WasmCompatSend + WasmCompatSync,
        Fut: Future + WasmCompatSend,
        Fut::Output: WasmCompatSend + WasmCompatSync,
        Self: Sized,
    {
        StreamThen::new(self, f)
    }

    /// Collect the stream into a vector (materializes the full stream).
    fn collect(self) -> StreamCollect<Self>
    where
        Self: Sized,
    {
        StreamCollect::new(self)
    }

    /// Collects a streaming agent prompt response into just the response text.
    fn collect_text<M>(self) -> StreamCollectText<M, Self>
    where
        M: CompletionModel + 'static,
        Self: Sized,
        Self::Output:
            Stream<Item = Result<MultiTurnStreamItem<M::StreamingResponse>, StreamingError>>,
    {
        StreamCollectText::new(PhantomData, self)
    }
}

/// A streaming combinator for [`StreamingOp`] that maps stream items into another type.
pub struct StreamMap<S, F> {
    stream_op: S,
    f: F,
}

impl<S, F> StreamMap<S, F> {
    pub fn new(stream_op: S, f: F) -> Self {
        Self { stream_op, f }
    }
}

impl<S, F, U> StreamingOp for StreamMap<S, F>
where
    S: StreamingOp,
    S::Output: 'static,
    F: Fn(S::Output) -> U + WasmCompatSend + WasmCompatSync + Clone + 'static,
    U: WasmCompatSend + WasmCompatSync,
{
    type Input = S::Input;
    type Output = U;

    async fn call_stream(
        &self,
        input: Self::Input,
    ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
        let stream = self.stream_op.call_stream(input).await;
        let f = self.f.clone();
        Box::pin(stream.map(move |item| f(item)))
    }
}

/// A stream combinator for [`StreamingOp`] that allows for async mapping operations for a stream.
pub struct StreamThen<S, F> {
    stream_op: S,
    f: F,
}

impl<S, F> StreamThen<S, F> {
    pub fn new(stream_op: S, f: F) -> Self {
        Self { stream_op, f }
    }
}

impl<S, F, Fut> StreamingOp for StreamThen<S, F>
where
    S: StreamingOp,
    S::Output: 'static,
    F: Fn(S::Output) -> Fut + WasmCompatSend + WasmCompatSync + Clone + 'static,
    Fut: Future + WasmCompatSend,
    Fut::Output: WasmCompatSend + WasmCompatSync,
{
    type Input = S::Input;
    type Output = Fut::Output;

    async fn call_stream(
        &self,
        input: Self::Input,
    ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
        let stream = self.stream_op.call_stream(input).await;
        let f = self.f.clone();
        Box::pin(stream.then(move |item| {
            let f = f.clone();
            async move { f(item).await }
        }))
    }
}

/// Collects a stream into a vector of stream items.
pub struct StreamCollect<S> {
    stream_op: S,
}

impl<S> StreamCollect<S> {
    pub fn new(stream_op: S) -> Self {
        Self { stream_op }
    }
}

impl<S> super::op::Op for StreamCollect<S>
where
    S: StreamingOp,
{
    type Input = S::Input;
    type Output = Vec<S::Output>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let stream = self.stream_op.call_stream(input).await;
        stream.collect().await
    }
}

/// A stream combinator for [`StreamingOp`] that collects a streaming response from [`super::agent::Agent`] into the response text (or returns an error).
pub struct StreamCollectText<M, S> {
    _ty: PhantomData<M>,
    op: S,
}

impl<M, S> StreamCollectText<M, S> {
    pub fn new(_ty: PhantomData<M>, op: S) -> Self {
        Self { _ty, op }
    }
}

impl<M, S> super::op::Op for StreamCollectText<M, S>
where
    M: CompletionModel,
    S: StreamingOp<Output = Result<MultiTurnStreamItem<M::StreamingResponse>, StreamingError>>,
{
    type Input = S::Input;
    type Output = Result<String, StreamingError>;

    async fn call(&self, input: Self::Input) -> Self::Output {
        let stream = self.op.call_stream(input).await;
        let mut text = String::new();

        futures::pin_mut!(stream);
        while let Some(result) = stream.next().await {
            match result {
                Ok(item) => {
                    if let MultiTurnStreamItem::StreamItem(
                        crate::streaming::StreamedAssistantContent::Text(Text { text: content }),
                    ) = item
                    {
                        text.push_str(&content);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(text)
    }
}

/// Converts a regular operation into a streaming operation.
pub struct OpToStream<O> {
    op: O,
}

impl<O> OpToStream<O> {
    /// Creates a new OpToStream<O> where the input is O.
    pub fn new(op: O) -> Self {
        Self { op }
    }
}

impl<O> StreamingOp for OpToStream<O>
where
    O: super::op::Op,
    O::Output: 'static,
{
    type Input = O::Input;
    type Output = O::Output;

    async fn call_stream(
        &self,
        input: Self::Input,
    ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
        let output = self.op.call(input).await;
        Box::pin(futures::stream::once(async move { output }))
    }
}

/// An extension trait for easy conversion from an item to a stream.
pub trait OpExt: super::op::Op + Sized {
    /// Convert this Op into a StreamingOp that yields a single item
    fn into_stream(self) -> OpToStream<Self> {
        OpToStream::new(self)
    }
}

impl<T: super::op::Op> OpExt for T {}

/// An extension trait for easy conversion from a stream to an item.
pub trait StreamingOpExt: StreamingOp + Sized {
    /// Collect the stream and continue with a regular Op
    fn then_op<O>(self, op: O) -> super::op::Sequential<StreamCollect<Self>, O>
    where
        O: super::op::Op<Input = Vec<Self::Output>>,
    {
        super::op::Sequential::new(StreamCollect::new(self), op)
    }
}

impl<T: StreamingOp> StreamingOpExt for T {}

pub struct StreamingPromptOp<M>
where
    M: CompletionModel,
{
    agent: Agent<M>,
}

impl<M> StreamingPromptOp<M>
where
    M: CompletionModel + 'static,
{
    pub fn new(agent: Agent<M>) -> Self {
        Self { agent }
    }
}

impl<M> StreamingOp for StreamingPromptOp<M>
where
    M: CompletionModel + Clone + 'static,
{
    type Input = String;
    type Output = Result<MultiTurnStreamItem<M::StreamingResponse>, StreamingError>;

    async fn call_stream(
        &self,
        input: Self::Input,
    ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
        use crate::streaming::StreamingPrompt;

        let stream = self.agent.stream_prompt(&input).await;
        stream
    }
}

pub struct StreamWith<S, F> {
    stream_op: S,
    f: F,
}

impl<S, F> StreamWith<S, F> {
    pub fn new(stream_op: S, f: F) -> Self {
        Self { stream_op, f }
    }
}

impl<S, F, Out> StreamingOp for StreamWith<S, F>
where
    S: StreamingOp,
    S::Output: 'static,
    F: Fn(Pin<Box<dyn Stream<Item = S::Output> + Send>>) -> Pin<Box<dyn Stream<Item = Out> + Send>>
        + WasmCompatSend
        + WasmCompatSync
        + Clone
        + 'static,
    Out: WasmCompatSend + WasmCompatSync,
{
    type Input = S::Input;
    type Output = Out;

    async fn call_stream(
        &self,
        input: Self::Input,
    ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
        let stream = self.stream_op.call_stream(input).await;
        (self.f.clone())(stream)
    }
}

/// Convert a stream of items into a StreamingOp
pub struct FromStream<F> {
    f: F,
}

impl<F, S> StreamingOp for FromStream<F>
where
    F: Fn() -> S + WasmCompatSend + WasmCompatSync,
    S: Stream + Send + 'static,
    S::Item: WasmCompatSend + WasmCompatSync,
{
    type Input = ();
    type Output = S::Item;

    async fn call_stream(
        &self,
        _: Self::Input,
    ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
        Box::pin((self.f)())
    }
}

#[cfg(test)]
mod tests {
    use crate::pipeline::{self};

    use super::StreamingOp;
    use super::*;
    use futures::stream;

    struct MockStreamOp;

    impl StreamingOp for MockStreamOp {
        type Input = i32;
        type Output = i32;

        async fn call_stream(
            &self,
            input: Self::Input,
        ) -> Pin<Box<dyn Stream<Item = Self::Output> + Send>> {
            Box::pin(stream::iter(vec![input, input * 2, input * 3]))
        }
    }

    #[tokio::test]
    async fn test_stream_map() {
        let op = MockStreamOp.map_stream(|x| x + 1);
        let stream = op.call_stream(5).await;
        let results: Vec<_> = stream.collect().await;
        assert_eq!(results, vec![6, 11, 16]);
    }

    #[tokio::test]
    async fn test_stream_then() {
        let op = MockStreamOp.then_stream(|x| async move { x * 2 });
        let stream = op.call_stream(3).await;
        let results: Vec<_> = stream.collect().await;
        assert_eq!(results, vec![6, 12, 18]);
    }

    #[tokio::test]
    async fn test_op_to_stream() {
        let results = pipeline::new()
            .map(|x: i32| x * 2)
            .into_stream()
            .call_stream(5)
            .await
            .collect::<Vec<_>>()
            .await;

        assert_eq!(results, vec![10]);
    }
}
