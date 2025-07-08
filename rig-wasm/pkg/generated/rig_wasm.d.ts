/* tslint:disable */
/* eslint-disable */
export function initPanicHook(): void;
/**
 * Configuration options for Cloudflare's image optimization feature:
 * <https://blog.cloudflare.com/introducing-polish-automatic-image-optimizati/>
 */
export enum PolishConfig {
  Off = 0,
  Lossy = 1,
  Lossless = 2,
}
export enum RequestRedirect {
  Error = 0,
  Follow = 1,
  Manual = 2,
}
/**
 * The `ReadableStreamType` enum.
 *
 * *This API requires the following crate features to be activated: `ReadableStreamType`*
 */
type ReadableStreamType = "bytes";
export class AssistantContent {
  private constructor();
  free(): void;
  static text(text: string): AssistantContent;
  static tool_call(id: string, _function: ToolFunction): AssistantContent;
  static tool_call_with_call_id(id: string, call_id: string, _function: ToolFunction): AssistantContent;
}
export class Document {
  free(): void;
  constructor(id: string, text: string);
  setAdditionalProps(additional_props: any): Document;
}
export class IntoUnderlyingByteSource {
  private constructor();
  free(): void;
  start(controller: ReadableByteStreamController): void;
  pull(controller: ReadableByteStreamController): Promise<any>;
  cancel(): void;
  readonly type: ReadableStreamType;
  readonly autoAllocateChunkSize: number;
}
export class IntoUnderlyingSink {
  private constructor();
  free(): void;
  write(chunk: any): Promise<any>;
  close(): Promise<any>;
  abort(reason: any): Promise<any>;
}
export class IntoUnderlyingSource {
  private constructor();
  free(): void;
  pull(controller: ReadableStreamDefaultController): Promise<any>;
  cancel(): void;
}
/**
 * A tool that uses JavaScript.
 * Unfortunately, JavaScript functions are *mut u8 at their core (when it comes to how they're typed in Rust).
 * This means that we need to use `send_wrapper::SendWrapper` which automatically makes it Send.
 * However, if it gets dropped from outside of the thread where it was created, it will panic.
 */
export class JsTool {
  private constructor();
  free(): void;
  static new(tool: JsToolObject): JsTool;
}
export class Message {
  private constructor();
  free(): void;
}
/**
 * Configuration options for Cloudflare's minification features:
 * <https://www.cloudflare.com/website-optimization/>
 */
export class MinifyConfig {
  private constructor();
  free(): void;
  js: boolean;
  html: boolean;
  css: boolean;
}
export class OpenAIAgent {
  private constructor();
  free(): void;
  prompt(prompt: string): Promise<string>;
  prompt_multi_turn(prompt: string, turns: number): Promise<string>;
  chat(prompt: string, messages: Message[]): Promise<string>;
}
export class OpenAIAgentBuilder {
  free(): void;
  constructor(client: OpenAIClient, model_name: string);
  setPreamble(preamble: string): OpenAIAgentBuilder;
  addTool(tool: JsToolObject): OpenAIAgentBuilder;
  build(): OpenAIAgent;
}
export class OpenAIClient {
  free(): void;
  constructor(api_key: string);
  static from_url(api_key: string, base_url: string): OpenAIClient;
  completion_model(model_name: string): OpenAICompletionModel;
  agent(model_name: string): OpenAIAgentBuilder;
}
export class OpenAICompletionModel {
  free(): void;
  constructor(client: OpenAIClient, model_name: string);
}
export class OpenAICompletionRequest {
  free(): void;
  constructor(model: OpenAICompletionModel, prompt: Message);
  setPreamble(preamble: string): OpenAICompletionRequest;
  setChatHistory(chat_history: Message[]): OpenAICompletionRequest;
  setDocuments(documents: Document[]): OpenAICompletionRequest;
  setTools(tools: ToolDefinition[]): OpenAICompletionRequest;
  setTemperature(temperature: number): OpenAICompletionRequest;
  setMaxTokens(max_tokens: bigint): OpenAICompletionRequest;
  setAdditionalParams(obj: any): OpenAICompletionRequest;
  send(): Promise<AssistantContent[]>;
}
export class R2Range {
  private constructor();
  free(): void;
  get offset(): number | undefined;
  set offset(value: number | null | undefined);
  get length(): number | undefined;
  set length(value: number | null | undefined);
  get suffix(): number | undefined;
  set suffix(value: number | null | undefined);
}
export class ToolDefinition {
  private constructor();
  free(): void;
}
export class ToolFunction {
  private constructor();
  free(): void;
  name(): string;
  args(): any;
}
