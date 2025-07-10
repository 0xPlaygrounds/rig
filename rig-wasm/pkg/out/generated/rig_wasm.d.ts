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
export class Embedding {
  free(): void;
  /**
   * Create a new Embedding instance.
   * Generally not recommended as these are typically generated automatically from sending embedding requests to model providers.
   */
  constructor(document: string, embedding: Float64Array);
  document(): string;
  embedding(): Float64Array;
}
export class ImageGenerationRequest {
  private constructor();
  free(): void;
  static new(prompt: string): ImageGenerationRequest;
  setPrompt(prompt: string): ImageGenerationRequest;
  setHeight(height: number): ImageGenerationRequest;
  setWidth(width: number): ImageGenerationRequest;
  setAdditionalParameters(json: JSONObject): ImageGenerationRequest;
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
 * A tool that can take a JavaScript function.
 * Generally speaking, any class that implements the `JsToolObject` TS interface will work when creating this.
 */
export class JsTool {
  free(): void;
  constructor(tool: JsToolObject);
}
export class JsVectorStore {
  free(): void;
  constructor(shim: JsVectorStore);
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
  free(): void;
  constructor(opts: AgentOpts);
  prompt(prompt: string): Promise<string>;
  prompt_multi_turn(prompt: string, turns: number): Promise<string>;
  chat(prompt: string, messages: Message[]): Promise<string>;
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
export class TranscriptionRequest {
  free(): void;
  constructor(arr: Uint8Array);
  setFilename(filename: string): TranscriptionRequest;
  setLanguage(language: string): TranscriptionRequest;
  setPrompt(prompt: string): TranscriptionRequest;
  setTemperature(temperature: number): TranscriptionRequest;
  setAdditionalParams(additional_params: JSONObject): TranscriptionRequest;
}
