import { Agent, ResponsesStreamingCompletionResponse } from "rig-wasm/openai";
import { CompletionsCompletionModel } from "rig-wasm/openai";
import { decodeReadableStream, RawStreamingChoice } from "rig-wasm/streaming";
import { initPanicHook } from "rig-wasm/utils";

initPanicHook();

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}

let prompt = `Please write the first 100 words of Lorem Ipsum. Skip all text and only respond with the text as I am testing a framework example.`;

try {
  console.log(`Attempting to create OpenAI agent...`);
  const agent = new Agent({
    apiKey: key,
    model: "gpt-4o",
  });

  console.log(`Prompt: ${prompt}`);
  const stream = await agent.prompt_stream(prompt);

  let aggregatedText = "";
  for await (const chunk of decodeReadableStream(stream)) {
    if (chunk.text !== null && chunk.text !== aggregatedText) {
      aggregatedText += chunk.text;
      process.stdout.write(chunk.text);
    }
  }
} catch (e) {
  if (e instanceof Error) {
    console.error(`Error while prompting: ${e.message}`);
  }
}

console.log(`\n---`);

prompt = `Hello world!`;

try {
  console.log(`Attempting to create OpenAI completion...`);
  const model = new CompletionsCompletionModel({
    apiKey: key,
    modelName: "gpt-4o",
  });

  let req = {
    preamble: "You are a helpful assistant.",
    messages: [
      {
        role: "user",
        content: [
          {
            text: prompt,
            type: "text",
          },
        ],
      },
    ],
  };

  console.log(`Prompt: ${prompt}`);
  let res = await model.completion(req);
  console.log(`GPT-4o completion response: ${res.choice[0].text}`);
} catch (e) {
  if (e instanceof Error) {
    console.error(`Error while prompting: ${e.message}`);
  }
}
