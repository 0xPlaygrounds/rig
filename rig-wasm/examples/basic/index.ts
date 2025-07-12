import { Agent } from "rig-wasm/openai";
import { CompletionsCompletionModel } from "rig-wasm/openai";
import { initPanicHook } from "rig-wasm/utils";

initPanicHook();

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}
let prompt = `Hello world!`;

try {
  console.log(`Attempting to create OpenAI agent...`);
  const agent = new Agent({
    apiKey: key,
    model: "gpt-4o",
  });

  console.log(`Prompt: ${prompt}`);
  let res = await agent.prompt(prompt);
  console.log(`GPT-4o: ${res}`);
} catch (e) {
  if (e instanceof Error) {
    console.error(`Error while prompting: ${e.message}`);
  }
}

console.log(`---`);

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
