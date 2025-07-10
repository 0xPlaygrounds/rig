// import { initPanicHook } from "rig-wasm";
import { OpenAIAgent } from "rig-wasm/openai";

// initPanicHook();

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}
let prompt = `Please increment the counter by 1 and let me know what the result is.`;

try {
  const agent = new OpenAIAgent({
    apiKey: key,
    model: "gpt-4o",
  });

  console.log(`Prompt: ${prompt}`);
  let res = await agent.prompt_multi_turn(prompt, 2);
  console.log(`GPT-4o: ${res}`);
} catch (e) {
  if (e instanceof Error) {
    console.log(`Error while prompting: ${e.message}`);
  }
}
