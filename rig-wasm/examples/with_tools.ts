import { OpenAIClient, initPanicHook } from "../pkg/index.ts";
import { counter } from "./tools/tools.ts";

initPanicHook();

let key = process.env.OPENAI_API_KEY;
if (key === undefined) {
  console.log(
    `OPENAI_API_KEY not present as an environment variable. Please add it then try running this example again.`,
  );
  process.exit(1);
}
let prompt = `Please increment the counter by 1 and let me know what the result is.`;

try {
  const agent = new OpenAIClient(key)
    .agent("gpt-4o")
    .setPreamble(
      "You are an AI agent equipped with a counter, which can only increment.",
    )
    .addTool(counter)
    .build();

  console.log(`Prompt: ${prompt}`);
  let res = await agent.prompt_multi_turn(prompt, 2);
  console.log(`GPT-4o: ${res}`);
} catch (e) {
  console.log(`Error while prompting: ${e.msg}`);
}
