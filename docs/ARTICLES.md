# Articles and forum posts (November 2026)

The results on the test pages in this repository, written up for engineers in the words they search with: agent drift, token cost, latency, the context window, sessions per GPU, prompt injection, model size, memory and tool calls. Each article gives its numbers and its limits, and says which test page they come from.

AXI is a layer on the host side, between a model and the state it edits. The model is unchanged, and every claim is about the state the host holds. The framework and the architecture are by **Rajnish Choubey**; the tests were built and run by Shalendra Gaur. Nothing here is a statement about any company's servers.

## Articles (dev.to originals)

- [Google's developer blog on agent context cost, and a measured host-side fix: 169x fewer tokens a step (Gemini 3.8 Flash)](https://dev.to/shalendra_gaur_ecd0567a49/googles-developer-blog-on-agent-context-cost-and-a-measured-host-side-fix-169x-fewer-tokens-a-4phb) (24 November; each public statement next to the measured result and its page)
- [AXI by Rajnish Choubey: a host-side layer that keeps LLM-edited state exact. Questions, answers and measured evidence](https://dev.to/shalendra_gaur_ecd0567a49/axi-by-rajnish-choubey-a-host-side-layer-that-keeps-llm-edited-state-exact-questions-answers-and-13cp)
- [AI agent drift: what it is, how to measure it, and the host-side fix that held 4,000 of 4,000 steps](https://dev.to/shalendra_gaur_ecd0567a49/ai-agent-drift-what-it-is-how-to-measure-it-and-the-host-side-fix-that-held-4000-of-4000-steps-4kg4)
- [LLM agent token cost: why a kept chat grows with the square of the steps, and the measured fix (prompt caching counted)](https://dev.to/shalendra_gaur_ecd0567a49/llm-agent-token-cost-why-a-kept-chat-grows-with-the-square-of-the-steps-and-the-measured-fix-1021)
- [Why LLM latency grows with conversation length, and the measured fix: a flat 0.1605 s to 0.1651 s step over 20,000 steps](https://dev.to/shalendra_gaur_ecd0567a49/why-llm-latency-grows-with-conversation-length-and-the-measured-fix-a-flat-01605-s-to-01651-s-2oi6)
- [How many LLM agent sessions can one GPU serve? Measured: 8 to 256 sessions on one NVIDIA L4, every step exact](https://dev.to/shalendra_gaur_ecd0567a49/how-many-llm-agent-sessions-can-one-gpu-serve-measured-8-to-256-sessions-on-one-nvidia-l4-every-1edn)
- [Host-held state for LLM agents: five questions to ask LangGraph, Letta, smolagents, PydanticAI and AXI, with measured answers](https://dev.to/shalendra_gaur_ecd0567a49/host-held-state-for-llm-agents-five-questions-to-ask-langgraph-letta-smolagents-pydanticai-and-1b43)
- [Prompt injection and LLM agent state: 0 of 1,401,041 injection proposals committed at the host's gate (10 million tested)](https://dev.to/shalendra_gaur_ecd0567a49/prompt-injection-and-llm-agent-state-0-of-1401041-injection-proposals-committed-at-the-hosts-2fbj)
- [Does model size matter for LLM agent reliability? 11 open models, 1.5B to 32B: host-held state exact on 10 of 11](https://dev.to/shalendra_gaur_ecd0567a49/does-model-size-matter-for-llm-agent-reliability-11-open-models-15b-to-32b-host-held-state-2c83)
- [LLM agent memory compared: whole chat, summary, sliding window, retrieval and host-held state on a 14B model](https://dev.to/shalendra_gaur_ecd0567a49/llm-agent-memory-compared-whole-chat-summary-sliding-window-retrieval-and-host-held-state-on-a-371i)
- [What happens when an LLM agent's chat outgrows the context window? Measured: 108 to 235 steps in 32k, 337 in 131k](https://dev.to/shalendra_gaur_ecd0567a49/what-happens-when-an-llm-agents-chat-outgrows-the-context-window-measured-108-to-235-steps-in-2fah)
- [LLM tool calls and structured edits: on a 7B model 57 to 79 percent of steps right as written, 94 to 99.8 with a rule check](https://dev.to/shalendra_gaur_ecd0567a49/llm-tool-calls-and-structured-edits-on-a-7b-model-57-to-79-percent-of-steps-right-as-written-94-om5)
- [LLM agent token cost and state drift: the fixes compared, with the authors' papers and measured numbers](https://dev.to/shalendra_gaur_ecd0567a49/llm-agent-token-cost-and-state-drift-the-fixes-compared-with-authors-papers-and-measured-numbers-2ae2)

The same articles are also on [Hashnode](https://axi-runtime.hashnode.dev) and on the [WordPress.com blog](https://axiruntimetests.wordpress.com). The comparison article is also on [Hashnode](https://axi-runtime.hashnode.dev/llm-agent-token-cost-and-state-drift-the-fixes-compared-with-authors-papers-and-measured-numbers) and on [Medium](https://medium.com/@gaur.shalendra2412/llm-agent-token-cost-and-state-drift-the-fixes-compared-with-authors-papers-and-measured-numbers-e21daf29f407).

## Forum posts with open questions

Two measurement posts on the Google AI Developers Forum, each ending with concrete questions for people who know the model and the API:

- [Gemini 3.8 Flash long agent loops: measured cost per correct step, and why first-token time still rises with about 97 percent of the prompt cached](https://discuss.ai.google.dev/t/gemini-3-8-flash-long-agent-loops-measured-cost-per-correct-step-and-why-first-token-time-still-rises-with-about-97-of-the-prompt-cached/201990)
- [Gemini 3.8 Flash, a 4,000-step test: compound delete requests left out dependants in 46 of 97 replies (0 of 472 on plain deletes)](https://discuss.ai.google.dev/t/gemini-3-8-flash-4-000-step-test-compound-delete-requests-left-out-dependants-in-46-of-97-replies-0-of-472-on-plain-deletes/202025)

Back to the [README](../README.md).
