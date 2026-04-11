# Open-ASR Model Explorer: Agent Prime Directives

You are operating in a dual-repository workspace. You modify code in the main repository, but your memory and documentation live in the `model-explorer-open-asr-AgentWiki`.

## MANDATORY POST-EXECUTION PROTOCOL
Every time you complete a coding task, fix a bug, or modify architecture based on a user prompt, you MUST perform the following steps before concluding your response:

0. **The Verification Loop:**
   - Before updating the ledger or concluding your task, you MUST verify your fixes.
   - If your environment allows terminal execution, run `cd frontend && npm run test:e2e`.
   - If tests fail, you must autonomously fix the code and re-run until they pass.
   - If you cannot execute commands natively, you must output the exact test command for the user to run before considering the task "Done".

1. **Update the Execution Ledger:**
   - Open `model-explorer-open-asr-AgentWiki/Prompts/Execution_Ledger.md`.
   - Add a new block under the current month's heading.
   - Summarize the exact prompt/instruction the user just gave you, and briefly describe the files you changed to execute it.

2. **Append to the Daily Log:**
   - Open the current month's log (e.g., `model-explorer-open-asr-AgentWiki/Logs/2026-04.md`).
   - Append a bullet point summarizing the action you just took and referencing the new ledger entry.

**DO NOT ask the user for permission to do this.** Consider the update of the ledger and the log as part of the compilation step; your task is not successfully completed until the paperwork is committed.