# mlx4 — q6 `q6_instructions` (instruction_following)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=124.1s decode=17.8 tok/s prefill=412.4 tok/s gen_tokens=2188 prompt_tokens=334 finish=stop
- thinking chars: 6315

## Prompt

Produce a migration checklist for moving an overnight ETL job from a single-process cron script to a job-registry runner with failure isolation. Follow ALL of these constraints exactly:

1. Output exactly three sections, in this order, with these exact H2 headings: `## Preconditions`, `## Migration Steps`, `## Rollback`.
2. `## Preconditions` must be exactly 4 bullets. Each bullet must start with a bolded 2-4 word label, then an em dash, then one sentence.
3. `## Migration Steps` must be a numbered list of exactly 7 steps. Each step must be one sentence and must name a concrete artifact (a file, table, env var, or command) in backticks.
4. `## Rollback` must be exactly 3 bullets, and the last bullet must begin with the literal word `Verify`.
5. Do not use the words "simply", "just", or "leverage" anywhere.
6. The entire response after your reasoning must be under 300 words.
7. End the response with a single line containing exactly: `CHECKLIST-END`

No preamble, no closing commentary beyond what the constraints allow.

## Answer

## Preconditions

- **Job inventory** — The cron entry and all referenced scripts are documented in a single manifest file.
- **Registry schema** — The `jobs` table in the runner database is deployed and accepts new entries.
- **Failure policy** — A `failure_policy.yaml` file defines retry limits and alerting thresholds per job.
- **Dry-run access** — The execution environment permits running the ETL in a sandboxed mode against a staging schema.

## Migration Steps

1. Create a new entry in the `jobs` table with the job ID `etl_overnight` and point its `entrypoint` to the migrated script path.
2. Set the environment variable `ETL_REGISTRY_MODE=shadow` in the runner's `.env` file so the job executes but results are discarded.
3. Run the command `registry-runner run --job etl_overnight --shadow` for three consecutive nights to validate parity.
4. Flip the environment variable `ETL_REGISTRY_MODE=active` in the runner's `.env` file to begin production execution.
5. Remove the legacy cron entry from `/etc/cron.d/etl_overnight` and archive the file to `archive/cron/`.
6. Add a row to the `dead_letter` table pointing to `etl_overnight` so that any unhandled exception is captured with a trace ID.
7. Update the `oncall.yaml` routing file to assign the new `etl_overnight` job to the data-platform escalation channel.

## Rollback

- Revert `ETL_REGISTRY_MODE` to `disabled` in the runner `.env` file to halt the registered job.
- Restore the archived cron file from `archive/cron/` back to `/etc/cron.d/etl_overnight` to re-enable the legacy schedule.
- Verify that the `jobs` table row for `etl_overnight` shows a status of `retired` and that the next cron tick fires within 60 seconds.

CHECKLIST-END
