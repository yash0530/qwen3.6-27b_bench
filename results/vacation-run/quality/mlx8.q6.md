# mlx8 — q6 `q6_instructions` (instruction_following)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=269.4s decode=9.7 tok/s prefill=379.1 tok/s gen_tokens=2611 prompt_tokens=334 finish=stop
- thinking chars: 6287

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

- **Cron schedule confirmed** — the existing crontab entry has been exported to a version-controlled file.
- **Registry table exists** — the `job_registry` table is provisioned in the target database with the required schema.
- **Runner environment ready** — the `JOB_REGISTRY_URL` environment variable is set in the runner's deployment config.
- **Failure handler deployed** — the `on_failure.py` script is tested and accessible by the runner process.

## Migration Steps

1. Register the ETL job in the `job_registry` table with a `nightly_etl` identifier and its schedule expression.
2. Replace the cron entry in `/etc/cron.d/etl_jobs` with a reference to the runner's `trigger_job.sh` command.
3. Configure the runner to read `JOB_REGISTRY_URL` from the environment and poll the `job_registry` table for due tasks.
4. Add a `failure_deadletter` queue in the `job_registry` schema to capture failed execution payloads.
5. Update the `on_failure.py` handler to write diagnostic context to the `execution_log` table.
6. Run a parallel dry-run via `runner_cli.py --dry-run nightly_etl` and compare output checksums against the legacy script.
7. Remove the legacy `etl_nightly.py` script from the cron schedule and archive it to `/opt/etl/archive/`.

## Rollback

- Restore the original crontab line in `/etc/cron.d/etl_jobs` pointing back to `etl_nightly.py`.
- Set the `nightly_etl` row in `job_registry` to status `disabled` to prevent dual execution.
- Verify the next scheduled run completes by checking the `execution_log` table for a `success` entry.

CHECKLIST-END
