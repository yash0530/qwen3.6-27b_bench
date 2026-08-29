# mlx6 — q6 `q6_instructions` (instruction_following)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=250.6s decode=12.6 tok/s prefill=330.7 tok/s gen_tokens=3137 prompt_tokens=334 finish=stop
- thinking chars: 9661

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

- **Registry accessible** — The job-registry service must be running and accepting registrations from the ETL host.
- **Cron entry identified** — The current crontab line for the overnight ETL script has been documented and its exit-code contract is understood.
- **Failure table created** — A dedicated `etl_failures` table exists in the operational database to capture isolated job failures.
- **Alerting channel configured** — A Slack webhook or equivalent alert sink is reachable from the runner environment.

## Migration Steps

1. Register the ETL task in the registry by running `registry register --name etl_nightly --schedule "0 2 * * *"`.
2. Create the wrapper script at `/opt/etl/runners/etl_nightly.sh` that invokes the original process and writes exit status to `etl_failures`.
3. Set the environment variable `ETL_REGISTRY_TOKEN` in the runner's systemd unit file `/etc/systemd/system/etl-runner.service`.
4. Disable the old cron entry by commenting out the line in `/etc/cron.d/etl_nightly` and reloading with `systemctl reload cron`.
5. Deploy the runner service using `systemctl enable --now etl-runner.service` on the target host.
6. Execute a dry-run with `etl-runner trigger --name etl_nightly --dry-run` and confirm the job appears in the registry dashboard.
7. Monitor the first two production runs and validate that failure rows land in `etl_failures` when the test-injected bad input is used.

## Rollback

- Re-enable the original cron entry by uncommenting the line in `/etc/cron.d/etl_nightly` and reloading cron.
- Stop the runner with `systemctl stop etl-runner.service` and disable it with `systemctl disable etl-runner.service`.
- Verify the next scheduled run executes under the original cron process and writes output to the expected destination.

CHECKLIST-END
