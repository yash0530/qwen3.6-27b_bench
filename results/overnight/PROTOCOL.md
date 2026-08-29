# Overnight 2026-08-22: MLX+MTP+caching without crashes
Goal: max speedup with MTP speculative decode + persistent KV/prompt caching, zero kernel panics.
Rails: abort <25% free RAM per turn; assert engine/model/env before timing; ctx<=65536;
no MTP+APC until the checkpoint-only patch is proven; panic check (uptime + ls /Library/Logs/DiagnosticReports/*.panic) after every stage.
Results appended per stage to STAGES.md in this dir. Summary for the user: SUMMARY.md.
