# NovaStack Database Migration Guide

Schema changes use the expand-and-contract pattern so old and new application versions can run together. The expand release adds backward-compatible columns or tables first. Application code then writes both representations, a background job backfills historical rows, and the contract release removes the old representation only after verification.

Every migration must be safe to retry and must not hold an exclusive table lock for more than five seconds. Indexes on production tables are created concurrently. Destructive statements such as dropping a column require a separate pull request and approval from the data-platform on-call engineer.

Backfill workers process at most 1,000 rows per batch and pause when database CPU exceeds 70 percent. Progress is stored in a checkpoint table, allowing the job to resume without starting over. Operators compare source and destination row counts and checksums before enabling reads from the new schema.

Database backups are retained daily for 35 days and monthly for 12 months. A restore drill is performed once per quarter. Migration rollback normally means deploying compatible application code; restoring a backup is reserved for data corruption because it discards newer writes.

