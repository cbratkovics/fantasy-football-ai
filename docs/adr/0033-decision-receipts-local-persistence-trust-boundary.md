# ADR-0033 — Decision receipts: browser-local persistence, appended events, and an honest trust boundary (2026-09-27)

**Context.** The Decision Lab (ADR-0032) asks the user to compare alternatives, record what they
chose, and later see what happened. That needs a saved record, but the project runs at $0/month
with no user accounts, no database and no server that could hold per-user state; the API on the
Space is stateless and the site is static. Whatever is saved must therefore live in the browser,
must survive a rebuild of the bundle without silently changing its meaning, and must not be
presented as more trustworthy than it is.

**Decision.** A saved decision is a **receipt** (`ffai.decision_lab.receipts`, mirrored in
TypeScript) with two kinds of identity:

* `decision_id = sha256(canonical(decision_inputs))` over the schema and policy versions, the
  snapshot reference including the inputs snapshot's content digest, the slot, every alternative
  with its availability fields, and the parameters. Replaying the same semantic inputs
  reproduces the id; timestamps, the user's action and attached outcomes never enter it.
* `event_id = sha256(canonical({decision_id, seq, event_type, at_utc, payload}))`. Events —
  `action_recorded`, `action_declined`, `outcome_attached` — are appended with their own ids and
  never rewrite the recommendation; the receipt's `action` and `outcome` blocks are projections of
  the event list and are recomputed on validation. At most one action event may exist; the
  initial state is `not_recorded`, and the recommendation is never auto-filled as the choice.

Changing an assumption (availability, a parameter, the slot) does not edit a receipt: it creates
a child decision whose `parent_decision_id` links back, so the original recommendation and the
changed one both remain. Import (`merge`) replays the policy over the imported inputs and rejects
a result that does not reproduce (`result_sha256`), an outcome whose snapshot is not the
decision's own, and two records with the same `decision_id` that disagree; identical or
superset records merge idempotently. Persistence is `localStorage`, keyed by the receipt schema
version, so a schema change starts a new store rather than reinterpreting old records.
`python -m ffai.decision_lab.replay` (`make lab-replay`) re-validates an exported receipt against
the committed bundle.

The trust boundary is stated in the UI and here: the digests are **integrity checks, not
signatures** — they let a reader confirm a receipt matches known evidence and an unmodified
result, and prove nothing about who acted or when. Local storage is editable and erasable, so
"append-only" is application behaviour, not a tamper-proof audit service. The outcome reveal
(inputs first, outcomes only after an action is recorded or declined) is educational sequencing,
not secure blinding: the outcome snapshots are static public files anyone can open.

**Consequences.** No account, database or server-side state; receipts belong to the browser that
made them and move only by explicit export and import. A receipt from an older bundle stays
valid as long as the snapshot digest it names still exists in the bundle; when a snapshot is
rebuilt with a new content digest, the receipt is reported as referring to superseded evidence
rather than silently re-pointed. The Python and TypeScript implementations must agree on
canonical JSON and hashing, which the golden fixtures enforce (ADR-0034).
