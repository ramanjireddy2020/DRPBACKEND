"""
DRP public API (`/v1`).

Implements the reverse-engineered DRP (Drug Repurposing Platform) OpenAPI contract
— onboarding, dashboard, sessions, the five agent modules, knowledge graph,
articles, projects and files/export.

The five agent modules delegate to the real pipeline logic that already lives in
`app/modules/*` and `app/api/v1/endpoints/*`; this package owns the HTTP contract,
async job lifecycle and persistence only.
"""
