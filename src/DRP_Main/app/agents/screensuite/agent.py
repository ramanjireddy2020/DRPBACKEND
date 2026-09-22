"""
ScreenSuite Agent — molecular docking.

The agent runs the resolution step itself, as ordinary code, before submitting
the pipeline. Resolution is not something the LLM decides whether to call: it
must happen before docking every time, and there is no ordering in which
skipping it makes sense. It stays a separate module (`resolution_service`) for
testing and maintainability — only the agent-facing surface is one submission.

Ambiguous names stop the run at `awaiting_structure_confirmation` and come back
as a shortlist, mirroring CurateX's `awaiting_target_confirmation`. That
checkpoint sits *before* any compute is spent, which is the whole point: docking
the wrong structure is far more expensive than one round-trip.
"""
from typing import Any, Dict, List, Optional

from DRP_Main.app.agents.base.base_agent import BaseAgent
from DRP_Main.app.core.logging import get_logger
from DRP_Main.app.modules.screening import pipeline_service, resolution_service
from DRP_Main.app.modules.screening.enums import ResolutionStage
from DRP_Main.app.modules.screening.schemas import (
    DrugBase,
    DrugQuery,
    ProteinBase,
    ProteinQuery,
)

logger = get_logger(__name__)


class ScreenSuiteAgent(BaseAgent):
    """Screening agent: resolve → confirm (if needed) → dock → profile."""

    def __init__(self):
        super().__init__(tools=[])

    @property
    def name(self) -> str:
        return "screensuite"

    # ── Input normalisation ───────────────────────────────────────────────────

    @staticmethod
    def _protein_queries(input_data: Dict[str, Any], session_state: Dict[str, Any]) -> List[ProteinQuery]:
        """
        Build protein queries from whatever the agent was handed.

        Accepts explicit `proteins`, a single `protein`/`target`, or a target
        carried over from TxKG/CurateX in the session.
        """
        raw = input_data.get("proteins")
        if isinstance(raw, list) and raw:
            return [
                ProteinQuery(**item) if isinstance(item, dict) else ProteinQuery(name=str(item))
                for item in raw
            ]

        single = input_data.get("protein")
        if isinstance(single, dict):
            # The legacy shape carries a local file path, not an identifier.
            if single.get("protein_name") and single.get("pdb_file_path"):
                return []
            return [ProteinQuery(**single)]
        if isinstance(single, str) and single.strip():
            return [ProteinQuery(name=single.strip())]

        for key in ("target", "selected_target"):
            value = input_data.get(key) or session_state.get(key)
            if isinstance(value, str) and value.strip():
                return [ProteinQuery(name=value.strip())]
        return []

    @staticmethod
    def _drug_queries(input_data: Dict[str, Any], session_state: Dict[str, Any]) -> List[DrugQuery]:
        """
        Build drug queries, including CurateX's candidate table carry-over.
        """
        raw = input_data.get("drugs")
        if isinstance(raw, list) and raw:
            queries = []
            for item in raw:
                if isinstance(item, dict):
                    # A legacy DrugBase (name + local SDF) is not a query.
                    if item.get("drug_name") and item.get("sdf_file_path"):
                        continue
                    name = item.get("name") or item.get("drug_name")
                    if not name:
                        continue
                    queries.append(
                        DrugQuery(
                            name=name,
                            identifier=item.get("identifier") or item.get("pubchem_id"),
                            source=item.get("source"),
                        )
                    )
                elif isinstance(item, str) and item.strip():
                    queries.append(DrugQuery(name=item.strip()))
            if queries:
                return queries

        carried = session_state.get("curatex_candidates") or session_state.get("selected_drugs")
        if isinstance(carried, list) and carried:
            names = [
                item.get("name") if isinstance(item, dict) else str(item) for item in carried
            ]
            return [DrugQuery(name=name) for name in names if name]
        return []

    @staticmethod
    def _legacy_records(input_data: Dict[str, Any]):
        """The already-downloaded-files shape, if that is what arrived."""
        protein = input_data.get("protein")
        drugs = input_data.get("drugs") or []
        if not isinstance(protein, dict):
            return None, []
        if not (protein.get("protein_name") and protein.get("pdb_file_path")):
            return None, []
        drug_records = [
            DrugBase(**drug)
            for drug in drugs
            if isinstance(drug, dict) and drug.get("drug_name") and drug.get("sdf_file_path")
        ]
        return ProteinBase(**protein), drug_records

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, input_data: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        session_state = session_state or {}

        # A caller that already has local structure files takes the legacy path
        # — there is nothing to resolve.
        legacy_protein, legacy_drugs = self._legacy_records(input_data)
        if legacy_protein is not None:
            return self._run_legacy(legacy_protein, legacy_drugs)

        proteins = self._protein_queries(input_data, session_state)
        drugs = self._drug_queries(input_data, session_state)
        if not proteins:
            return {
                "error": "No protein provided",
                "recommendation": "Name a protein (or carry a target over from target identification).",
            }
        if not drugs:
            return {
                "error": "No drugs provided",
                "recommendation": "Name the compounds to screen, or carry a candidate list over from CurateX.",
            }

        # Confirmed choices from a previous checkpoint, if the caller is
        # answering one.
        selections = input_data.get("confirmations") or input_data.get("selections")
        if selections:
            resolution = resolution_service.confirm(selections, proteins, drugs)
        else:
            resolution = resolution_service.resolve(proteins, drugs, allow_partial=True)

        if resolution.stage == ResolutionStage.awaiting_confirmation:
            return {
                "stage": resolution.stage,
                "ambiguous": [entity.model_dump() for entity in resolution.ambiguous],
                "unresolved": [entity.model_dump() for entity in resolution.unresolved],
                "resolved_so_far": {
                    "proteins": [p.model_dump() for p in resolution.proteins],
                    "drugs": [d.model_dump() for d in resolution.drugs],
                },
                "recommendation": resolution.message,
                "session_state_update": {
                    "screensuite_pending_proteins": [p.model_dump() for p in proteins],
                    "screensuite_pending_drugs": [d.model_dump() for d in drugs],
                },
            }

        if not resolution.proteins or not resolution.drugs:
            return {
                "error": "Nothing resolved to screen",
                "unresolved": [entity.model_dump() for entity in resolution.unresolved],
                "recommendation": resolution.message,
            }

        return self._run_batch(resolution)

    @staticmethod
    def _artifact(records: List[Dict[str, Any]], protein_name: str) -> Optional[Dict[str, Any]]:
        """
        Binding-pose artifact for the supervisor's molecular view.

        The renderer is imported here rather than at module level so a broken or
        absent artifact layer degrades this one field instead of the agent.
        """
        if not records:
            return None
        try:
            from DRP_Main.app.artifacts.binding_pose_renderer import BindingPoseRenderer

            return BindingPoseRenderer().render(records, protein_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Binding-pose artifact unavailable: %s", exc)
            return None

    def _run_batch(self, resolution) -> Dict[str, Any]:
        """Submit the resolved batch and assemble the supervisor object."""
        try:
            result = pipeline_service.screen_batch(
                resolution.proteins,
                resolution.drugs,
                skip_interactions=False,
            )
        except pipeline_service.ScreeningError as exc:
            return {"error": str(exc), "recommendation": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.error("ScreenSuite batch failed: %s", exc, exc_info=True)
            return {"error": str(exc), "recommendation": "ScreenSuite could not complete this run."}

        top_hits = [
            {**record, "protein": protein.protein_name}
            for protein in result.proteins
            for record in protein.top_affinity_records
        ]
        top_hits.sort(key=lambda record: record.get("Affinity_kcal_per_mol", 0.0))

        primary = next(
            (protein.protein_name for protein in result.proteins if protein.top_affinity_records),
            resolution.proteins[0].name if resolution.proteins else "",
        )

        # The resolution step's unresolved list and the pipeline's own failures
        # are different stages, so both travel — together they say what did and
        # did not make it through the run.
        return {
            "run_id": result.run_id,
            "status": result.status,
            "docking_results": [protein.model_dump() for protein in result.proteins],
            "top_hits": top_hits[:5],
            "interaction_artifact": self._artifact(top_hits, primary),
            "interaction_reports": [
                report.model_dump()
                for protein in result.proteins
                for report in protein.interaction_reports
            ],
            "files": result.files,
            "unresolved": [entity.model_dump() for entity in resolution.unresolved],
            "failures": [failure.model_dump() for failure in result.failures],
            "time_taken": result.time_taken,
            "recommendation": result.recommendation,
            "session_state_update": {
                "screensuite_run_id": result.run_id,
                "screensuite_top_hits": [hit.get("ligand") for hit in top_hits[:10]],
            },
        }

    def _run_legacy(self, protein: ProteinBase, drugs: List[DrugBase]) -> Dict[str, Any]:
        """One protein, already-downloaded files. No resolution involved."""
        try:
            result = pipeline_service.protein_preparation(protein, drugs, "0")
        except Exception as exc:  # noqa: BLE001
            logger.error("Docking failed: %s", exc, exc_info=True)
            return {"error": str(exc), "recommendation": "Try different protein/drug inputs."}

        records = [
            record.model_dump() if hasattr(record, "model_dump") else record
            for record in (result.top_n_affinity_records or [])
        ]
        return {
            "status": result.status,
            "docking_results": records,
            "top_hits": records[:5],
            "interaction_artifact": self._artifact(records, protein.protein_name),
            "recommendation": f"Docking complete. {len(records)} top-ranked record(s).",
        }
