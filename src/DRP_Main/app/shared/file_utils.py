"""
Shared file I/O utilities used across all modules.
Replaces app/utils/app_utils.py — fixes:
  - Duplicate extract_interactions (removed first definition, kept correct one)
  - Typo protein_output_directory (was protien_output_directory)
"""
import os
import json
import csv
from pathlib import Path
import pandas as pd
import xmltodict


# ── Excel / CSV / JSON helpers ────────────────────────────────────────────────

def read_excel_file(file_path, sheet_name=None) -> list[dict]:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.fillna("")
    return df.to_dict(orient="records")


def read_csv_file(file_path) -> list[dict]:
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")


def read_json_file(file_path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def write_json_file(file_path, data) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def update_json_file(file_path, data) -> None:
    existing = read_json_file(file_path)
    existing.update(data)
    write_json_file(file_path, existing)


def append_to_json_list(file_path, key, data) -> None:
    existing = read_json_file(file_path)
    existing.setdefault(key, []).append(data)
    write_json_file(file_path, existing)


def get_env_variable(var_name, default_value=None):
    return os.getenv(var_name, default_value)


# ── Protein directory ─────────────────────────────────────────────────────────

def protein_output_directory(protein_name: str) -> str:
    """
    Return the output directory path for a given protein name.

    Honours `SCREENING_OUTPUT_ROOT` so the legacy per-protein path can be
    pointed at persistent storage (a UC volume) the same way batch runs are —
    on Databricks, local cluster storage does not survive the session. Empty
    keeps the in-repo layout: app/shared/ → app/ → data/processed_proteins_output
    """
    from DRP_Main.app.core.config import settings

    configured = (getattr(settings, "SCREENING_OUTPUT_ROOT", "") or "").strip()
    root = Path(configured) if configured else Path(__file__).resolve().parent.parent / "data"
    return str(root / "processed_proteins_output" / protein_name)


# ── XML / interaction helpers ─────────────────────────────────────────────────

def convert_xml_to_json(xml_string: str) -> dict:
    return xmltodict.parse(xml_string)


def normalize_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(normalize_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.extend(normalize_dict({f"{new_key}_{i}": item}, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def restructure_data(data: dict) -> dict:
    transformed = {}
    for key, value in data.items():
        if isinstance(value, dict):
            transformed[key] = normalize_dict(value)
        elif isinstance(value, list):
            transformed[key] = [
                normalize_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            transformed[key] = value
    return transformed


def remove_nested_dicts(data):
    if isinstance(data, list):
        return [remove_nested_dicts(item) for item in data]
    elif isinstance(data, dict):
        return {k: v for k, v in data.items() if not isinstance(v, dict)}
    return data


def extract_interactions(data: dict, interaction_type: str):
    """
    Extract interaction data of a given type from a PLIP XML-parsed dict.
    interaction_type examples: 'hydrophobic_interactions', 'hydrogen_bonds', ...
    """
    if (
        interaction_type.endswith("s")
        and data.get("bindingsite", {}).get("interactions", {}).get(interaction_type)
    ):
        interaction_sub_type = interaction_type[:-1]
        interactions_with_sub_type = (
            data["bindingsite"]["interactions"][interaction_type].get(interaction_sub_type)
        )
        if interactions_with_sub_type:
            cleaned = remove_nested_dicts(interactions_with_sub_type)
            if cleaned:
                return cleaned
        else:
            interactions = data["bindingsite"]["interactions"].get(interaction_type)
            cleaned = remove_nested_dicts(interactions)
            if cleaned:
                return cleaned
    return "No interactions found"


def extract_bs_residues(data: dict):
    bs_residues = data.get("bindingsite", {}).get("bs_residues")
    if bs_residues:
        return bs_residues.get("bs_residue")
    return bs_residues


def create_interaction_table(data: dict) -> dict:
    table = {}
    bindingsite = data.get("bindingsite", {})
    bs_residues = bindingsite.get("bs_residues", [])
    interactions = {
        k: v
        for k in [
            "hydrophobic", "hydrogen_bonds", "water_bridges",
            "salt_bridges", "pi_stacks", "pi_cation_interactions",
            "halogen_bonds", "metal_complexes",
        ]
        if (v := bindingsite.get("interactions", {}).get(k))
    }
    table["bs_residues"] = bs_residues.get("bs_residue") if isinstance(bs_residues, dict) else bs_residues
    return table
