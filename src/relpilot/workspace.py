"""Workspace loader and validator for RelPilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import yaml


class WorkspaceValidationError(ValueError):
    """Raised when a workspace fails structural or schema validation."""


class Workspace:
    """Encapsulates a RelPilot workspace directory.

    Structure:
        <root>/
          data/*.csv|parquet     relational tables
          schema.yaml            primary keys, foreign keys, event time columns
          tasks/*.yaml           declarative prediction tasks
          skills/*.md            agent playbooks
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.data_dir = self.root_dir / "data"
        self.schema_path = self.root_dir / "schema.yaml"
        self.tasks_dir = self.root_dir / "tasks"
        self.skills_dir = self.root_dir / "skills"
        self.schema: dict[str, Any] = {}
        self.tasks: dict[str, Path] = {}
        self.skills: dict[str, Path] = {}

        self._load()

    def _load(self) -> None:
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise WorkspaceValidationError(f"Workspace directory does not exist: {self.root_dir}")

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise WorkspaceValidationError(f"Missing required 'data' directory in workspace: {self.data_dir}")

        # Load or introspect schema
        if self.schema_path.exists():
            try:
                with open(self.schema_path, "r", encoding="utf-8") as f:
                    self.schema = yaml.safe_load(f) or {}
            except Exception as e:
                raise WorkspaceValidationError(f"Failed to parse schema.yaml: {e}") from e
        else:
            self.schema = self._introspect_schema()

        # Discover tasks
        if self.tasks_dir.exists() and self.tasks_dir.is_dir():
            for task_file in sorted(self.tasks_dir.glob("*.yaml")):
                self.tasks[task_file.stem] = task_file

        # Discover skills
        if self.skills_dir.exists() and self.skills_dir.is_dir():
            for skill_file in sorted(self.skills_dir.glob("*.md")):
                self.skills[skill_file.stem] = skill_file

        self.validate()

    def _introspect_schema(self) -> dict[str, Any]:
        """Introspect tables from data/ if schema.yaml is missing."""
        schema: dict[str, Any] = {}
        data_files = list(self.data_dir.glob("*.csv")) + list(self.data_dir.glob("*.parquet"))
        for f in data_files:
            table_name = f.stem
            try:
                df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, nrows=10)
            except Exception as e:
                raise WorkspaceValidationError(f"Failed to read table data from {f.name}: {e}") from e

            cols = list(df.columns)
            # Heuristic primary key: column matching '<table_singular>_id' or 'id'
            pkey = None
            candidates = [c for c in cols if c.lower() in (f"{table_name}_id", f"{table_name[:-1]}_id", "id")]
            if candidates:
                pkey = candidates[0]

            # Heuristic time column: columns containing 'time', 'date', or 'ts'
            time_col = None
            time_candidates = [
                c for c in cols if any(k in c.lower() for k in ("timestamp", "date", "_ts", "time"))
            ]
            if time_candidates:
                time_col = time_candidates[0]

            entry: dict[str, Any] = {"path": f.name, "columns": cols}
            if pkey:
                entry["pkey"] = pkey
            if time_col:
                entry["time_col"] = time_col

            schema[table_name] = entry

        return schema

    def validate(self) -> None:
        """Validate workspace integrity with clear error messages."""
        if not self.schema:
            raise WorkspaceValidationError(f"No tables found in workspace schema or data directory.")

        # Check that data files referenced in schema exist
        for table_name, meta in self.schema.items():
            rel_path = meta.get("path", f"{table_name}.parquet")
            full_path = self.data_dir / rel_path
            if not full_path.exists():
                raise WorkspaceValidationError(
                    f"Table '{table_name}' references file '{rel_path}' which does not exist in {self.data_dir}"
                )

            # Check foreign keys point to valid tables
            fkeys = meta.get("fkeys", {})
            for fk_col, parent_table in fkeys.items():
                if parent_table not in self.schema:
                    raise WorkspaceValidationError(
                        f"Table '{table_name}' has foreign key '{fk_col}' pointing to non-existent table '{parent_table}'"
                    )
                parent_pkey = self.schema[parent_table].get("pkey")
                if not parent_pkey:
                    raise WorkspaceValidationError(
                        f"Table '{parent_table}' referenced by foreign key in '{table_name}' has no primary key ('pkey') declared."
                    )

        # Validate task files
        for task_name, task_path in self.tasks.items():
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    t_data = yaml.safe_load(f) or {}
            except Exception as e:
                raise WorkspaceValidationError(f"Error parsing task YAML '{task_path.name}': {e}") from e

            required_fields = [
                "database",
                "entity_table",
                "entity_col",
                "time_col",
                "target_col",
                "task_type",
                "timedelta",
                "val_timestamp",
                "test_timestamp",
                "query",
            ]
            missing = [k for k in required_fields if k not in t_data]
            if missing:
                raise WorkspaceValidationError(
                    f"Task '{task_name}' missing required fields: {', '.join(missing)}"
                )

            ent_table = t_data["entity_table"]
            if ent_table not in self.schema:
                raise WorkspaceValidationError(
                    f"Task '{task_name}' targets entity_table '{ent_table}' which is not in the schema."
                )

            # Check temporal ordering
            val_ts = pd.Timestamp(t_data["val_timestamp"])
            test_ts = pd.Timestamp(t_data["test_timestamp"])
            if val_ts >= test_ts:
                raise WorkspaceValidationError(
                    f"Task '{task_name}' val_timestamp ({val_ts}) must be strictly before test_timestamp ({test_ts})."
                )

    def describe(self) -> dict[str, Any]:
        """Return a compact JSON-serializable overview of the workspace."""
        tables_summary: dict[str, Any] = {}
        for name, meta in self.schema.items():
            tables_summary[name] = {
                "pkey": meta.get("pkey"),
                "time_col": meta.get("time_col"),
                "fkeys": meta.get("fkeys", {}),
                "columns": meta.get("columns", []),
            }

        tasks_summary: dict[str, Any] = {}
        for name, path in self.tasks.items():
            with open(path, "r", encoding="utf-8") as f:
                t = yaml.safe_load(f) or {}
            tasks_summary[name] = {
                "entity_table": t.get("entity_table"),
                "target_col": t.get("target_col"),
                "task_type": t.get("task_type"),
                "timedelta": t.get("timedelta"),
                "test_timestamp": t.get("test_timestamp"),
            }

        return {
            "workspace_dir": str(self.root_dir),
            "tables": tables_summary,
            "tasks": tasks_summary,
            "skills": list(self.skills.keys()),
        }

    def read_skill(self, name: str) -> str:
        """Read markdown text of a skill playbook."""
        clean_name = name.removesuffix(".md")
        if clean_name not in self.skills:
            raise FileNotFoundError(
                f"Skill '{name}' not found. Available skills: {list(self.skills.keys())}"
            )
        return self.skills[clean_name].read_text(encoding="utf-8")

    def get_task_file(self, name: str) -> Path:
        """Get the filesystem path for a task YAML."""
        clean_name = name.removesuffix(".yaml")
        if clean_name not in self.tasks:
            raise FileNotFoundError(
                f"Task '{name}' not found. Available tasks: {list(self.tasks.keys())}"
            )
        return self.tasks[clean_name]
