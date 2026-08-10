from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path


pipeline = importlib.import_module("70_build_precise_table_corpus")


class PreciseTableCorpusTest(unittest.TestCase):
    def test_registry_paths_are_relative_to_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            registry = root / "registry.json"
            registry.write_text(json.dumps({"documents": {"doc": "map.json"},
                                             "source_defaults": {"KR": "kr.json"}}), encoding="utf-8")
            loaded = pipeline.load_registry(registry)
            self.assertEqual(loaded["documents"]["doc"], (root / "map.json").resolve())
            self.assertEqual(loaded["source_defaults"]["KR"], (root / "kr.json").resolve())

    def test_document_mapping_overrides_kr_source_default(self):
        registry = {"documents": {"doc": Path("doc.json")},
                    "source_defaults": {"KR": Path("kr.json")}}
        self.assertEqual(pipeline.resolve_mapping(registry, "doc", "KR"), (Path("doc.json"), "document"))
        self.assertEqual(pipeline.resolve_mapping(registry, "other", "KR"), (Path("kr.json"), "source:KR"))
        self.assertEqual(pipeline.resolve_mapping(registry, "other", "DNV"), (None, "none"))

    def test_serialize_rows_preserves_headers_and_merged_row_label(self):
        structure = {
            "cells": [
                {"cell_id": "R00C00", "row": 0, "column": 0, "rowspan": 1, "type": "header", "text_raw": "길이"},
                {"cell_id": "R00C01", "row": 0, "column": 1, "rowspan": 1, "type": "header", "text_raw": "두께"},
                {"cell_id": "R01C00", "row": 1, "column": 0, "rowspan": 2, "type": "data", "text_raw": "L < 170"},
                {"cell_id": "R01C01", "row": 1, "column": 1, "rowspan": 1, "type": "data", "text_raw": "10.5", "column_header_path": ["두께"]},
                {"cell_id": "R02C01", "row": 2, "column": 1, "rowspan": 1, "type": "data", "text_raw": "11", "column_header_path": ["두께"]},
            ]
        }
        rows = pipeline.serialize_rows({}, structure, {}, {})
        self.assertIn("두께: 10.5", rows[1]["text"])
        self.assertIn("병합 행 머리글=L < 170", rows[2]["text"])

    def test_restored_cell_map_rejects_unknown_glyphs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "restored.json"
            path.write_text(json.dumps({"cells": [{"cell_id": "R00C00", "formula_dominant": True,
                "inline_text_restored": "", "inline_unknown_glyphs": [],
                "formula": {"normalized": "Z=1", "unknown_glyphs": ["U+E999"]}}]}), encoding="utf-8")
            values, unknown = pipeline.restored_cell_map(path, {})
            self.assertEqual(values["R00C00"], "Z=1")
            self.assertEqual(unknown, {"U+E999"})


if __name__ == "__main__":
    unittest.main()
