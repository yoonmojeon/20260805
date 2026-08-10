from __future__ import annotations

import unittest

from hancomeqn_restore import Glyph, restore_formula, restore_inline


def glyph(char: str, x: float, y: float, size: float = 9.0) -> Glyph:
    return Glyph(char, (x, y - size, x + size / 2, y), (x, y), size, "HancomEQN")


class RestoreFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = {"glyphs": {"U+E019": {"text": "Z"}, "U+E047": {"text": "="},
                                    "U+E002": {"text": "C"}, "U+E034": {"text": "1"},
                                    "U+E035": {"text": "2"}}}

    def test_scripts(self) -> None:
        chars = [glyph("\ue019", 0, 10), glyph("\ue047", 5, 10), glyph("\ue002", 10, 10),
                 glyph("\ue034", 14, 12.5, 6), glyph("\ue035", 18, 6, 6)]
        result = restore_formula(chars, self.mapping)
        self.assertEqual("Z=C₁²", result["display"])
        self.assertEqual("Z=C_{1}^{2}", result["latex"])
        self.assertFalse(result["needs_review"])

    def test_unknown_is_not_silenced(self) -> None:
        result = restore_formula([glyph("\ue111", 0, 10)], self.mapping)
        self.assertTrue(result["needs_review"])
        self.assertEqual(["U+E111"], result["unknown_glyphs"])

    def test_sqrt_keeps_superscript(self) -> None:
        mapping = {"glyphs": {**self.mapping["glyphs"], "U+E06D": {"text": "√", "role": "sqrt"}}}
        chars = [glyph("\ue06d", 0, 10, 10), glyph("l", 5, 10), glyph("\ue035", 9, 6, 6)]
        result = restore_formula(chars, mapping)
        self.assertEqual("√l²", result["display"])
        self.assertEqual(r"\sqrt{l^{2}}", result["latex"])

    def test_composite_delimiter_pieces_are_silenced(self) -> None:
        mapping = {
            "glyphs": {
                **self.mapping["glyphs"],
                "U+E078": {"text": "{", "role": "delimiter_anchor"},
                "U+E079": {"text": "", "role": "delimiter_piece"},
                "U+E07A": {"text": "", "role": "delimiter_piece"},
            }
        }
        chars = [glyph("\ue078", 0, 10), glyph("\ue079", 4, 10), glyph("\ue07a", 8, 10)]
        result = restore_formula(chars, mapping)
        self.assertEqual("{", result["display"])
        self.assertFalse(result["needs_review"])
        inline, unknown = restore_inline("\ue078\ue079\ue07a", mapping)
        self.assertEqual("{", inline)
        self.assertEqual([], unknown)


if __name__ == "__main__": unittest.main()
