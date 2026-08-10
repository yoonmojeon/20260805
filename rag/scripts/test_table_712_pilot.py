from __future__ import annotations
import unittest
from table_712_pilot import _select_value


class Table712PilotTests(unittest.TestCase):
    def test_minimum_line(self) -> None:
        record = {"value_display": "Z=93.5C₁C₂KShl²\nZₘᵢₙ=2.72K√LSl²"}
        self.assertEqual("Zₘᵢₙ=2.72K√LSl²", _select_value(record, True))



if __name__ == "__main__": unittest.main()
