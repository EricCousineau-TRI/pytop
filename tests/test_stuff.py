import unittest

import pytop


class TestStuff(unittest.TestCase):
    def test_affinity_str(self):
        s = pytop.affinity_str([0], cpu_count=2)
        self.assertEqual(s, "[0]")

        s = pytop.affinity_str([0, 1, 2, 3, 4], cpu_count=5)
        self.assertEqual(s, "<all>")

        s = pytop.affinity_str([0, 1, 2, 3, 10, 11], cpu_count=11)
        self.assertEqual(s, "[0-3, 10, 11]")


if __name__ == "__main__":
    unittest.main()
