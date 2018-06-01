from unittest import TestCase
import toolbox.alignment as al



class TestFactorial(TestCase):

    def test_FD_rule_bins(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        x = [i for i in range(100)]
        result  = al.FD_rule_bins(x)
        self.assertEqual(int(result[1]), int(21.32890343))