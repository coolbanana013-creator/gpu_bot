import unittest
from src.utils.indicator_parser import parse_indicator_params, parse_indices


class TestIndicatorParser(unittest.TestCase):
    def test_basic_parse(self):
        s = 'EMA(20)(8.0,71.6,0.0) | RSI(7)(26.8,50.3,26.3)'
        params = parse_indicator_params(s)
        self.assertEqual(len(params), 2)
        self.assertAlmostEqual(params[0][0], 8.0)
        self.assertAlmostEqual(params[1][1], 50.3)

    def test_missing_values(self):
        s = 'SMA(14)(, , 0.5) | MACD()'
        params = parse_indicator_params(s)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0][0], 0.0)
        self.assertEqual(params[1], [0.0, 0.0, 0.0])

    def test_parse_indices(self):
        self.assertEqual(parse_indices('[1,2,3]'), [1,2,3])
        self.assertEqual(parse_indices('1,2,3'), [1,2,3])
        self.assertEqual(parse_indices('EMA(20)(8) | RSI(7)(26)'), [20,7])


if __name__ == '__main__':
    unittest.main()
