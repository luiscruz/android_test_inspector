"""Common-Language Effect Size"""
from statistics import mean, variance
from math import sqrt
from scipy.stats import norm

def _cles_comp(a,b):
  # Mean difference between x and y
  diff = mean(b) - mean(a)
  
  # Standard deviation of difference
  stdev = sqrt(variance(a) + variance(b))
  
  # Probability derived from normal distribution
  # that random x is higher than random y -
  # or in other words, that diff is larger than 0.
  # p.norm <- 1 - pnorm(0, diff, sd = stdev)
  p_b_higher_than_x = 1 - norm(0, stdev).pdf(diff)
  # Return result
  return p_b_higher_than_x

def cles_brute(a,b):
    def _score_diff(diff):
        if diff > 0:
            return 1
        if diff == 0:
            return 0.5
        return 0

    differences_scores = [
        _score_diff(y-x)
        for x in a
        for y in b
    ]
    return mean(differences_scores)
