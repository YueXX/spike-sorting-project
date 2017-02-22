
import numpy as np

def param2str(p):
  """Converts parameter to string.

  Args:
    p: Parameter (float).

  Returns:
    p_str: String corresponding to p.
  """
  if p == 0:
    p_str = str(0)
  else:
    p_pow = int(np.floor(np.log10(p)))
    p_first = abs(int(p /(10 ** p_pow)))
    p_str = str(p_first) + "e" + str(p_pow)
  return p_str