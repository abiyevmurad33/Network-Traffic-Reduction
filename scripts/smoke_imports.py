import sys
print("Python:", sys.version)

import numpy as np
import pandas as pd
import matplotlib
import torch
import vizdoom as vzd

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("matplotlib:", matplotlib.__version__)
print("torch:", torch.__version__)
print("vizdoom:", vzd.__version__ if hasattr(vzd, "__version__") else "version-unknown")
print("OK: imports successful")
