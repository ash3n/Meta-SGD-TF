from .classes import *

import os
C_ = namespace()
C_.homepath = 'C:/Users/Alex'
C_.workspace = os.path.relpath(C_.homepath)

from .exu import *
from . import nlp