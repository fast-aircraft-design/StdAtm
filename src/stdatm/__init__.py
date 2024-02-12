#  This file is part of StdAtm
#  Copyright (C) 2024 ONERA & ISAE-SUPAERO
#  StdAtm is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys

from .atmosphere import Atmosphere, AtmosphereSI  # noqa: F401
from .atmosphere_partials import AtmosphereWithPartials  # noqa: F401

if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "StdAtm"
    __version__ = importlib_metadata.distribution(dist_name).version
except importlib_metadata.PackageNotFoundError:
    __version__ = "unknown"
finally:
    del importlib_metadata, sys
