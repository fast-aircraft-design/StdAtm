=========
Changelog
=========

Version 0.4.2
=============
- Fixed:
    - Removed usage of deprecated pkg_resources. (#20)

Version 0.4.1
=============
Removed

Version 0.4.0
=============
- Changed:
    - delta_t attribute can now be an array, like other attributes. (#18)

Version 0.3.1
=============
- Fixed:
    - speed values were converted to numpy arrays, even if provided as scalars. (#17)

Version 0.3.0
=============
- Added:
    - Added computation of dynamic viscosity. (#12)
    - Added new class AtmosphereWithPartials. Currently computes partial derivatives of state parameters versus altitude. (#13 and #16)

- Changed:
    - Python 3.10 is now officially supported. Python 3.8 is now the minimum required version. (#14)
    - Enhanced overall CPU performances, especially for computation with scalars. (#9 and #11)


Version 0.2.0
=============
- Changed:
  - Added computation of dynamic pressure, impact pressure and calibrated airspeed. (#3)


Version 0.1.0
=============
- First release
