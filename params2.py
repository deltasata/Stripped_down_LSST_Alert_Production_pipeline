import numpy as np
import importlib, sie_quads
importlib.reload(sie_quads)
from sie_quads import image_locs_from_sie

fwhm_tmpl   = 0.6
t_tmpl      = 0.0
t_visits    = [8,    12,  20,  35,  45,  55,  90,   105,  175]
fwhm_visits = [0.7, 1.0, 1.2, 2.1, 1.1, 1.4, 1.61, 0.7, 1.1]

R50_arcsec, n_sersic, I_e, ellip, theta_deg = 1.2, 4.0, 20.0, 0.3, -30.0

A_LC, T0_LC = 0.5, 20.0
var_bg, C_t = 1.0, 0.05
npix, pix_size = 101, 0.2

seed = 42

# Lens: SIE aligned with the LRG (same q, same PA).
# prtb = source offset from the symmetric ideal, in caustic-scale units.
#   prtb=0   -> source at symmetric centre / on-axis ideal
#   prtb~1   -> source near caustic boundary
# Tested valid prtb at q=0.7, theta_E=1.6":
#   morphology='cross'   prtb in (0, 1.5]      (always 4 images)
#   morphology='cusp'    0.05 <= prtb <= 0.75  (4 images; beyond -> double)
#   morphology='fold'    0.05 <= prtb <= 0.50  (4 images; beyond -> double)
#   morphology='double'  0.00 <= prtb <= 1.00  (2 images on lens major axis)
theta_E_arcsec = 2.4
morphology     = 'cross'
prtb           = 0.4

image_locs, _, _ = image_locs_from_sie(
    theta_E_arcsec=theta_E_arcsec,
    q=1.0 - ellip,
    PA_deg=theta_deg,
    morphology=morphology,
    prtb=prtb,
    npix=npix, pix_size=pix_size,
)

Nim         = len(image_locs)
mags        = [1.0] * Nim
time_delays = list(np.linspace(0.0, 30.0, Nim)) if Nim > 1 else [0.0]

mags        = [1.0,1.1,4.8,1.5,2.0,10.0][:Nim]
time_delays = [0.0,10,20.0,30.0,40.0,50.0][:Nim]