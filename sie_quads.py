import numpy as np
from scipy.optimize import fsolve


def sie_alpha(x, y, theta_E, q):
    if abs(q - 1.0) < 1e-6:
        r = np.hypot(x, y)
        if r < 1e-12: return 0.0, 0.0
        return theta_E * x / r, theta_E * y / r
    eps = np.sqrt(1.0 - q * q)
    psi = np.sqrt(q * q * x * x + y * y)
    if psi < 1e-12: return 0.0, 0.0
    ax = theta_E * np.sqrt(q) / eps * np.arctan(eps * x / psi)
    ay = theta_E * np.sqrt(q) / eps * np.arctanh(eps * y / psi)
    return ax, ay


def _lens_eq(t, beta, theta_E, q):
    ax, ay = sie_alpha(t[0], t[1], theta_E, q)
    return [t[0] - ax - beta[0], t[1] - ay - beta[1]]


def _find_images(beta_x, beta_y, theta_E, q, tol=1e-7):
    R = 1.8 * theta_E
    n = 28
    xs = np.linspace(-R, R, n)
    ys = np.linspace(-R, R, n)
    seen = []
    for x0 in xs:
        for y0 in ys:
            try:
                sol, info, ier, _ = fsolve(
                    _lens_eq, [x0, y0],
                    args=([beta_x, beta_y], theta_E, q),
                    xtol=1e-10, full_output=True)
                if ier != 1: continue
                res = _lens_eq(sol, [beta_x, beta_y], theta_E, q)
                if abs(res[0]) > tol or abs(res[1]) > tol: continue
                if any(np.hypot(sol[0]-s[0], sol[1]-s[1]) < 0.01*theta_E for s in seen):
                    continue
                seen.append(sol)
            except Exception:
                continue
    return np.array(seen)


def image_locs_from_sie(theta_E_arcsec=1.6, q=0.7, PA_deg=30.0,
                        morphology='cross', prtb=0.3,
                        npix=59, pix_size=0.2):
    """
    prtb = source offset from the symmetric ideal, in caustic-scale units:
        0   -> source at the symmetric centre / on-axis ideal
        ~1  -> source approaching the caustic boundary

    Tested valid prtb ranges at q=0.7, theta_E=1.6":
        cross   :  any prtb in (0, 1.5]  (always 4 images)
        cusp    :  0.05 <= prtb <= 0.75  (4 images; beyond -> double)
        fold    :  0.05 <= prtb <= 0.50  (4 images; beyond -> double)
        double  :  0.00 <= prtb <= 1.00  (2 images on lens major axis)

    The function does NOT raise if Nim != expected; it returns whatever the
    solver finds, so sweeping prtb across the quad->double transition is
    natural.
    """
    c = (1.0 - q) * theta_E_arcsec
    if morphology == 'cross':
        beta = (0.05 * prtb * theta_E_arcsec, 0.05 * prtb * theta_E_arcsec)
    elif morphology == 'cusp':
        beta = (0.0, 0.9 * prtb * c)
    elif morphology == 'fold':
        beta = (0.55 * prtb * c, 0.55 * prtb * c)
    elif morphology == 'double':
        beta = ((0.30 + 0.60 * prtb) * theta_E_arcsec, 0.0)
    else:
        raise ValueError("morphology must be 'cross', 'cusp', 'fold', or 'double'")

    imgs = _find_images(beta[0], beta[1], theta_E_arcsec, q)
    if len(imgs) < 2:
        raise RuntimeError(
            f"SIE solver found {len(imgs)} images. Source likely outside the "
            f"radial caustic — reduce prtb (got {prtb} for '{morphology}')."
        )

    angles = np.arctan2(imgs[:, 1], imgs[:, 0])
    imgs = imgs[np.argsort(angles)]

    pa = np.deg2rad(PA_deg)
    R = np.array([[np.cos(pa), -np.sin(pa)],
                  [np.sin(pa),  np.cos(pa)]])
    imgs_rot = imgs @ R.T

    cx = cy = npix // 2
    locs = [(int(round(cx + x / pix_size)),
             int(round(cy + y / pix_size))) for (x, y) in imgs_rot]
    return locs, imgs_rot, np.array(beta)
