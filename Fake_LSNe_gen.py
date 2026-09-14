import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaincinv, gamma


def sersic_analytical_totalflux(n_sersic=3.0, R50_arcsec=1.2, I_e=20.0, ellip=0.3):
    q = 1.0 - ellip
    b_n = gammaincinv(2.0 * n_sersic, 0.5)
    return (I_e * R50_arcsec**2 * 2.0 * np.pi * n_sersic
            * np.exp(b_n) * gamma(2.0 * n_sersic)
            / (b_n ** (2.0 * n_sersic)) * q)


def light_curve(t, A_LC=1.0, T0_LC=20.0):
    t = np.asarray(t, dtype=float)
    F = A_LC * t**2 * np.exp(-t / T0_LC)
    return np.where(t > 0.0, F, 0.0)


def gaussian_kernel_2d(fwhm_arcsec, npix=59, pix_size=0.2):
    if fwhm_arcsec <= 0:
        ker = np.zeros((npix, npix))
        ker[npix // 2, npix // 2] = 1.0
        return ker
    sigma_pix = (fwhm_arcsec / pix_size) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    c = (npix - 1) / 2.0
    y, x = np.mgrid[0:npix, 0:npix]
    r2 = (x - c) ** 2 + (y - c) ** 2
    ker = np.exp(-0.5 * r2 / sigma_pix**2)
    return ker / ker.sum()


def matching_kernel(fwhm_target, fwhm_template, npix=59, pix_size=0.2):
    if fwhm_target < fwhm_template:
        raise ValueError(f"target FWHM ({fwhm_target}) must be >= template FWHM ({fwhm_template})")
    fwhm_k = np.sqrt(fwhm_target**2 - fwhm_template**2)
    return gaussian_kernel_2d(fwhm_k, npix=npix, pix_size=pix_size)


def sersic_2d(npix, pix_size, R50_arcsec, n_sersic=4.0,
              I_e=1.0, ellip=0.0, theta=0.0, center=None):
    if center is None:
        center = ((npix - 1) / 2.0, (npix - 1) / 2.0)
    cx, cy = center
    q   = 1.0 - ellip
    b_n = gammaincinv(2.0 * n_sersic, 0.5)
    def _I_of_r(r):
        return I_e * np.exp(-b_n * ((r / R50_arcsec) ** (1.0 / n_sersic) - 1.0))
    def _ellip_r(dx, dy):
        xr =  dx * np.cos(theta) + dy * np.sin(theta)
        yr = -dx * np.sin(theta) + dy * np.cos(theta)
        return np.sqrt(xr**2 + (yr / q) ** 2)
    y, x = np.mgrid[0:npix, 0:npix]
    dx = (x - cx) * pix_size
    dy = (y - cy) * pix_size
    img = _I_of_r(_ellip_r(dx, dy))
    inner_r_pix = max(5, int(np.ceil(2.0 * R50_arcsec / pix_size)))
    oversample  = 15
    rr_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = rr_pix <= inner_r_pix
    if mask.any():
        s = (np.arange(oversample) - (oversample - 1) / 2.0) / oversample
        sx, sy = np.meshgrid(s, s)
        xi = x[mask][:, None, None]
        yi = y[mask][:, None, None]
        dxi = (xi - cx + sx) * pix_size
        dyi = (yi - cy + sy) * pix_size
        sub = _I_of_r(_ellip_r(dxi, dyi))
        img[mask] = sub.mean(axis=(-2, -1))
    return img * pix_size**2


def make_lensed_frame(t_obs, psf, lrg_intrinsic,
                      image_locs, mags, time_delays, rng,
                      A_LC=1.0, T0_LC=20.0,
                      npix=59, var_bg=1.0, C_t=0.05):
    src   = np.zeros((npix, npix), dtype=float)
    F_vec = np.zeros(0, dtype=float)
    if image_locs is not None:
        image_locs  = list(image_locs)
        mags        = np.asarray(mags, dtype=float)
        time_delays = np.asarray(time_delays, dtype=float)
        Nim = len(image_locs)
        if not (len(mags) == len(time_delays) == Nim):
            raise ValueError("image_locs, mags, time_delays must all have length Nim")
        if time_delays[0] != 0.0:
            raise ValueError("time_delays[0] must be 0 (image 0 defines t=0)")
        F_vec = np.empty(Nim, dtype=float)
        for i, (loc, mu, dt) in enumerate(zip(image_locs, mags, time_delays)):
            F_i = mu * float(light_curve(t_obs - dt, A_LC=A_LC, T0_LC=T0_LC))
            m, n = loc
            src[n, m] += F_i
            F_vec[i] = F_i
    I_points = fftconvolve(src,           psf, mode="same")
    I_lrg    = fftconvolve(lrg_intrinsic, psf, mode="same")
    I_clean  = I_points + I_lrg
    var_map  = var_bg + C_t * np.maximum(I_clean, 0.0)
    bg_noise      = rng.normal(0.0, np.sqrt(var_bg), (npix, npix))
    poisson_noise = rng.normal(0.0, np.sqrt(np.maximum(C_t * I_clean, 0.0)))
    I = I_clean + bg_noise + poisson_noise
    return I.astype(np.float32), var_map.astype(np.float32), F_vec


def generate_sample(times, fwhms,
                    image_locs, mags, time_delays,
                    R50_arcsec, n_sersic=4.0, I_e=1.0,
                    ellip=0.0, theta=0.0, lrg_center=None,
                    rng=None,
                    A_LC=1.0, T0_LC=20.0,
                    npix=59, pix_size=0.2,
                    var_bg=1.0, C_t=0.05):
    if rng is None:
        rng = np.random.default_rng()
    times = np.atleast_1d(times).astype(float)
    fwhms = np.atleast_1d(fwhms).astype(float)
    if times.shape != fwhms.shape:
        raise ValueError(f"times and fwhms must have the same shape; got {times.shape} vs {fwhms.shape}")
    if image_locs is not None:
        image_locs  = list(image_locs)
        mags        = np.asarray(mags, dtype=float)
        time_delays = np.asarray(time_delays, dtype=float)
        Nim = len(image_locs)
        if not (len(mags) == len(time_delays) == Nim):
            raise ValueError(f"image_locs, mags, time_delays must all have length Nim; got {Nim}, {len(mags)}, {len(time_delays)}")
        if time_delays[0] != 0.0:
            raise ValueError("time_delays[0] must be 0 (image 0 defines t=0)")
    Nframes = len(times)
    lrg = sersic_2d(npix=npix, pix_size=pix_size,
                    R50_arcsec=R50_arcsec, n_sersic=n_sersic,
                    I_e=I_e, ellip=ellip, theta=theta, center=lrg_center)
    frames = []
    Nim_out = len(image_locs) if image_locs is not None else 0
    lightcurves = np.zeros((Nim_out, Nframes), dtype=float)
    for j, (t_j, fwhm_j) in enumerate(zip(times, fwhms)):
        psf_j = gaussian_kernel_2d(fwhm_j, npix=npix, pix_size=pix_size)
        I, V, F = make_lensed_frame(t_j, psf_j, lrg,
                                    image_locs, mags, time_delays, rng,
                                    A_LC=A_LC, T0_LC=T0_LC,
                                    npix=npix, var_bg=var_bg, C_t=C_t)
        frames.append({"image": I, "var": V,
                       "psf":   psf_j.astype(np.float32),
                       "t":     float(t_j),
                       "fwhm":  float(fwhm_j),
                       "F":     F})
        if Nim_out > 0:
            lightcurves[:, j] = F
    return frames, lightcurves, lrg.astype(np.float32)


def matching_kernels_from_fwhms(fwhm_template, fwhms_visits,
                                npix=59, pix_size=0.2):
    fwhms_visits = np.atleast_1d(fwhms_visits).astype(float)
    out = np.empty((len(fwhms_visits), npix, npix), dtype=float)
    for i, fwhm_v in enumerate(fwhms_visits):
        out[i] = matching_kernel(fwhm_v, fwhm_template, npix=npix, pix_size=pix_size)
    return out