# =============================================================================
# Project     : CEP Rocket Simulation
# File        : core_sim.py
# Author      : Oka Sudiana
# Created     : 2026-10-06
# Updated     : 2026-10-06
# Version     : 1.0
# =============================================================================
#
# Description:
#     Core 6-DOF rocket flight dynamics simulation.
#
# =============================================================================


# -*- coding: utf-8 -*-
"""
core_sim.py
Refactored 6-DOF CEP simulation engine (physics core, no GUI).

Refactor notes vs. the original cep_simulation.py:
- All geometry/mass/inertia values that were hardcoded inside
  equations_of_motion() and get_aero_forces_moments_opt() are now carried
  in a Geometry dataclass and passed explicitly.
- Xthrust is a new, separate parameter for the thrust application point
  (original code assumed nozzle location == total length L).
- Burn_time is NOT a user input: it is read directly from data_thrust's
  own time axis (last timestamp of the thrust curve).
- The aerodynamic Moment Reference Center (MRC) is assumed equal to
  Xcg_wet (see caveat in the accompanying report / GUI notes).
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from dataclasses import dataclass

G0 = 9.80665
R_AIR = 287.05287
GAMMA = 1.4


# ==============================================================================
# Atmosphere
# ==============================================================================
def get_atmosphere(alt):
    if alt < 0:
        alt = 0.0
    if alt < 11000.0:
        T = 288.15 - 0.0065 * alt
        P = 101325.0 * (T / 288.15) ** 5.25588
    elif alt < 20000.0:
        T = 216.65
        P = 22632.0 * np.exp(-G0 * (alt - 11000.0) / (R_AIR * T))
    elif alt < 32000.0:
        T = 216.65 + 0.001 * (alt - 20000.0)
        P = 5474.89 * (T / 216.65) ** (-G0 / (R_AIR * 0.001))
    else:
        T = 228.65
        P = 868.0 * np.exp(-G0 * (alt - 32000.0) / (R_AIR * T))
    rho = P / (R_AIR * T)
    a = np.sqrt(GAMMA * R_AIR * T)
    return T, P, rho, a


# ==============================================================================
# Geometry (user-supplied, fixed for a given design case)
# ==============================================================================
@dataclass
class Geometry:
    D: float            # m
    L: float             # m
    mass_wet: float       # kg
    mass_dry: float       # kg
    Ixx_wet: float        # kg.m2
    Iyy_wet: float
    Izz_wet: float
    Ixx_dry: float
    Iyy_dry: float
    Izz_dry: float
    Xcg_wet: float        # m, from nose
    Xcg_dry: float        # m, from nose
    Xthrust: float        # m, from nose (nozzle / thrust application point)
    L_rail: float         # m
    theta0_deg: float     # rail elevation, deg
    psi0_deg: float       # rail azimuth, deg

    @property
    def S(self):
        return np.pi * self.D ** 2 / 4.0

    @property
    def MRC(self):
        # Assumption: aero database moment reference center == Xcg_wet
        return self.Xcg_wet


# ==============================================================================
# Aerodynamic database (loaded from data_aero .mat)
# ==============================================================================
class AeroDB:
    REQUIRED_KEYS = ['ALPHA', 'MACH', 'CA', 'CN', 'CYB', 'CLLB', 'CLNB',
                      'CLLP', 'CMQ', 'CLNR', 'XCP']

    def __init__(self, mat_path):
        data = sio.loadmat(mat_path)
        missing = [k for k in self.REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"data_aero .mat file is missing required variables: {missing}. "
                f"Expected schema: {self.REQUIRED_KEYS}"
            )
        self.alpha_grid = data['ALPHA'].flatten()
        self.mach_grid = data['MACH'].flatten()
        self.CA = data['CA']
        self.CN = data['CN']
        self.CYB = data['CYB']
        self.CLLB = data['CLLB']
        self.CLNB = data['CLNB']
        self.CLLP = data['CLLP']
        self.CMQ = data['CMQ']
        self.CLNR = data['CLNR']
        self.XCP = data['XCP']
        self.n_alpha = len(self.alpha_grid)
        self.n_mach = len(self.mach_grid)
        self.a_lo, self.a_hi = float(self.alpha_grid[0]), float(self.alpha_grid[-1])

    def interp(self, alpha, mach):
        a_clipped = np.clip(alpha, self.a_lo, self.a_hi)
        m_clipped = np.clip(mach, self.mach_grid[0], self.mach_grid[-1])

        # alpha index via searchsorted (grid spacing assumed monotonic, not
        # necessarily uniform 1.0 deg as in the original hardcoded version)
        idx_a = np.searchsorted(self.alpha_grid, a_clipped) - 1
        idx_a = int(np.clip(idx_a, 0, self.n_alpha - 2))
        da = self.alpha_grid[idx_a + 1] - self.alpha_grid[idx_a]
        w_a = (a_clipped - self.alpha_grid[idx_a]) / da if da != 0 else 0.0

        idx_m = np.searchsorted(self.mach_grid, m_clipped) - 1
        idx_m = int(np.clip(idx_m, 0, self.n_mach - 2))
        dm = self.mach_grid[idx_m + 1] - self.mach_grid[idx_m]
        w_m = (m_clipped - self.mach_grid[idx_m]) / dm if dm != 0 else 0.0

        w00 = (1 - w_a) * (1 - w_m)
        w10 = w_a * (1 - w_m)
        w01 = (1 - w_a) * w_m
        w11 = w_a * w_m

        def bilerp(mat):
            return (w00 * mat[idx_a, idx_m] + w10 * mat[idx_a + 1, idx_m] +
                    w01 * mat[idx_a, idx_m + 1] + w11 * mat[idx_a + 1, idx_m + 1])

        return {
            'ca': bilerp(self.CA), 'cn': bilerp(self.CN), 'cyb': bilerp(self.CYB),
            'cllb': bilerp(self.CLLB), 'clnb': bilerp(self.CLNB),
            'cllp': bilerp(self.CLLP), 'cmq': bilerp(self.CMQ),
            'clnr': bilerp(self.CLNR), 'xcp': bilerp(self.XCP),
        }


# ==============================================================================
# Thrust / mass-flow database (loaded from data_thrust .mat). Burn duration
# is derived from the file's own time axis, not a separate user input.
# ==============================================================================
class ThrustDB:
    def __init__(self, mat_path):
        data = sio.loadmat(mat_path)
        for k in ['RTX200TCsimTh', 'massflow']:
            if k not in data:
                raise ValueError(
                    f"data_thrust .mat file is missing required variable '{k}'. "
                    f"Expected schema: 'RTX200TCsimTh' [t, thrust_kgf], "
                    f"'massflow' [t, mass_burned_kg]"
                )
        t_thrust = data['RTX200TCsimTh'][:, 0]
        f_thrust_kgf = data['RTX200TCsimTh'][:, 1]
        t_mass = data['massflow'][:, 0]
        m_burned = data['massflow'][:, 1]

        # Burn duration comes directly from the thrust curve's own time axis
        # (last timestamp with nonzero/tabulated thrust) -- not a separate
        # user input.
        self.burn_time = float(t_thrust[-1])
        f_thrust_N = f_thrust_kgf * G0

        self.get_thrust = interp1d(t_thrust, f_thrust_N, bounds_error=False,
                                    fill_value=0.0)
        self.get_mass_burned = interp1d(t_mass, m_burned, bounds_error=False,
                                         fill_value=(0.0, m_burned[-1]))
        self.m_prop_nominal_from_curve = float(m_burned[-1])


# ==============================================================================
# Monte Carlo perturbation set (mirrors original MissileParams uncertainty
# factors; geometry itself is NOT perturbed here, only these scale factors)
# ==============================================================================
@dataclass
class Perturb:
    u0: float = 0.1
    theta0: float = 0.0   # rad, absolute (sampled around nominal rail elevation)
    psi0: float = 0.0     # rad, absolute
    f_Ixx: float = 1.0
    f_Iyy: float = 1.0
    f_Izz: float = 1.0
    f_m: float = 1.0
    V_wx: float = 0.0
    V_wy: float = 0.0
    f_T: float = 1.0
    epsilon_y: float = 0.0
    epsilon_z: float = 0.0
    f_CA: float = 1.0
    f_CN: float = 1.0
    f_CY: float = 1.0
    f_CLL: float = 1.0
    f_CM: float = 1.0
    f_CLN: float = 1.0


# ==============================================================================
# Equations of motion
# ==============================================================================
def make_eom(geo: Geometry, aero: AeroDB, thrust: ThrustDB):
    m_prop_nominal = geo.mass_wet - geo.mass_dry

    def eom(t, y, p: Perturb):
        x, y_pos, z, u, v, w, pr, qy, r, q0, q1, q2, q3 = y

        q_len = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        q0, q1, q2, q3 = q0 / q_len, q1 / q_len, q2 / q_len, q3 / q_len

        Rib = np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
        ])

        dx = Rib[0, 0] * u + Rib[0, 1] * v + Rib[0, 2] * w
        dy = Rib[1, 0] * u + Rib[1, 1] * v + Rib[1, 2] * w
        dz = Rib[2, 0] * u + Rib[2, 1] * v + Rib[2, 2] * w

        alt = -z
        _, _, rho, a_snd = get_atmosphere(alt)

        V_w_inertial = np.array([p.V_wx, p.V_wy, 0.0])
        V_w_body = Rib.T @ V_w_inertial
        u_rel = u - V_w_body[0]
        v_rel = v - V_w_body[1]
        w_rel = w - V_w_body[2]
        V_rel = np.sqrt(u_rel ** 2 + v_rel ** 2 + w_rel ** 2)

        if V_rel > 1e-4:
            mach = V_rel / a_snd
            alpha = np.arctan2(w_rel, u_rel) * (180.0 / np.pi)
            beta = np.arctan2(v_rel, u_rel) * (180.0 / np.pi)
        else:
            mach, alpha, beta = 0.0, 0.0, 0.0

        m_wet_p = geo.mass_wet * p.f_m
        m_dry_p = geo.mass_dry * p.f_m
        m_prop_p = m_wet_p - m_dry_p

        Ixx_wet_p = geo.Ixx_wet * p.f_Ixx
        Iyy_wet_p = geo.Iyy_wet * p.f_Iyy
        Izz_wet_p = geo.Izz_wet * p.f_Izz
        Ixx_dry_p = geo.Ixx_dry * p.f_Ixx
        Iyy_dry_p = geo.Iyy_dry * p.f_Iyy
        Izz_dry_p = geo.Izz_dry * p.f_Izz

        mb_nom = np.clip(thrust.get_mass_burned(t), 0.0, m_prop_nominal)
        f = mb_nom / m_prop_nominal if m_prop_nominal > 0 else 0.0
        f = np.clip(f, 0.0, 1.0)

        mass = m_wet_p - f * m_prop_p
        xcg = geo.Xcg_wet + f * (geo.Xcg_dry - geo.Xcg_wet)
        ixx = Ixx_wet_p + f * (Ixx_dry_p - Ixx_wet_p)
        iyy = Iyy_wet_p + f * (Iyy_dry_p - Iyy_wet_p)
        izz = Izz_wet_p + f * (Izz_dry_p - Izz_wet_p)

        c = aero.interp(alpha, mach)
        ca = c['ca'] * p.f_CA
        cn = c['cn'] * p.f_CN
        cyb = c['cyb'] * p.f_CY
        cllb = c['cllb'] * p.f_CLL
        cllp = c['cllp'] * p.f_CLL
        cmq = c['cmq']
        clnb = c['clnb']
        clnr = c['clnr'] * p.f_CLN
        xcp = c['xcp']

        q_inf = 0.5 * rho * V_rel ** 2
        S = geo.S
        D = geo.D

        Fax = -ca * q_inf * S
        Fay = cyb * beta * q_inf * S
        Faz = -cn * q_inf * S

        if V_rel > 1.0:
            p_hat = pr * D / (2.0 * V_rel)
            q_hat = qy * D / (2.0 * V_rel)
            r_hat = r * D / (2.0 * V_rel)
        else:
            p_hat = q_hat = r_hat = 0.0

        Cl = cllb * beta + cllp * p_hat
        Laero = Cl * q_inf * S * D

        Cm_cg = (cn * xcp) + cn * (xcg - geo.MRC) / D + cmq * q_hat
        Cm_cg *= p.f_CM
        Maero = Cm_cg * q_inf * S * D

        Cn_cg = (clnb * beta) + (cyb * beta) * (xcg - geo.MRC) / D + clnr * r_hat
        Cn_cg *= p.f_CLN
        Naero = Cn_cg * q_inf * S * D

        T = p.f_T * thrust.get_thrust(t)
        Tx = T * np.cos(p.epsilon_y) * np.cos(p.epsilon_z)
        Ty = T * np.cos(p.epsilon_y) * np.sin(p.epsilon_z)
        Tz = -T * np.sin(p.epsilon_y)

        d_nozzle = geo.Xthrust - xcg
        Mt_y = d_nozzle * Tz
        Mt_z = -d_nozzle * Ty

        Fgb = Rib.T @ np.array([0.0, 0.0, mass * G0])
        Fgx, Fgy, Fgz = Fgb

        s = np.sqrt(x ** 2 + y_pos ** 2 + z ** 2)
        if s < geo.L_rail:
            du = (Tx + Fax + Fgx) / mass
            dv = dw = dp = dqy = dr = 0.0
            dq0 = dq1 = dq2 = dq3 = 0.0
        else:
            du = (Tx + Fax + Fgx) / mass - (qy * w - r * v)
            dv = (Ty + Fay + Fgy) / mass - (r * u - pr * w)
            dw = (Tz + Faz + Fgz) / mass - (pr * v - qy * u)

            dp = (Laero + (iyy - izz) * qy * r) / ixx
            dqy = (Maero + Mt_y + (izz - ixx) * r * pr) / iyy
            dr = (Naero + Mt_z + (ixx - iyy) * pr * qy) / izz

            dq0 = 0.5 * (-pr * q1 - qy * q2 - r * q3)
            dq1 = 0.5 * (pr * q0 + r * q2 - qy * q3)
            dq2 = 0.5 * (qy * q0 - r * q1 + pr * q3)
            dq3 = 0.5 * (r * q0 + qy * q1 - pr * q2)

        return [dx, dy, dz, du, dv, dw, dp, dqy, dr, dq0, dq1, dq2, dq3]

    return eom


def ground_strike_event(t, y, p):
    if t < 1.0:
        return 1.0
    return y[2]


ground_strike_event.terminal = True
ground_strike_event.direction = 1


# ==============================================================================
# Single-run simulation
# ==============================================================================
def run_single_simulation(geo: Geometry, aero: AeroDB, thrust: ThrustDB,
                           p: Perturb, full_output=False):
    """Returns (x_impact, y_impact) or, if full_output, a metrics dict."""
    theta0 = p.theta0
    psi0 = p.psi0
    phi0 = 0.0

    q0i = np.cos(phi0/2)*np.cos(theta0/2)*np.cos(psi0/2) + np.sin(phi0/2)*np.sin(theta0/2)*np.sin(psi0/2)
    q1i = np.sin(phi0/2)*np.cos(theta0/2)*np.cos(psi0/2) - np.cos(phi0/2)*np.sin(theta0/2)*np.sin(psi0/2)
    q2i = np.cos(phi0/2)*np.sin(theta0/2)*np.cos(psi0/2) + np.sin(phi0/2)*np.cos(theta0/2)*np.sin(psi0/2)
    q3i = np.cos(phi0/2)*np.cos(theta0/2)*np.sin(psi0/2) - np.sin(phi0/2)*np.sin(theta0/2)*np.cos(psi0/2)

    y0 = [0.0, 0.0, 0.0, p.u0, 0.0, 0.0, 0.0, 0.0, 0.0, q0i, q1i, q2i, q3i]
    eom = make_eom(geo, aero, thrust)

    try:
        sol = solve_ivp(eom, (0.0, 150.0), y0, args=(p,), method='RK45',
                         events=ground_strike_event, rtol=1e-6, atol=1e-8,
                         max_step=0.05)
    except Exception:
        return (np.nan, np.nan) if not full_output else None

    if not full_output:
        return sol.y[0, -1], sol.y[1, -1]

    t = sol.t
    X, Y, Z = sol.y[0], sol.y[1], sol.y[2]
    U, V, W = sol.y[3], sol.y[4], sol.y[5]
    alt = -Z
    speed = np.sqrt(U ** 2 + V ** 2 + W ** 2)

    mach = np.zeros_like(t)
    for i in range(len(t)):
        _, _, _, a_snd = get_atmosphere(alt[i])
        mach[i] = speed[i] / a_snd if a_snd > 0 else 0.0

    i_apogee = int(np.argmax(alt))
    x_imp, y_imp = X[-1], Y[-1]

    return {
        't': t, 'x': X, 'y': Y, 'alt': alt, 'speed': speed, 'mach': mach,
        'total_flight_time': float(t[-1]),
        'apogee_altitude': float(alt[i_apogee]),
        'apogee_time': float(t[i_apogee]),
        'impact_range': float(np.sqrt(x_imp ** 2 + y_imp ** 2)),
        'impact_x': float(x_imp), 'impact_y': float(y_imp),
        'max_velocity': float(np.max(speed)),
        'max_mach': float(np.max(mach)),
    }


# ==============================================================================
# Monte Carlo sample generation (uncertainty magnitudes follow the original
# script's defaults)
# ==============================================================================
def generate_samples(method, N, geo: Geometry, seed=42):
    rng = np.random.default_rng(seed)

    u0_nom = 0.1
    theta0_nom = np.deg2rad(geo.theta0_deg)
    psi0_nom = np.deg2rad(geo.psi0_deg)

    dv = 2.0
    dtheta = np.deg2rad(2.0)
    dpsi = np.deg2rad(2.0)
    dI = 0.02
    dm = 0.02
    dw = 2.0
    dT = 0.02
    depsilon = np.deg2rad(0.5)
    dCA, dCN, dCY, dCLL, dCM, dCLN = 0.02, 0.05, 0.05, 0.10, 0.10, 0.10

    samples = []

    if method == 'uniform':
        for _ in range(N):
            u0 = rng.uniform(u0_nom - dv, u0_nom + dv)
            theta0 = rng.uniform(theta0_nom - dtheta, theta0_nom + dtheta)
            psi0 = rng.uniform(psi0_nom - dpsi, psi0_nom + dpsi)
            f_Ixx = rng.uniform(1 - dI, 1 + dI)
            f_Iyy = rng.uniform(1 - dI, 1 + dI)
            f_Izz = rng.uniform(1 - dI, 1 + dI)
            f_m = rng.uniform(1 - dm, 1 + dm)
            w_speed = rng.uniform(0.0, dw)
            w_dir = rng.uniform(0.0, 2 * np.pi)
            V_wx = w_speed * np.cos(w_dir)
            V_wy = w_speed * np.sin(w_dir)
            f_T = rng.uniform(1 - dT, 1 + dT)
            eps_y = rng.uniform(-depsilon, depsilon)
            eps_z = rng.uniform(-depsilon, depsilon)
            f_CA = rng.uniform(1 - dCA, 1 + dCA)
            f_CN = rng.uniform(1 - dCN, 1 + dCN)
            f_CY = rng.uniform(1 - dCY, 1 + dCY)
            f_CLL = rng.uniform(1 - dCLL, 1 + dCLL)
            f_CM = rng.uniform(1 - dCM, 1 + dCM)
            f_CLN = rng.uniform(1 - dCLN, 1 + dCLN)
            samples.append(Perturb(u0, theta0, psi0, f_Ixx, f_Iyy, f_Izz, f_m,
                                    V_wx, V_wy, f_T, eps_y, eps_z, f_CA, f_CN,
                                    f_CY, f_CLL, f_CM, f_CLN))

    elif method == 'gaussian':
        for _ in range(N):
            samples.append(Perturb(
                u0=rng.normal(u0_nom, dv/3), theta0=rng.normal(theta0_nom, dtheta/3),
                psi0=rng.normal(psi0_nom, dpsi/3),
                f_Ixx=rng.normal(1, dI/3), f_Iyy=rng.normal(1, dI/3), f_Izz=rng.normal(1, dI/3),
                f_m=rng.normal(1, dm/3), V_wx=rng.normal(0, dw/3), V_wy=rng.normal(0, dw/3),
                f_T=rng.normal(1, dT/3), epsilon_y=rng.normal(0, depsilon/3), epsilon_z=rng.normal(0, depsilon/3),
                f_CA=rng.normal(1, dCA/3), f_CN=rng.normal(1, dCN/3), f_CY=rng.normal(1, dCY/3),
                f_CLL=rng.normal(1, dCLL/3), f_CM=rng.normal(1, dCM/3), f_CLN=rng.normal(1, dCLN/3),
            ))

    elif method == 'mcmc':
        mu = np.array([u0_nom, theta0_nom, psi0_nom, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1])
        sigmas = np.array([dv, dtheta, dpsi, dI, dI, dI, dm, dw, dw, dT,
                            depsilon, depsilon, dCA, dCN, dCY, dCLL, dCM, dCLN]) / 3.0
        curr_p = mu.copy()

        def log_target(pp):
            if pp[0] < 0.01 or pp[3] <= 0 or pp[4] <= 0 or pp[5] <= 0 or pp[6] <= 0 or pp[9] <= 0:
                return -np.inf
            return -0.5 * np.sum(((pp - mu) / sigmas) ** 2)

        curr_log_prob = log_target(curr_p)
        prop_sigmas = 0.2 * sigmas
        total_steps = 500 + 2 * N
        chain = []
        for step in range(total_steps):
            proposal = curr_p + rng.normal(0.0, prop_sigmas)
            prop_log_prob = log_target(proposal)
            if np.log(rng.uniform(0, 1)) < (prop_log_prob - curr_log_prob):
                curr_p, curr_log_prob = proposal, prop_log_prob
            if step >= 500 and (step - 500) % 2 == 0:
                chain.append(curr_p.copy())
        for pp in chain[:N]:
            samples.append(Perturb(pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6],
                                    pp[7], pp[8], pp[9], pp[10], pp[11], pp[12],
                                    pp[13], pp[14], pp[15], pp[16], pp[17]))
    else:
        raise ValueError(f"Unknown method: {method}")

    return samples


# ==============================================================================
# OAT sensitivity analysis
# ==============================================================================
def run_sensitivity_analysis(geo: Geometry, aero: AeroDB, thrust: ThrustDB):
    nom = Perturb(theta0=np.deg2rad(geo.theta0_deg), psi0=np.deg2rad(geo.psi0_deg))
    x_ref, y_ref = run_single_simulation(geo, aero, thrust, nom)

    dv, dtheta, dpsi = 2.0, np.deg2rad(2.0), np.deg2rad(2.0)
    dI, dm, dw, dT = 0.02, 0.02, 2.0, 0.02
    depsilon = np.deg2rad(0.5)
    dCA, dCN, dCY, dCLL, dCM, dCLN = 0.02, 0.05, 0.05, 0.10, 0.10, 0.10
    th0, ps0 = nom.theta0, nom.psi0

    def P(**kw):
        base = dict(theta0=th0, psi0=ps0)
        base.update(kw)
        return Perturb(**base)

    scenarios = {
        'Initial Velocity': [P(u0=0.1 + dv), P(u0=0.1 - dv)],
        'Elevation': [P(theta0=th0 + dtheta), P(theta0=th0 - dtheta)],
        'Azimuth': [P(psi0=ps0 + dpsi), P(psi0=ps0 - dpsi)],
        'Ixx': [P(f_Ixx=1 + dI), P(f_Ixx=1 - dI)],
        'Iyy': [P(f_Iyy=1 + dI), P(f_Iyy=1 - dI)],
        'Izz': [P(f_Izz=1 + dI), P(f_Izz=1 - dI)],
        'mass_wet': [P(f_m=1 + dm), P(f_m=1 - dm)],
        'Wind crossing': [P(V_wx=dw), P(V_wy=dw), P(V_wx=-dw), P(V_wy=-dw)],
        'Thrust': [P(f_T=1 + dT), P(f_T=1 - dT)],
        'Thrust misalignment x': [P(epsilon_y=depsilon), P(epsilon_y=-depsilon)],
        'Thrust misalignment y': [P(epsilon_z=depsilon), P(epsilon_z=-depsilon)],
        'CA (Aero)': [P(f_CA=1 + dCA), P(f_CA=1 - dCA)],
        'CN (Aero)': [P(f_CN=1 + dCN), P(f_CN=1 - dCN)],
        'CY (Aero)': [P(f_CY=1 + dCY), P(f_CY=1 - dCY)],
        'CLL (Aero)': [P(f_CLL=1 + dCLL), P(f_CLL=1 - dCLL)],
        'CM (Aero)': [P(f_CM=1 + dCM), P(f_CM=1 - dCM)],
        'CLN (Aero)': [P(f_CLN=1 + dCLN), P(f_CLN=1 - dCLN)],
    }

    sens = {}
    for name, plist in scenarios.items():
        shifts = []
        for p in plist:
            x, y = run_single_simulation(geo, aero, thrust, p)
            if np.isnan(x):
                continue
            shifts.append(np.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2))
        sens[name] = max(shifts) if shifts else 0.0

    return sens, (x_ref, y_ref)
