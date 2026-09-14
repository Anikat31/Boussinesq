import numpy as npp
import math
import jax.numpy as np
import jax.numpy.fft as fft
import jax
import gc
from time import time
import pathlib
import json
import os
from tqdm import tqdm
from sympy import LeviCivita
curr_path = pathlib.Path('./data')
curr_path.mkdir(exist_ok=True, parents=True)
os.listdir(curr_path)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
key = jax.random.PRNGKey(9231472983719)
jax.config.update('jax_enable_x64', True)
forcestart = True
idx = 3
isforcing = False
viscosity_integrator = 'implicit'
if viscosity_integrator == 'explicit':
    isexplicit = 1.0
else:
    isexplicit = 0.0
rank = 0
num_process = 1
N = 256
dt = 0.001
T = 0.1
save_every = 1
f_corr = 1.0
N_bs = [5, 10, 15, 20]
N_b = N_bs[idx]
fvec = np.array([0.0, 0.0, f_corr])
N_bvec = np.array([0, 0, N_b])
PI = np.pi
TWO_PI = 2 * PI
Nf = N // 2 + 1
Np = N // num_process
sx = slice(0, N)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint=False)
dx, dy, dz = (X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0])
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')
Kx = Ky = fft.fftfreq(N, 1.0 / N) * TWO_PI / L
Kz = np.abs(Ky[:Nf])
kx, ky, kz = np.meshgrid(Kx, Ky[sx], Kz, indexing='ij')
kvec = np.array([kx, ky, kz])
@jax.jit
def irfft(x):
    return fft.irfftn(x, (N, N, N), axes=(-3, -2, -1))
@jax.jit
def rfft(x):
    return fft.rfftn(x, axes=(-3, -2, -1))
lp = 8
nu0 = 0.5
m = 1
nu = nu0 * (3 * m / N) ** (2 * (lp - 1 / 3))
fbyN = f_corr / N_b if N_b != 0 else 0.0
einit = 1
nshells = 4
shell_no = np.arange(1, 1 + nshells)
f0 = 0.02 / N_b ** 2 * nshells
re = np.inf if nu == 0 else 1 / nu
if rank == 0:
    print(f' Power input  : {nshells * f0} \nViscosity : {nu}, Re : {re}, dt : {dt}')
nu, f0 * nshells
if nu != 0:
    savePath = curr_path / f'data/bsnq/f_{f_corr:.1f}_Nb_{N_b:.1f}/' f'forced_{isforcing}/N_{N}_Re_{re:.1f}'
else:
    savePath = curr_path / f'data/bsnq/forced_{isforcing}/N_{N}_Re_inf'
if rank == 0:
    print(savePath)
    try:
        savePath.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
lap = -1.0 * (kx ** 2 + ky ** 2 + kz ** 2)
k = np.linalg.norm(kvec, axis=0)
kint = np.clip(np.round(k, 0).astype(int), None, N // 2 + 1)
kh = (kx ** 2 + ky ** 2) ** 0.5
dealias = kint <= N / 3
invlap = dealias / np.where(lap == 0, np.inf, lap)
lapwv = -1.0 * (kx ** 2 + ky ** 2 + fbyN ** 2 * kz ** 2)
invlapwv = dealias / np.where(lapwv == 0, np.inf, lapwv)
vis = nu * k ** (2 * lp)
normalize = np.where((kz == 0) + (kz == N // 2), 1 / (N ** 6 / TWO_PI ** 3), 2 / (N ** 6 / TWO_PI ** 3))
shells = np.arange(-0.5, Nf, 1.0)
shells = shells.at[0].set(0.0)
cond_kx = np.abs(np.round(Kx)) <= N // 3
cond_ky = np.abs(np.round(Ky)) <= N // 3
cond_kz = np.abs(np.round(Kz)) <= N // 3
epsilon = np.array([[[float(LeviCivita(i, j, kk)) for kk in range(3)] for j in range(3)] for i in range(3)])
del kx, ky
@jax.jit
def e3d_to_1d(x):
    return np.histogram(k.ravel(), bins=shells, weights=x.ravel())[0].real
@jax.jit
def vortex(uk, bk, fbyN=fbyN):
    uk_v = 0.0 * uk.copy()
    pv = 1j * (kvec[0] * uk[1] - kvec[1] * uk[0] + kvec[2] * bk * fbyN)
    uk_v = uk_v.at[0].set(-1j * kvec[1] * pv * invlapwv)
    uk_v = uk_v.at[1].set(1j * kvec[0] * pv * invlapwv)
    bk_v = 1j * kvec[2] * pv * invlapwv * fbyN
    return (uk_v, bk_v)
@jax.jit
def forcing(uk, bk):
    uk_v, bk_v = vortex(uk, bk)
    uk_w = uk - uk_v
    bk_w = bk - bk_v
    ek = 0.5 * (np.abs(uk_w[0]) ** 2 + np.abs(uk_w[1]) ** 2 + np.abs(uk_w[2]) ** 2 + np.abs(bk_w) ** 2) * dealias * normalize * (kh > 0.5) * (kz > 0.5)
    ek_arr = e3d_to_1d(ek)
    ek_arr = np.where(np.abs(ek_arr) < 1e-10, 1e20, ek_arr)
    factor = 0.0 * ek_arr.copy()
    factor = factor.at[shell_no].set(f0 / (2 * ek_arr[shell_no]))
    factor3d = factor[kint] * dealias * (kh > 0.5) * (kz > 0.5)
    fk = factor3d[None, ...] * uk_w
    fkb = factor3d * bk_w
    pk = invlap * np.einsum('i...,i...->...', kvec, fk) * dealias
    fk = fk + kvec * pk[None, ...]
    return (fk * isforcing * dealias, fkb * isforcing * dealias)
@jax.jit
def RHS(uk, bk, visc=1, forc=1, isexplicit=isexplicit, invlap=invlap, fvec=fvec, N_bvec=N_bvec):
    fk, fkb = forcing(uk, bk)
    fk *= forc
    fkb *= forc
    u = irfft(uk)
    omg = irfft(1j * np.einsum('ijk,j...,k...->i...', epsilon, kvec, uk))
    rhsk = (rfft(np.einsum('ijk,j...,k...->i...', epsilon, u, omg + fvec[:, None, None, None])) + N_bvec[:, None, None, None] * bk[None, ...] + fk) * dealias[None, ...]
    rhsbk = (-1j * np.einsum('i...,i...->...', kvec, rfft(u * irfft(bk)[None, ...])) - N_b * uk[2] + fkb) * dealias
    pk = 1j * invlap * np.einsum('i...,i...->...', kvec, rhsk)
    return (rhsk - 1j * kvec * pk[None, ...] - (nu * ((-lap) ** lp) * isexplicit * visc)[None, ...] * uk, rhsbk - nu * ((-lap) ** lp) * bk * isexplicit * visc)
@jax.jit
def RK4(i, h, ti, uk, bk, semi_G_half, semi_G, hypervisc):
    ku, kb = RHS(uk, bk)
    uknew = semi_G * uk + h / 6.0 * semi_G * ku
    bknew = semi_G * bk + h / 6.0 * semi_G * kb
    ku, kb = RHS(semi_G_half * (uk + h / 2.0 * ku), semi_G_half * (bk + h / 2.0 * kb))
    uknew += h / 6.0 * 2 * semi_G_half * ku
    bknew += h / 6.0 * 2 * semi_G_half * kb
    ku, kb = RHS(semi_G_half * uk + h / 2.0 * ku, semi_G_half * bk + h / 2.0 * kb)
    uknew += h / 6.0 * 2 * semi_G_half * ku
    bknew += h / 6.0 * 2 * semi_G_half * kb
    ku, kb = RHS(semi_G * uk + semi_G_half * h * ku, semi_G * bk + semi_G_half * h * kb)
    uknew += h / 6.0 * ku
    bknew += h / 6.0 * kb
    u = irfft(uknew * hypervisc)
    b = irfft(bknew * hypervisc)
    uk = rfft(u)
    bnew = rfft(b)
    pk = invlap * np.einsum('i...,i...->...', kvec, uk)
    uknew = uk + kvec * pk[None, ...]
    return (uknew, bnew)
def save(t, uk, bk, step):
    ku, kb = RHS(uk, bk, visc=0, forc=0)
    ek_arr = e3d_to_1d(0.5 * (np.abs(uk[0]) ** 2 + np.abs(uk[1]) ** 2 + np.abs(uk[2]) ** 2 + np.abs(bk) ** 2) * normalize)
    Pik_arr = e3d_to_1d(np.real(np.conjugate(uk[0]) * ku[0] + np.conjugate(uk[1]) * ku[1] + np.conjugate(uk[2]) * ku[2] + np.conjugate(bk) * kb) * dealias * normalize)
    Pik_arr = np.cumsum(Pik_arr[::-1])[::-1]
    uk_v, bk_v = vortex(uk, bk)
    uk_w = uk - uk_v
    bk_w = bk - bk_v
    ek_v_val = np.sum(0.5 * (np.abs(uk_v[0]) ** 2 + np.abs(uk_v[1]) ** 2 + np.abs(uk_v[2]) ** 2 + np.abs(bk_v) ** 2) * normalize)
    ek_w_val = np.sum(0.5 * (np.abs(uk_w[0]) ** 2 + np.abs(uk_w[1]) ** 2 + np.abs(uk_w[2]) ** 2 + np.abs(bk_w) ** 2) * normalize)
    u = irfft(uk)
    b = irfft(bk)
    u.block_until_ready()
    b.block_until_ready()
    ek_arr.block_until_ready()
    Pik_arr.block_until_ready()
    new_dir = savePath / f'step_{step:08d}'
    try:
        new_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    np.savez(f'{new_dir}/Fields_cmp.npz', u=npp.asarray(u[0]), v=npp.asarray(u[1]), w=npp.asarray(u[2]), b=npp.asarray(b))
    np.savez(f'{new_dir}/Energy_spectrum', ek=npp.asarray(ek_arr))
    np.savez(f'{new_dir}/Flux_spectrum', Pik=npp.asarray(Pik_arr))
    eng1 = np.sum(0.5 * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2 + b ** 2) * dx * dy * dz)
    eng2 = np.sum(ek_arr)
    eng1.block_until_ready()
    eng2.block_until_ready()
    print('\n#----------------------------' f'\nStep : {step}' f'\nTime : {float(t)}' f'\nEnergy : {float(eng1)}, {float(eng2)}' '\n#----------------------------')
    print(f'Vortex energy {float(ek_v_val)}, wave energy {float(ek_w_val)}')
    return 'Done!'
def load(t, path):
    uk = np.zeros((3, N, N, Nf), dtype=np.complex128)
    data = np.load(path / 'Fields_cmp.npz')
    uk = uk.at[0].set(rfft(data['u']))
    uk = uk.at[1].set(rfft(data['v']))
    uk = uk.at[2].set(rfft(data['w']))
    bk = rfft(data['b'])
    return (uk, bk)
def evolve_and_save(t, uk, bk):
    h = t[1] - t[0]
    if viscosity_integrator == 'implicit':
        hypervisc = dealias * (1.0 + h * vis) ** -1
    else:
        hypervisc = 1.0
    if viscosity_integrator == 'exponential':
        semi_G = np.exp(-nu * k ** (2 * lp) * h)
        semi_G_half = semi_G ** 0.5
    else:
        semi_G = semi_G_half = 1.0
    simulation_start = time()
    interval_start = simulation_start
    total_steps = len(t) - 1
    print('\n')
    print('=' * 70)
    print('STARTING SIMULATION')
    print('=' * 70)
    print(f'Total RK4 steps : {total_steps}')
    print(f'Saving every    : {save_every} steps')
    print(f'Number of saves : {total_steps // save_every}')
    print('=' * 70)
    print('\n')
    for i in tqdm(range(total_steps)):
        ti = t[i]
        uk, bk = RK4(i, h, ti, uk, bk, semi_G_half, semi_G, hypervisc)
        uk.block_until_ready()
        bk.block_until_ready()
        if (i + 1) % save_every == 0:
            save(ti + h, uk, bk, i + 1)
            uk.block_until_ready()
            bk.block_until_ready()
            interval_end = time()
            interval_elapsed = interval_end - interval_start
            interval_steps = save_every
            interval_iter_per_sec = interval_steps / interval_elapsed
            interval_ms = 1000.0 / interval_iter_per_sec
            total_elapsed = interval_end - simulation_start
            completed_steps = i + 1
            overall_iter_per_sec = completed_steps / total_elapsed
            print('\n')
            print('-' * 70)
            print(f'Completed steps        : {completed_steps}')
            print(f'Steps in last interval : {interval_steps}')
            print(f'Interval wall time     : {interval_elapsed:.6f} s')
            print(f'Speed incl. saving     : {interval_iter_per_sec:.3f} iter/s')
            print(f'Time incl. saving      : {interval_ms:.3f} ms/iter')
            print(f'Overall average speed  : {overall_iter_per_sec:.3f} iter/s')
            print('-' * 70)
            interval_start = time()
    uk.block_until_ready()
    bk.block_until_ready()
    simulation_end = time()
    total_elapsed = simulation_end - simulation_start
    total_iter_per_sec = total_steps / total_elapsed
    total_ms_per_iter = 1000.0 / total_iter_per_sec
    print('\n')
    print('=' * 70)
    print('FINAL SIMULATION PERFORMANCE')
    print('=' * 70)
    print(f'Total RK4 steps       : {total_steps}')
    print(f'Total wall-clock time : {total_elapsed:.6f} s')
    print(f'Average iterations/s  : {total_iter_per_sec:.3f} iter/s')
    print(f'Average time/step     : {total_ms_per_iter:.3f} ms')
    print(f'Saving interval       : every {save_every} steps')
    print('=' * 70)
    print('\n')
    return (uk, bk)
if not forcestart:
    print('Found existing simulation! Using last saved data.')
    paths = savePath / 'last'
    print(f'Loading data from {paths}')
    tinit = 0.0
    uk, bk = load(tinit, paths)
    trm = kx * uk[0] + ky * uk[1] + kz * uk[2]
    uk = uk + (invlap * trm)[None, ...] * kvec
    u = irfft(uk)
    b = irfft(bk)
if forcestart:
    kinit = 31
    th = jax.random.uniform(key, shape=kvec.shape, minval=0, maxval=TWO_PI)
    eprofile = kint ** 2 * np.exp(-kint ** 2 / 2) / normalize
    amp = (eprofile / np.where(kint == 0, np.inf, kint ** 2)) ** 0.5
    uk = (amp * (kint ** 2 < kinit ** 2) * (kint > 0) * dealias)[None, ...] * np.exp(1j * th)
    print(uk.shape)
    u = irfft(uk)
    uk = rfft(u)
    bk = 0.0 * uk[0].copy()
    pk = np.einsum('i...,i...->...', kvec, uk)
    uk += (invlap * pk)[None, ...] * kvec
    uk_v, bk_v = vortex(uk, bk)
    uk, bk = (uk - uk_v, bk - bk_v)
    tinit = 0.0
    ek_arr0 = e3d_to_1d(0.5 * (np.abs(uk[0]) ** 2 + np.abs(uk[1]) ** 2 + np.abs(uk[2]) ** 2 + np.abs(bk) ** 2) * normalize)
    e0 = np.sum(ek_arr0)
    uk *= (einit / e0) ** 0.5
    bk *= (einit / e0) ** 0.5
    u = irfft(uk)
    b = irfft(bk)
    ek_arr0 = e3d_to_1d(0.5 * (np.abs(uk[0]) ** 2 + np.abs(uk[1]) ** 2 + np.abs(uk[2]) ** 2 + np.abs(bk) ** 2) * normalize)
    print(ek_arr0, np.sum(ek_arr0))
    uk_v, bk_v = vortex(uk, bk)
    uk_w, bk_w = (uk - uk_v, bk - bk_v)
    del th, amp
ek_cross = (np.einsum('ipqr,ipqr->', np.conjugate(uk_w), uk_v * normalize) + np.einsum('pqr,pqr->', np.conjugate(bk_w), bk_v * normalize)).real
ek_v = np.sum(0.5 * (np.abs(uk_v[0]) ** 2 + np.abs(uk_v[1]) ** 2 + np.abs(uk_v[2]) ** 2 + np.abs(bk_v) ** 2) * normalize)
ek_w = np.sum(0.5 * (np.abs(uk_w[0]) ** 2 + np.abs(uk_w[1]) ** 2 + np.abs(uk_w[2]) ** 2 + np.abs(bk_w) ** 2) * normalize)
ek_arr0 = ek_arr0.at[0:shell_no[0]].set(0.0)
ek_arr0 = ek_arr0.at[shell_no[-1] + 1:].set(0.0)
divmax = np.max(np.abs(np.einsum('i...,i...->...', kvec, uk)))
print(divmax)
print(ek_cross, ek_v, ek_w)
gc.collect()
h = dt
if viscosity_integrator == 'implicit':
    hypervisc = dealias * (1.0 + h * vis) ** -1
else:
    hypervisc = 1.0
if viscosity_integrator == 'exponential':
    semi_G = np.exp(-nu * k ** (2 * lp) * h)
    semi_G_half = semi_G ** 0.5
else:
    semi_G = semi_G_half = 1.0
print('\n')
print('=' * 70)
print('JIT WARM-UP')
print('=' * 70)
print('Compiling RK4 ...')
t_compile_start = time()
uk_warm, bk_warm = RK4(0, h, 0, uk, bk, semi_G_half, semi_G, hypervisc)
uk_warm.block_until_ready()
bk_warm.block_until_ready()
t_compile_end = time()
print(f'First RK4 call completed in {t_compile_end - t_compile_start:.6f} s')
print('JIT compilation / warm-up complete.')
print('=' * 70)
print('\n')
t = np.arange(0, T + h, h)
uk, bk = evolve_and_save(t, uk, bk)
