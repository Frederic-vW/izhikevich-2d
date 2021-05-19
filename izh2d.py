#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Izhikevich model on a 2D lattice
# FvW 03/2018

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def izh2d(N, T, t0, dt, s, D, a, b, c, d, v0, vpeak, I0, stim, blocks):
    # initialize Izhikevich system
    v, u = v0*np.ones((N,N)), 0.0*np.ones((N,N))
    dv, du = np.zeros((N,N)), np.zeros((N,N))
    s_sqrt_dt = s*np.sqrt(dt)
    X = np.zeros((T,N,N))
    # stimulation protocol
    I = np.zeros((t0+T,N,N))
    #I = I0*np.ones((t0+T,N,N))
    for st in stim:
        t_on, t_off = st[0]
        x0, x1 = st[1]
        y0, y1 = st[2]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] = I0

    # iterate
    for t in range(1, t0+T):
        if (t%100 == 0): print(f"    t = {t:d}/{t0+T:d}\r", end="")
        # Izhikevich equations
        dv = 0.04*v*v + 5*v + 140.0 - u + L(v) + I[t,:,:]
        du = a*(b*v - u)
        # Ito stochastic integration
        v += (dv*dt + s_sqrt_dt*np.random.randn(N,N))
        u += (du*dt)
        # spiking units
        ipeak = np.where(v > vpeak)
        v[ipeak] = c # vpeak
        u[ipeak] += d
        # dead block(s):
        for bl in blocks:
            v[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = c
            u[bl[0][0]:bl[0][1], bl[1][0]:bl[1][1]] = 0.0
        if (t >= t0):
            X[t-t0,:,:] = v
    print("\n")
    return X


def animate_video(fname, x):
    # BW
    y = 255 * (x-x.min()) / (x.max()-x.min())
    # BW inverted
    #y = 255 * ( 1 - (x-x.min()) / (x.max()-x.min()) )
    y = y.astype(np.uint8)
    nt, nx, ny = x.shape
    #print(f"nt = {nt:d}, nx = {nx:d}, ny = {ny:d}")
    # write video using opencv
    frate = 30
    out = cv2.VideoWriter(fname, \
                          cv2.VideoWriter_fourcc(*'mp4v'), \
                          frate, (nx,ny))
    print("[+] Animate:")
    for i in range(0,nt):
        print(f"\ti = {i:d}/{nt:d}\r", end="")
        img = np.ones((nx, ny, 3), dtype=np.uint8)
        for j in range(3): img[:,:,j] = y[i,::-1,:]
        out.write(img)
    out.release()
    print("")


def L(x):
    # Laplace operator
    # periodic boundary conditions
    xU = np.roll(x, shift=-1, axis=0)
    xD = np.roll(x, shift=1, axis=0)
    xL = np.roll(x, shift=-1, axis=1)
    xR = np.roll(x, shift=1, axis=1)
    Lx = xU + xD + xL + xR - 4*x
    # non-periodic boundary conditions
    Lx[0,:] = 0.0
    Lx[-1,:] = 0.0
    Lx[:,0] = 0.0
    Lx[:,-1] = 0.0
    return Lx


def main():
    print("Izhikevich lattice model\n")
    # a, b, c, d, v0, vpeak, I
    # TonicSpiking: [0.02, 0.20, -65, 6.00, -70, 30, 14.0]
    # PhasicSpiking: [0.02, 0.25, -65, 6.00, -64, 30,  0.5]
    # TonicBursting: [0.02, 0.20, -50, 2.00, -70, 30, 15.0]
    # PhasicBursting: [0.02, 0.25, -55, 0.05, -64, 30,  0.6]
    # MixedMode: [0.02, 0.20, -55, 4.00, -70, 30, 10.0]
    # Integrator: [0.02, -0.1, -55, 6.00, -60, 30, 9.00]
    N = 128
    T = 15000
    t0 = 500
    dt = 0.05
    s = 1.0 # 0.02 # 0.10
    D = 0.075
    # TonicSpiking
    #a, b, c, d, v0, vpeak, I = 0.02, 0.20, -65, 6.0, -70, 30, 14.0
    # PhasicSpiking
    #a, b, c, d, v0, vpeak, I = 0.02, 0.25, -65, 6.00, -64, 30, 1.5
    # TonicBursting:
    a, b, c, d, v0, vpeak, I = 0.02, 0.20, -50, 2.00, -70, 30, 15.0
    print("[+] Lattice size N: ", N)
    print("[+] Time steps T: ", T)
    print("[+] Warm-up steps t0: ", t0)
    print("[+] Integration time step dt: ", dt)
    print("[+] Noise intensity: ", s)
    print("[+] Diffusion coefficient D: ", D)
    print("[+] Parameter a: ", a)
    print("[+] Parameter b: ", b)
    print("[+] Parameter c: ", c)
    print("[+] Parameter d: ", d)
    print("[+] Parameter v0: ", v0)
    print("[+] Parameter vpeak: ", vpeak)
    print("[+] Stimulation current I: ", I)

    # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    #stim = [ [[50,550], [1,5], [1,10]] ]
    stim = [ [[0,550], [0,5], [0,10]],
             [[2400,3000], [45,50], [0,30]] ]
    #stim = []
    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[1,20], [10,15]] ]
    #blocks = []
    # run simulation
    data = izh2d(N, T, t0, dt, s, D, a, b, c, d, v0, vpeak, I, stim, blocks)
    print("[+] Data dimensions: ", data.shape)

    # plot mean voltage
    m = np.mean(np.reshape(data, (T,N*N)), axis=1)
    plt.figure(figsize=(12,4))
    plt.plot(m, "-k")
    plt.tight_layout()
    plt.show()

    # save data
    #fname1 = f"izh2d_I_{I:.2f}_s_{s:.2f}_D_{D:.2f}.npy"
    #npzwrite(fname1, data)
    #println("[+] Data saved as: ", fname1)

    fname2 = f"izh2d_I_{I:.2f}_s_{s:.2f}_D_{D:.2f}.mp4"
    animate_video(fname2, data)
    print("[+] Data saved as: ", fname2)


if __name__ == "__main__":
    os.system("clear")
    main()
