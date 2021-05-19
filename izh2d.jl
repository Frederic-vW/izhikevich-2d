#!/usr/local/bin/julia
# last tested Julia version: 1.6.1
# Izhikevich model on a 2D lattice
# FvW 03/2018

using NPZ
using PyCall
using PyPlot
using Statistics
using VideoIO
@pyimport matplotlib.animation as anim

function izh2d(N, T, t0, dt, s, D, a, b, c, d, v0, vpeak, I0, stim, blocks)
    # initialize Izhikevich system
    v = v0*ones(Float64,N,N)
    u = zeros(Float64,N,N)
    dv = zeros(Float64,N,N)
    du = zeros(Float64,N,N)
    s_sqrt_dt = s*sqrt(dt)
    X = zeros(Float64,T,N,N)
    # stimulation protocol
    I = zeros(Float64,t0+T,N,N)
    for st in stim
        t_on, t_off = st[1]
        x0, x1 = st[2]
        y0, y1 = st[3]
        I[t0+t_on:t0+t_off, x0:x1, y0:y1] .= I0
    end
    # iterate
    for t in range(1, stop=t0+T, step=1)
        (t%100 == 0) && print("    t = ", t, "/", t0+T, "\r")
        # Izhikevich equations
        dv = 0.04 .* v .* v .+ 5.0*v .+ 140.0 - u + I[t,:,:] + L(v)
        du = a*(b*v - u)
        # Ito stochastic integration
        v += (dv*dt + s_sqrt_dt*randn(N,N))
        u += (du*dt)
        # spiking units
        ipeak = findall(v .> vpeak)
        v[ipeak] .= c # vpeak
        u[ipeak] .+= d
        # dead block(s):
        for bl in blocks
            v[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= c
            u[bl[1][1]:bl[1][2], bl[2][1]:bl[2][2]] .= 0.0
        end
        (t > t0) && (X[t-t0,:,:] = v)
    end
    println("\n")
    return X
end

function animate_pyplot(fname, data)
    """
    Animate 3D array as .mp4 using PyPlot, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    nt, nx, ny = size(data)
    vmin = minimum(data)
    vmax = maximum(data)
    # setup animation image
    fig = figure(figsize=(6,6))
    axis("off")
    t = imshow(data[1,:,:], origin="lower", cmap=ColorMap("gray"),
			   vmin=vmin, vmax=vmax)
    tight_layout()
    # frame generator
    println("[+] animate")
    function animate(i)
        (i%100 == 0) && print("    t = ", i, "/", nt, "\r")
        t.set_data(data[i+1,:,:])
    end
    # create animation
    ani = anim.FuncAnimation(fig, animate, frames=nt, interval=10)
    println("\n")
    # save animation
    ani[:save](fname, bitrate=-1,
               extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    show()
end

function animate_video(fname, data)
    """
    Animate 3D array as .mp4 using VideoIO, save as `fname`
    array dimensions:
        1: time
        2, 3: space
    """
    # BW
    y = UInt8.(round.(255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    # BW inverted
    #y = UInt8.(round.(255 .- 255*(data .- minimum(data))/(maximum(data)-minimum(data))))
    encoder_options = (color_range=2, crf=0, preset="medium")
    framerate=30
    T = size(data,1)
    open_video_out(fname, y[1,end:-1:1,:], framerate=framerate,
                   encoder_options=encoder_options) do writer
        for i in range(2,stop=T,step=1)
            write(writer, y[i,end:-1:1,:])
        end
    end
end

function L(x)
    # Laplace operator
    # periodic boundary conditions
    xU = circshift(x, [-1 0])
    xD = circshift(x, [1 0])
    xL = circshift(x, [0 -1])
    xR = circshift(x, [0 1])
    Lx = xU + xD + xL + xR - 4x
    # non-periodic boundary conditions
    Lx[1,:] .= 0.0
    Lx[end,:] .= 0.0
    Lx[:,1] .= 0.0
    Lx[:,end] .= 0.0
    return Lx
end

function main()
    println("Izhikevich lattice model\n")
    N = 128
    T = 15000
    t0 = 500
    dt = 0.05
    s = 1.0 # 0.02 # 0.10
    D = 0.075
    # Izhikevich parameters: a, b, c, d, v0, vpeak, I
    # TonicSpiking: [0.02, 0.20, -65, 6.00, -70, 30, 14.0]
    # PhasicSpiking: [0.02, 0.25, -65, 6.00, -64, 30,  0.5]
    # TonicBursting: [0.02, 0.20, -50, 2.00, -70, 30, 15.0]
    # PhasicBursting: [0.02, 0.25, -55, 0.05, -64, 30,  0.6]
    # MixedMode: [0.02, 0.20, -55, 4.00, -70, 30, 10.0]
    # Integrator: [0.02, -0.1, -55, 6.00, -60, 30, 9.00]
    # TonicSpiking
    #a, b, c, d, v0, vpeak, I = 0.02, 0.20, -65, 6.0, -70, 30, 14.0
    # PhasicSpiking
    #a, b, c, d, v0, vpeak, I = 0.02, 0.25, -65, 6.00, -64, 30, 1.5
    # TonicBursting:
    a, b, c, d, v0, vpeak, I = 0.02, 0.20, -50, 2.00, -70, 30, 15.0
    println("[+] Lattice size N: ", N)
    println("[+] Time steps T: ", T)
    println("[+] Warm-up steps t0: ", t0)
    println("[+] Integration time step dt: ", dt)
    println("[+] Noise std. dev.: ", s)
    println("[+] Diffusion coefficient D: ", D)
    println("[+] Parameter a: ", a)
    println("[+] Parameter b: ", b)
    println("[+] Parameter c: ", c)
    println("[+] Parameter d: ", d)
    println("[+] Parameter v0: ", v0)
    println("[+] Parameter vpeak: ", vpeak)
    println("[+] Stimulation current I: ", I)

    # stim protocol, array of elements [[t0,t1], [x0,x1], [y0,y1]]
    #stim = [ [[50,550], [1,5], [1,10]] ]
    stim = [ [[1,550], [1,5], [1,10]],
             [[2400,2800], [45,50], [1,30]] ]
    #stim = []

    # dead blocks, array of elementy [[x0,x1], [y0,y1]]
    blocks = [ [[1,20], [10,15]] ]
    #blocks = []

    # run simulation
    data = izh2d(N, T, t0, dt, s, D, a, b, c, d, v0, vpeak, I, stim, blocks)
    println("[+] Data dimensions: ", size(data))

    # plot mean voltage
    m = mean(reshape(data, (T,N*N)), dims=2)
    plot(m, "-k"); show()

    # save data
    I_str = rpad(I, 4, '0') # stim. current amplitude as 4-char string
    s_str = rpad(s, 4, '0') # noise as 4-char string
    D_str = rpad(D, 4, '0') # diffusion coefficient as 4-char string
    fname1 = string("izh2d_I_", I_str, "_s_", s_str, "_D_", D_str, ".npy")
    #npzwrite(fname1, data)
    #println("[+] Data saved as: ", fname1)

    # video
    fname2 = string("izh2d_I_", I_str, "_s_", s_str, "_D_", D_str, ".mp4")
    #animate_pyplot(fname2, data) # slow
    animate_video(fname2, data) # fast
    println("[+] Data saved as: ", fname2)
end

main()
