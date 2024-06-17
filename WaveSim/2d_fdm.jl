# Pressure Field at p(x,z,t)
p = zeros(nx, nz)
# Pressure Field at p(x,z,t-dt)
p_prev = zeros(nx, nz)
# Pressure Field at p(x,z,t+dt)
p_next = zeros(nx, nz)

# Looping over time
for it in range(start=2, stop=nt)
    # Looping over Space
    for ix in range(start=2, stop=nx-1)
        for iz in range(start=2, stop=nz-1)
            # Evaluating 2nd derivative wrt x
            d2p_dx2 = (p[ix+1,iz] - 2*p[ix,iz] + p[ix-1,iz])/(dx^2)
            # Evaluating 2nd derivative wrt z
            d2p_dz2 = (p[ix,iz+1] - 2*p[ix,iz] + p[ix,iz-1])/(dz^2)
            # Updating Solution
            if ix == isrc[1] && iz == isrc[2]
                p_next[ix,iz] = (c[ix,iz]*dt)^2 * (d2p_dx2 + d2p_dz2) + 2*p[ix,iz] - p_prev[ix,iz] + dt^2 * src[it]
            else
                p_next[ix,iz] = (c[ix,iz]*dt)^2 * (d2p_dx2 + d2p_dz2) + 2*p[ix,iz] - p_prev[ix,iz]
            end
        end
    end

    # Boundary conditions
    if boundary == "zero"
        p_next[1,:] .= 0
        p_next[nx,:] .= 0
        p_next[:,1] .= 0
        p_next[:,nz] .= 0
    elseif boundary == "neumann"
        p_next[1,:] .= p_next[2,:]
        p_next[nx,:] .= p_next[nx-1,:]
        p_next[:,1] .= p_next[:,2]
        p_next[:,nz] .= p_next[:,nz-1]
    elseif boundary == "absorbing"
        p_next[1,:] = p[2,:] + (c[1,:]*dt .- dx) ./ (c[1,:]*dt .+ dx) .* (p_next[2,:]-p[1,:])
        p_next[nx,:] = p[nx-1,:] + (c[nx,:]*dt .- dx) ./ (c[nx,:]*dt .+ dx) .* (p_next[nx-1,:]-p[nx,:])
        p_next[:,1] = p[:,2] + (c[:,1]*dt .- dx) ./ (c[:,1]*dt .+ dx) .* (p_next[:,2]-p[:,1])
        p_next[:,nz] = p[:,nz-1] + (c[:,nz]*dt .- dx) ./ (c[:,nz]*dt .+ dx) .* (p_next[:,nz-1]-p[:,nz])
    end

    # Current Sol becomes Previous Sol
    p_prev[:,:] = p[:,:]
    # Next Sol becomes Current Sol
    p[:,:] = p_next[:,:]
end