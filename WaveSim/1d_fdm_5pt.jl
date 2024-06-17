# Pressure Field at p(x,t)
p = zeros(nx)
# Pressure Field at p(x,t-dt)
p_prev = zeros(nx)
# Pressure Field at p(x,t+dt)
p_next = zeros(nx)

# Looping over time
for it in range(start=2, stop=nt)
    # Looping over Space
    for ix in range(start=3, stop=nx-2)
        # Evaluating 2nd derivative wrt x
        d2p_dx2 = (-1/12 * p[ix+2] + 4/3  * p[ix+1] - 5/2 * p[ix] +4/3  * p[ix - 1] - 1/12 * p[ix - 2])/(dx^2)
        # Updating Solution
        if ix == isrc
            p_next[ix] = (c*dt)^2 * d2p_dx2 + 2*p[ix] - p_prev[ix] + dt^2 * src[it]
        else
            p_next[ix] = (c*dt)^2 * d2p_dx2 + 2*p[ix] - p_prev[ix]
        end
    end

    # Boundary conditions
    p_next[1:2] .= 0
    p_next[nx-1:nx] .= 0

    # Current Sol becomes Previous Sol
    p_prev[:] = p[:]
    # Next Sol becomes Current Sol
    p[:] = p_next[:]
end