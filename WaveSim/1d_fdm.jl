# Pressure Field at p(x,t-dt)
p_prev = zeros(nx)
# Pressure Field at p(x,t)
p = zeros(nx)
# Pressure Field at p(x,t+dt)
p_next = zeros(nx)

# Looping over time
for it in range(start=2, stop=nt)
    # Looping over Space
    for ix in range(start=2, stop=nx-1)
        # Evaluating 2nd derivative wrt x
        d2p_dx2 = (p[ix+1] - 2*p[ix] + p[ix-1])/(dx^2)
        # Updating Solution
        if ix == isrc
            p_next[ix] = (c*dt)^2 * d2p_dx2 + 2*p[ix] - p_prev[ix] + dt^2 * src[it]
        else
            p_next[ix] = (c*dt)^2 * d2p_dx2 + 2*p[ix] - p_prev[ix]
        end
    end

    # Boundary conditions
    if boundary == "zero"
        p_next[1] = 0
        p_next[nx] = 0
    elseif boundary == "neumann"
        p_next[1] = p_next[2]
        p_next[nx] = p_next[nx-1]
    elseif boundary == "absorbing"
        p_next[1] = p[2] + (c*dt-dx)/(c*dt+dx) * (p_next[2]-p[1])
        p_next[nx] = p[nx-1] + (c*dt-dx)/(c*dt+dx) * (p_next[nx-1]-p[nx])
    end

    # Current Sol becomes Previous Sol
    p_prev[:] = p[:]
    # Next Sol becomes Current Sol
    p[:] = p_next[:]
end