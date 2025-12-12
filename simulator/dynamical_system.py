from phi.flow import field, math


def reaction_diffusion(
    n,
    tu,
    i,
    u,
    dn,
    dtu,
    di,
    du,
    r1,
    r2,
    a1,
    a2,
    a3,
    b1,
    b2,
    c1,
    c2,
    c3,
    c4,
    d1,
    d2,
    uc,
    s,
    rho,
    alpha,
    dt,
):
    sn = dn * field.laplace(n) + r1 * n * (1 - b1 * n) - c1 * n * tu - a1 * (1 - math.exp(-u)) * n
    stu = dtu * field.laplace(tu) + r2 * tu * (1 - b2 * tu) - c2 * tu * n - c3 * tu * i - a2 * (1 - math.exp(-u)) * tu
    si = di * field.laplace(i) + s + rho * i * tu / (alpha + tu) - c4 * i * tu - d1 * i - a3 * (1 - math.exp(-u)) * i
    su = du * field.laplace(u) - d2 * u + uc
    return n + dt * sn, tu + dt * stu, i + dt * si, u + dt * su
