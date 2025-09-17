import numpy as np
import matplotlib.pyplot as plt
from qutils.orbital import OE2ECI, dim2NonDim6
from qutils.integrators import ode87

OENoThrust = np.load("classifier/OEArrayNoThrust.npz")["OEArrayNoThrust"]
dataset_labels = np.load("classifier/dataset_orbit_labels.npz")["dataset_orbit_labels"]


# search dataset_labels for one leo, heo, meo, geo


leo_index = 0
for i in range(len(dataset_labels)):
    if dataset_labels[i] == "meo":
        meo_index = i
    if dataset_labels[i] == "heo":
        heo_index = i
    if dataset_labels[i] == "geo":
        geo_index = i

heo_index = heo_index - 10

leoOE = OENoThrust[leo_index,0,0:6]
heoOE = OENoThrust[heo_index,0,0:6]
meoOE = OENoThrust[meo_index,0,0:6]
geoOE = OENoThrust[geo_index,0,0:6]

leo_period = OENoThrust[leo_index,0,6]
heo_period = OENoThrust[heo_index,0,6]
meo_period = OENoThrust[meo_index,0,6]
geo_period = OENoThrust[geo_index,0,6]

print(leoOE)
print(heoOE)
print(meoOE)
print(geoOE)

leoECI = OE2ECI(leoOE) * 1000
heoECI = OE2ECI(heoOE) * 1000
meoECI = OE2ECI(meoOE) * 1000
geoECI = OE2ECI(geoOE) * 1000

print(leoECI)
print(heoECI)
print(meoECI)
print(geoECI)

mu = 3.986004418e14  # Earth’s mu in m^3/s^2
R = 6371e3 # radius of earth in m
DU = R
TU = np.sqrt(R**3/mu) # time unit in seconds

def twoBodyJ2(t, y, p=mu):
    # two body problem with J2 perturbation in 6 dimensions taken from astroforge library
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_models.py
    # https://github.com/mit-ll/AstroForge/blob/main/src/astroforge/force_models/_forces.py

    # x, v = np.split(y, 2) # orginal line in Astroforge
    # faster than above
    x = y[:3]
    v = y[3:]

    J2 = 4.84165368e-4 * np.sqrt(5)

    M2 = J2 * np.diag(np.array([0.5, 0.5, -1.0]))
    r = np.sqrt(x @ x) # faster than np.linalg.norm(x) (original line in Astroforge)

    # compute monopole force
    F0 = -mu * x / r**3

    # compute the quadropole force in ITRS
    acc = (mu * R**2 / r**5) * (-5 * x * (x @ M2 @ x) / r**2 + 2 * M2 @ x) + F0

    # ydot = np.hstack((v, acc)) # orginal line in Astroforge
    # faster than above
    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc

    return ydot

_, y_leo = ode87(twoBodyJ2,[0,leo_period],leoECI,rtol=1e-12,atol=1e-15)
_, y_meo = ode87(twoBodyJ2,[0,meo_period],meoECI,rtol=1e-12,atol=1e-15)
_, y_heo = ode87(twoBodyJ2,[0,heo_period],heoECI,rtol=1e-12,atol=1e-15)
_, y_geo = ode87(twoBodyJ2,[0,geo_period],geoECI,rtol=1e-12,atol=1e-15)

def plotOrbit(ax,y,label):


    ax.plot(y[:, 0], y[:, 1], y[:, 2], label=label)
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.legend(fontsize=10)
    ax.grid(True)

    plt.rcParams.update({'font.size': 10})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
axisFontsize = 7
plt.rcParams.update({'font.size': axisFontsize})
ax.set_xlabel('x [m]',fontsize=axisFontsize)
ax.set_ylabel('y [m]',fontsize=axisFontsize)
ax.set_zlabel('z [m]',fontsize=axisFontsize)
ax.plot(0, 0, 0, 'ko', label='Earth')

plotOrbit(ax,y_leo,'leo')
plotOrbit(ax,y_heo,'heo')
plotOrbit(ax,y_meo,'meo')
plotOrbit(ax,y_geo,'geo')
plt.show()