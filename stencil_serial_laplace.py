from numba import stencil, jit
from numba import njit, prange
import numba
import numpy as np
from matplotlib import pyplot


vel = 3000.0
f0 = 15
nt = 50
dt = .002
nx = 50
ny = 50
nz = 50
#nx = 50
#ny = 50
#nz = 50
h = vel/(f0*5.0)
x_idx = nx//2
y_idx = ny//2
z_idx = nz//2
#finite difference coefficients for different order of stencils
#radius = 1
#coeff = [-2, 1] 
#radius = 4
#coeff = [-2.8472222, 1.6, -.2, 2.53968/100]
radius = 8
coeff = [-3.0548446, 1.777777, -3.1111111/10, 7.572087/100, -1.767676/100, 3.480962/1000, -5.180005/10000, 5.074287/(10*10*10*10*10), -2.42812/np.power(10, 6)]



@stencil(neighborhood = ((-radius, radius), (-radius, radius), (-radius, radius),), standard_indexing=("coeff",) )
#@jit(nopython=True, parallel=True)
#@numba.njit(parallel=True)
def laplace(a, coeff):
	laplace = coeff[0]*a[0, 0, 0]*3
	for i in range(1, radius+1):
		laplace+= coeff[i]*((a[0, 0, -i] + a[0, 0, i]) + (a[0, -i, 0] + a[0, i, 0]) + (a[-i, 0, 0] + a[i, 0, 0]))
	return laplace






def Ricker(t, f0):
	r  = (np.pi*f0 * (t-1./f0))
	return (1-2.*r**2)*np.exp(-r**2)

source = []
for i in range(nt):
	source.append(Ricker(dt*i, f0))


prev_timestep = np.zeros((nx+2*radius)*(ny+2*radius)*(nz+2*radius)).reshape((nx+2*radius, ny+2*radius, nz+2*radius))
curr_timestep = np.zeros((nx+2*radius)*(ny+2*radius)*(nz+2*radius)).reshape((nx+2*radius, ny+2*radius, nz+2*radius))
next_timestep = np.zeros((nx+2*radius)*(ny+2*radius)*(nz+2*radius)).reshape((nx+2*radius, ny+2*radius, nz+2*radius))
curr_timestep[x_idx+radius, y_idx+radius, z_idx+radius] += source[0]*(vel*dt)**2

#@jit(nopython=False, parallel=True)
#@numba.njit(parallel=True)
#@numba.njit(nopython=True, parallel=True)
def time_step(next_timestep, curr_timestep, prev_timestep, nt):
	for i in range(1, nt): 
		next_timestep = 2*curr_timestep - prev_timestep + laplace(curr_timestep, coeff)*((dt*vel/h)**2) 
		next_timestep[x_idx+radius, y_idx+radius, z_idx+radius] += source[i]*(vel*dt)**2
		prev_timestep = curr_timestep
		curr_timestep = next_timestep
		print(next_timestep[x_idx+radius, y_idx+radius, z_idx+radius])

	return next_timestep


next_timestep = time_step(next_timestep, curr_timestep, prev_timestep, nt)

fig, ax = pyplot.subplots()
next_timestep = (next_timestep[:, :, z_idx + radius]) 
cax = ax.imshow(next_timestep, interpolation='nearest', cmap='hot')
ax.set_title('Heatmap of pressure')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
min, max = np.amin(next_timestep), np.amax(next_timestep) 
print ("min, max of pressure array are: %f, %f" %(min, max))
cbar = fig.colorbar(cax, ticks=[min,(max+min)/2.0, max])
cbar.ax.set_yticklabels(['%.2f' %min, '%.2f' %((max+min)/2.0) ,'%.2f' %max])  
pyplot.savefig('run_pressure.png')
