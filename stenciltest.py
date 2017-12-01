from numba import stencil
import numpy as np


#finite difference coefficients for different order of stencils
radius = 1
coeff = [-2, 1] 
#radius = 4
#coeff = [-2.8472222, 1.6, -.2, 2.53968/100]
#radius = 8
#coeff = [-3.0548446, 1.777777, -3.1111111/10, 7.572087/100, -1.767676/100, 3.480962/1000, -5.180005/10000, 5.074287/(10*10*10*10*10), -2.42812/np.pow(10, 6)]


@stencil(neighborhood = ((-radius, radius), (-radius, radius), (-radius, radius),), standard_indexing=("coeff",) )
#@stencil(neighborhood = ((-radius, radius), (-radius, radius), (-radius, radius),))
def laplace(a, coeff):
#def laplace(a):
	laplace = coeff[0]*a[0, 0, 0]*3
	for i in range(1, radius+1):
		laplace+= coeff[i]*((a[0, 0, -i] + a[0, 0, i]) + (a[0, -i, 0] + a[0, i, 0]) + (a[-i, 0, 0] + a[i, 0, 0]))
	#return laplace
	return laplace

@stencil
def kernel1(a):
	return .25 * (a[0, 1, 0] + a[1, 0, 0] + a[0, -1, 0] + a[-1, 0, 0])


input_arr = np.ones(3**3).reshape((3, 3, 3))

#print input_arr

laplace_arr = laplace(input_arr, coeff)
print laplace_arr
#print laplace_arr[1, 1, 1]

#print kernel1(input_arr)
