from __future__ import print_function
import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
import scipy.constants as ct
import scipy.integrate as integrate
import scipy.special as spec
import math as mh
import time
from astropy.cosmology import WMAP9
from astropy.cosmology import LambdaCDM
from astropy.cosmology import z_at_value
from astropy.units import u
import scipy.stats
from scipy.optimize import curve_fit
from scipy.misc import factorial
import numpy.random as rm
import scipy.stats as stats
from multiprocessing import Pool,Process, freeze_support
import multiprocessing
import argparse

G=ct.G
c=ct.c
Msun = 1.989e30
pi = ct.pi

noisecurvepts = 900

Hubble_Parameter = WMAP9.H(0.0).value #km/sec/Mpc
h = Hubble_Parameter/100.0
Omega_M = WMAP9.Om(0.0)
Omega_vac = 1.0-Omega_M


#this is to cut out scale factors that occur before the first real mergers?????
def scaleadjustfunc(scale_factor):
	change = (scale_factor<=1.0/138.0)
	keep = (scale_factor>1.0/138.0)
	return scale_factor*keep+(1.0/138.0)*change

def Least_squared_binned_fit(counts):
	atest = np.linspace(0.001, 20.0, 10000)
	chisquared = np.zeros(len(atest))
	for i in np.arange(len(counts[0])):
		if counts[0][i] != 0:
			chisquared = chisquared+((counts[0][i]+np.sum(counts[0])*(np.exp(-counts[1][i+1]/atest)-np.exp(-counts[1][i]/atest)))/(counts[0][i]**0.5))**2.0
	return atest[np.argmin(chisquared)], chisquared[np.argmin(chisquared)]

def func(x, a):
    return 1.0/a*np.exp(-x/a)

def chisquaredpdf(x,nu):
	return 1.0/(2.0*spec.gamma(nu/2.0))*(x/2.0)**(nu/2.0-1.0)*np.exp(-x/2.0)

def trimfunc(comoving_t,comoving_r, time_cut):
	keep1 = ((comoving_r-comoving_t)<= time_cut*365.25*24.0*3600.0*c)
	comoving_t = comoving_t*keep1
	comoving_r = comoving_r*keep1
	keep2 = ((comoving_r-comoving_t)> 0.0)
	comoving_t = comoving_t*keep2
	comoving_r = comoving_r*keep2
	remove = np.where(comoving_t==0.0)
	comoving_t = np.delete(comoving_t, remove)
	comoving_r = np.delete(comoving_r, remove)
	return comoving_t, comoving_r

def parallelfunc(kernel, kernel_generator, generate_count, i):
	distrib = 100
	print("num = %i --- %s seconds ---" % (i,time.time() - start_time))	
	rm.seed(int(time.time())+i)
	dataout = []
	for j in range(distrib):
		#random_data = np.fabs(kernel_generator.resample(size = int(generate_count/distrib)))
		random_data = np.fabs(kernel.resample(size = int(generate_count/distrib)))
	
		random_distance = boundary.value*1e6*3.086e16*((rm.uniform(low = -1.0, high = 1.0, size = int(generate_count/distrib)))**2.0+(rm.uniform(low = -1.0, high = 1.0, size = int(generate_count/distrib)))**2.0+(rm.uniform(low = -1.0, high = 1.0, size = int(generate_count/distrib)))**2.0)**0.5
		
		diff = np.subtract(random_distance,random_data[0])/(c*3600.00*24.0*365.25)
		#diff = np.subtract(random_distance,random_data[0])/(c*3600.00*24.0*365.25)

		#print("data gen num = %i --- %s seconds ---" % (i,time.time() - start_time))

		keep = np.where((diff >= -5.0) & (diff <= 1e4))
		random_data = [random_data[i][keep[0]] for i in range(3)]
		random_distance = random_distance[keep[0]]
		diff = diff[keep[0]]
		
		print("removal num = %i --- %s seconds ---" % (i,time.time() - start_time))
		
		dataout.append([random_data[0], random_distance, diff, random_data[1], random_data[2]])	
		
	dataout = np.concatenate(dataout, axis=1)	
	print("num = %i --- %s seconds ---" % (i,time.time() - start_time))	
	return dataout

if __name__ == '__main__':

	import sys
	numprocessors = 4
	i = 1
	
	while i < len(sys.argv):
		if sys.argv[i-1]=='-n':
			numprocessors = int(sys.argv[i])
		i += 1
	
	f = h5py.File('blackhole_mergers-ILL3.hdf5', 'r')
	mass = f['details']['mass']
	scale_factor = f['details']['time'][:,0]

	scale_factor = scaleadjustfunc(scale_factor)

	masses = np.array([f['details']['mass'][:,0]*1e10/h,f['details']['mass'][:,1]*1e10/h,f['details']['mass'][:,2]*1e10/h])

	z = (1.0-scale_factor)/scale_factor

	remove = np.where(z>10.0)

	z = np.delete(z, remove)
	masses = np.delete(masses, remove, axis = 1)

	remove1 = np.where(masses<1e3)

	masses = np.delete(masses, remove1, axis = 1)
	z = np.delete(z, remove1)

	cosmo = LambdaCDM(WMAP9.H(0.0).value, WMAP9.Om(0.0), 1.0-WMAP9.Om(0.0))

	comoving_distance = cosmo.comoving_distance(z).value*1e6*3.086e16

	f.close()
	
	#perform kernel density estimate
	#values = np.vstack([masses[0],masses[1],comoving_distance])
	kernel_generate = stats.gaussian_kde(comoving_distance)
	kernel = stats.gaussian_kde(np.vstack([comoving_distance,masses[0], masses[1]]))
	
	boundary = cosmo.comoving_distance(10.0)

	comoving_volume_cube = 106.5**3.0 #Mpc
	comoving_volume_boundary = 4.0/3.0*pi*boundary.value**3.0

	generate_counts = int(np.sum(len(masses[0]))*np.floor(comoving_volume_boundary/comoving_volume_cube))
	#generate_counts = int(1e8+6.6e2)


	add = int(generate_counts-int(generate_counts/numprocessors)*numprocessors)
	
	
	

	start_time = time.time()

	#Below is for testing parallelfunc
	#results = parallelfunc(kernel, kernel_generate, generate_counts, 1)
	
	args = [(kernel, kernel_generate, int(generate_counts/numprocessors), i) for i in range(numprocessors-1)]
	args.append((kernel,kernel_generate, int(generate_counts/numprocessors)+add,numprocessors-1))
	
	freeze_support()
	
	pool = Pool(numprocessors)
	res = pool.starmap(parallelfunc, args)

	pool.terminate()
	pool.join()
	del pool	

	#pdb.set_trace()
	out = np.concatenate(res, axis=1)
	
	
	print("--- %s seconds ---" % (time.time() - start_time))
	start_time = time.time()

	time_of_day = time.strftime("%H:%M:%S")
	date = time.strftime("%d-%m-%Y")
	

	np.savetxt('BH_collisions_output-1.txt', np.transpose(out), header = 'comoving_t_coord	comoving_r_coord	diff	m1	m2', fmt='%1.12e') 
	
	#np.savetxt('BH_collisions_output-1.txt', np.transpose(out), header = 'comoving_t_coord	comoving_r_coord	diff', fmt='%1.12e') 
	
		
	print("--- %s seconds ---" % (time.time() - start_time))


