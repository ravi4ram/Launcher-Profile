import os.path
import math
import numpy as np
import matplotlib.pyplot as plt

from bisect import bisect_left
from scipy.interpolate import make_interp_spline, BSpline

# parse flight sequence data
def readFlightData(fileName, sh=1):
	ldata = np.genfromtxt(fileName, delimiter='|', skip_header=sh, autostrip=True, dtype='unicode')
	# split the arrays
	event = ldata[:,0]
	time = ldata[:,1]
	altitude = ldata[:,2]
	velocity = ldata[:,3]
	# strip str
	event = [i.strip() for i in event]
	# convert str to float	
	time = [float(i) for i in time]
	altitude = [float(i) for i in altitude]
	velocity = [float(i) for i in velocity]	
	return event, time, altitude, velocity

# parse velocity data
def readVelocityData(fileName, sh=1):
	ldata = np.genfromtxt(fileName, delimiter='|', skip_header=sh, autostrip=True, dtype='unicode')
	# split the arrays
	time = ldata[:,0]
	velocity = ldata[:,1]
	# convert str to float
	time = [float(i) for i in time]
	velocity = [float(i) for i in velocity]	
	return time, velocity

# parse altitude data
def readAltitudeData(fileName, sh=1):
	ldata = np.genfromtxt(fileName, delimiter='|', skip_header=sh, autostrip=True, dtype='unicode')
	# split the arrays
	time = ldata[:,0]
	altitude = ldata[:,1]
	# convert str to float
	time = [float(i) for i in time]
	altitude = [float(i) for i in altitude]	
	return time, altitude

# parse atmospheric data
def readAtmosphericData(fileName='./atm_data.dat', sh=2):
	atw = np.genfromtxt(fileName, delimiter='|', skip_header=sh)
	return atw
	
# simple linear interpolation
def interpolate(x_list, y_list, x):  
	if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
		raise ValueError("x_list must be in strictly ascending order!")
	intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
	slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals] 

	if x <= x_list[0]:
		return y_list[0]
	elif x >= x_list[-1]:
		return y_list[-1]
	else:
		i = bisect_left(x_list, x) - 1
		return y_list[i] + slopes[i] * (x - x_list[i])

# returns interpolated atmos data for each alt (altitude value)	
def getAtmosphericData(ndarray, alt):
	# split the arrays
	altitude = ndarray[:,0]
	temperature = ndarray[:,1]
	density = ndarray[:,2]
	pressure = ndarray[:,3]
	molwt = ndarray[:,4]
	# interpolate values
	t = interpolate(altitude, temperature, alt)
	d = interpolate(altitude, density, alt)
	p = interpolate(altitude, pressure, alt)
	m = interpolate(altitude, molwt, alt)	
	return t, d, p, m

# find the nearest index in the list for the given value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# sort multiple list based on mainList indices	
def sortLists(mainList, sec1List, sec2List ):
	indices = [b[0] for b in sorted(enumerate(mainList),key=lambda i:i[1])]
	a=[]; b=[]; c=[]; d=[];
	for i in (indices):
		a.append(mainList[i])
		b.append(sec1List[i])
		c.append(sec2List[i])
	return a, b, c

# combines datas from multiple files. Units returned (sec, km, m/sec)
def combineVelocityData(file_evt, file_alt, file_vel):
	combined_time = []; combined_velocity = []; combined_altitude = [];
	# read velocity data (sec, m/sec)
	f_time, f_velocity = readVelocityData(file_vel, 1)
	# read altitude data (sec, km)
	h_time, h_altitude = readAltitudeData(file_alt, 1)
	# combine data
	for index, (ht, ha, ft, fv) in enumerate(zip(h_time, h_altitude, f_time, f_velocity)):
		if(ht == ft): 
			combined_time.append(ht)
			combined_altitude.append(ha)
			combined_velocity.append(fv)			

	# file check
	if os.path.exists(file_evt):
		# read event, time, altitude, velocity
		event, time, altitude, velocity = readFlightData(file_evt, 1)
		events_data = (event, time, altitude, velocity)		
		# relative velocity
		velocity = [x - velocity[0] for x in velocity]

	# combine data
	for index, (t, a, v) in enumerate(zip(time, altitude, velocity)):
		i, val = find_nearest(combined_time, t)
		combined_time.insert(i, t)
		combined_altitude.insert(i, a)
		combined_velocity.insert(i, v)

	# sort combined data
	combined_time, combined_altitude, combined_velocity = sortLists( combined_time, combined_altitude, combined_velocity )
	# return
	return combined_time, combined_altitude, combined_velocity, events_data

# get atmospheric data for each altitude. Units returned (kg/m3)
def combineDensityData(file_atm, altitude):	
	temperature = []; pressure = []; density = [];
	# file check
	if os.path.exists(file_atm):
		# read Altitude(km), Temp(K), Density(kg/m3), Pressure(Pa), Mol. Wt.(kg/kmol)
		ndr = readAtmosphericData(file_atm, 2)	# skip first 2 headers
		# get atmos data for each altitude 
		for index, elem in enumerate(altitude):
			t, d, p, m = getAtmosphericData(ndr, elem )
			density.append(d)
	return density

# calculate avg acceleration from velocity and time intervel
def getAcceleration(velocity, time):
	acceleration = []
	pre_time = 0; pre_velocity = 0;
	for i, (t, v) in enumerate( zip(time, velocity)): 
		dv = v - pre_velocity
		dt = t - pre_time			
		if dt != 0: 
			a = dv/dt
		else: a = 0

		pre_time = t; pre_velocity = v;		
		acceleration.append(a)	
	return acceleration

# calculate dynamic pressure (Pa-N/m2) and max-q with its corresponding altitude	
def getMaxQ(altitude, velocity, density, vehicle_diameter):
	# vehicle constants assumed
	A      = math.pi/4*(vehicle_diameter)**2; 	# frontal area (m^2)
	CD     = 0.5;           		# drag coefficient (assumed constant)
	# variables
	D = []; Q = [];
	# calculate dynamic pressure and drag force
	for a, v, d in zip(altitude, velocity, density):
		q  = 1/2 * d * v**2;	# dynamic pressure
		Q.append(q)
		d  = q * A * CD;		# drag	
		D.append(d)		
	# max-q 
	max_q = max(Q)
	return Q, max_q

# generate plot values
def generatePlotValues(filename_atm, filename_evt, filename_alt, filename_vel):
	# get combined data
	# events_data = (event, time, altitude, velocity)
	time, altitude, velocity, events_data = combineVelocityData(filename_evt, filename_alt, filename_vel)	
	# get avg acceleration
	acceleration = getAcceleration(velocity, time)	
	# get atmospheric data
	density = combineDensityData(filename_atm, altitude)
	# calculate dynamic pressure and max-q	
	q, max_q = getMaxQ(altitude, velocity, density, vehicle_diameter=3.0)			
	
	# velocity (m/s) to (Km/s) for plotting
	velocity = [v/1000 for v in velocity]
	# acceleration (m/s2) to (in G's) for plotting
	acceleration = [a/9.8 for a in acceleration]	
	# Pascal to KPa for plotting
	q = [p/1000 for p in q]	
		
	# returns time, altitude, velocity, acceleration, q, events_data
	return time, altitude, velocity, acceleration, q, events_data
	
# set font sizes
def setPlotFonts():
	# font size
	S_SIZE = 7
	plt.rc('font', size=S_SIZE)         # controls default text sizes
	plt.rc('axes', titlesize=S_SIZE)    # fontsize of the axes title
	plt.rc('axes', labelsize=S_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=S_SIZE)   # fontsize of the tick labels
	plt.rc('ytick', labelsize=S_SIZE)   # fontsize of the tick labels	 
	return

# time serie plot
def plot_time_series( ax, title_label, alt, alt_label, time, time_label, events):
	# fix font sizes
	setPlotFonts()
	S_SIZE = 6
	ax.set_title(title_label, fontsize=S_SIZE)
	ax.tick_params(axis='both', which='major', labelsize=S_SIZE) 
	ax.tick_params(axis='both', which='minor', labelsize=S_SIZE)	
	# color
	plot_bg_color = (1, 1, 0.5)
	# plot
	ax.plot(time, alt)
	for i, (event, px)  in enumerate(zip(events[0], events[1])):
		time_index = time.index(px)
		py = alt[time_index]
		ax.scatter( px, py, s=30, color='g')
		ax.text( px, py, event, fontsize='x-small')
	# set axis labels 	
	ax.xaxis.set_label_coords(0.5, -0.06)
	ax.set_xlabel(time_label, fontsize=S_SIZE)
	ax.yaxis.set_label_coords(-0.1,0.5)
	ax.set_ylabel(alt_label, fontsize=S_SIZE)
	# draw ticks like a graph sheet
	ax.set_facecolor(plot_bg_color)
	# turn on the minor TICKS
	ax.minorticks_on()
	# customize the major grid
	ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
	# customize the minor grid
	ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')		
	return	

def plot_alt_series( ax, title_label, x, x_label, y, y_label, events):
	# fix font sizes
	setPlotFonts()
	S_SIZE = 6
	ax.set_title(title_label, fontsize=S_SIZE)
	ax.tick_params(axis='both', which='major', labelsize=S_SIZE) 
	ax.tick_params(axis='both', which='minor', labelsize=S_SIZE)	
	# color
	plot_bg_color = (1, 1, 0.5)
	# plot
	ax.plot(x, y)
	for i, (event, px)  in enumerate(zip(events[0], events[2])):
		py = 0
		ax.scatter( px, py, s=30, color='g')
		ax.text( px, py+1, event, rotation=90, va='bottom', fontsize='x-small')
	# find max-q
	ymax = max(y)
	xpos = y.index(ymax)
	xmax = x[xpos]
	max_label = 'Max-Q = ' + str(round(ymax,2)) + ' kPa at altitude '+ str(round(xmax,2)) + ' km '
	# indicate max-q
	ax.annotate(max_label, xy=(xmax, ymax), xytext=(xmax, ymax-10),
				arrowprops=dict(facecolor='black', shrink=0.1, width=0.5) )
	# set axis labels
	ax.xaxis.set_label_coords(0.5, -0.06)
	ax.set_xlabel(x_label, fontsize=S_SIZE)
	ax.yaxis.set_label_coords(-0.1,0.5)
	ax.set_ylabel(y_label, fontsize=S_SIZE)
	# draw ticks like a graph sheet
	ax.set_facecolor(plot_bg_color)
	# turn on the minor TICKS
	ax.minorticks_on()
	# customize the major grid
	ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
	# customize the minor grid
	ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')	
	return	
	
# plot(time, altitude, velocity, acceleration, q, events_data)
def plot(plot_label, time, altitude, velocity, acceleration, dynamic_pressure, events_data):
	# Create a figure with 2 rows and 2 cols of subplots
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,7))
	fig.suptitle(plot_label, fontsize=10)
	fig.subplots_adjust(top=0.8)
	# plot altitude vs time
	plot_time_series(ax1, 'Alt vs Time', altitude,  "Alt (km)", time, "Time (sec)", events_data )
	# plot velocity vs time
	plot_time_series(ax2, 'Vel vs Time', velocity, "Vel (km/s)", time, "Time (sec)", events_data )
	# plot acceleration vs time
	plot_time_series(ax3, 'Acel vs Time', acceleration,  "Accleration (in G's)", time, "Time (sec)", events_data )
	# plot dynamic pressure vs altitude
	plot_alt_series(ax4, 'Alt vs Dyn Pressure', altitude,  "Alt (km)", dynamic_pressure, "Dynamic Pressure (kPa)", events_data )
	# show
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	s_name = plot_label + '.png'
	plt.savefig(s_name)
	plt.show()
	return	

# main function
if __name__ == "__main__":
	# atmospheric data
	filename_atm = 'atm_data.dat'

	# GSLV-MK3-D2 Data
	filename_evt = 'gslv-mk3-d2-gsat29-flight-events.dat'
	filename_alt = 'gslv-mk3-d2-gsat29-48-alt.dat'
	filename_vel = 'gslv-mk3-d2-gsat29-48-vel.dat'
	# generate values
	time, altitude, velocity, acceleration, q, events_data = generatePlotValues(filename_atm, filename_evt, filename_alt, filename_vel)	
	# plot
	plot('GSLV-MK3-D2-GSAT-29', time, altitude, velocity, acceleration, q, events_data)

	# PSLV-C26 Data
	filename_evt = 'pslv-c26-irnss-1c-flight-events.dat'
	filename_alt = 'pslv-c26-irnss-1c-11-alt.dat'
	filename_vel = 'pslv-c26-irnss-1c-11-vel.dat'
	# generate values
	time, altitude, velocity, acceleration, q, events_data = generatePlotValues(filename_atm, filename_evt, filename_alt, filename_vel)
	# plot
	plot('PSLV-C26-IRNSS-1C', time, altitude, velocity, acceleration, q, events_data)	

