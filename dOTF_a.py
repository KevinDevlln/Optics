

import numpy as np
import matplotlib.pyplot as plt
#import fft_s as ft
import gaussian2d as G

#np.set_printoptions(threshold=np.nan)
#generate NxN Coordinate Grid

N = 1024                                #number of points
L = 1                                   #length of grid side (m)
delta = L/N                             #sample spacing
D = 0.1                                 #aperture diameter
r = D/2                                 
f = 0.5                                 #focal length
f_n = f/D  
wave = 0.63e-6
k = 2*np.pi/wave

#coordinates
x = np.arange(-N/2, N/2)*delta
(xx, yy) = np.meshgrid(x, x)

#image freq coordinates
df = wave*f/(N*delta)                  #frequency spacing
fx = np.arange(-N/2, N/2)*df
fX, fY = np.meshgrid(fx, fx)

rho = np.sqrt(xx**2 + yy**2)
phi = np.arctan2(yy,xx)

#normalise co-ordinates
R_norm = rho/r

#MTFaxis = (N*delta)/wave/f_n

#generate aperture
outer_disc = np.zeros((N,N))
inner_disc = np.zeros((N,N))
obs=0.2

mask = (R_norm<=1)
mask2 = (R_norm<=(1*obs))
outer_disc[mask]=1
inner_disc[mask2]=1

aperture = outer_disc # - inner_disc

#------------------------------------------------------------------------------

#aberration
rms = 0.25*wave
w = np.sqrt(8)*(3*(R_norm**3)-2*R_norm)*np.sin(phi) *rms #Vertical Coma  
#w = 2*R_norm*np.sin(phi) *0.25  #tilt
w[R_norm>1]=0

#pupil plane A
A_noab = aperture * np.exp(1j * k *0)
A = aperture * np.exp(1j * k *w)

#image plane B
B = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A)))*delta**2

PSF = np.abs(B)**2
PSF = PSF/np.max(PSF) 

#exit plane C
OTF = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(PSF)))
MTF = np.abs(OTF)

#-----------------------------------------------------------------------------
#create perturbation

#Gaussian perturbation function
gauss = G.gaussian2d(N, 14, amplitude=wave*0.25, cent=(220*2, 220*2))

#finger at edge of pupil
finger = np.zeros((N, N))
y_edge = r/delta + N/2 +1
x_edge = N/2 + 1
finger[int(x_edge-4):int(x_edge), int(y_edge-8):int(y_edge)] = 1  #width*height


perturbation = aperture - finger

#------------------------------------------------------------------------------
#create a second pupil with perturbation

A2 = A * perturbation
B2 = B = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A2)))*delta**2

PSF2 = np.abs(B2)**2
PSF2 = PSF2/np.max(PSF2)

OTF2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(PSF2)))
MTF2 = np.abs(OTF2)

#differential optical trasfer function
dOTF = OTF - OTF2


#data visualisation
##############################################################################

fig1 = plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.angle(A), extent = [x[0], x[-1], x[0], x[-1]])
plt.title('input pupil')
plt.gca().set_xlim(-0.1, 0.1)
plt.gca().set_ylim(-0.1, 0.1)
plt.xlabel('x, (m)')
plt.ylabel('y, (m)')
plt.colorbar()

plt.subplot(122)
plt.imshow(np.angle(A2), extent = [x[0], x[-1], x[0], x[-1]])
plt.title('perturbed input pupil')
plt.gca().set_xlim(-0.1, 0.1)
plt.gca().set_ylim(-0.1, 0.1)
plt.xlabel('x, (m)')
plt.ylabel('y, (m)')
plt.colorbar()

plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow((PSF), extent=[fx[0], fx[-1], fx[0], fx[-1]])
plt.title('PSF')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.gca().set_xlim(-50e-6, 50e-6)
plt.gca().set_ylim(-50e-6, 50e-6)
plt.colorbar()

plt.subplot(122)
plt.imshow(PSF2, extent = [fx[0], fx[-1], fx[0], fx[-1]])
plt.title('perturbed PSF')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.gca().set_xlim(-50e-6, 50e-6)
plt.gca().set_ylim(-50e-6, 50e-6)
plt.colorbar()

plt.tight_layout()
plt.show()


fig3 = plt.figure(figsize=(6,6))
plt.imshow(MTF)
plt.title('Modulation Transfer Function')
plt.colorbar()



fig4 = plt.figure(figsize=(6,6))
plt.imshow(np.abs(dOTF))
plt.title('dOTF')
plt.colorbar()

fig5 = plt.figure(figsize=(6,6))
plt.imshow(np.angle(dOTF))
plt.title('dOTF phase')
plt.colorbar()


