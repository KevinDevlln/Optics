# Zernike-wavefront-sensor
Simulation of a Zernike phase contrast wavefront sensor (ZWFS) with a phase contrast filter (PCF).
A basic script that generates 2D images of Zernike polynomials modes as well as calculating there Point Spread Functions (PSF) through the use of a Fourier transform.
A phase contrast filter is applied to the central region of the PSF to simulate the effect of a Zernike wavefront sensor.
Taking the inverse Fourier transfomr of the multiplication of the PCF with the PSF results in a mapping of the phase errors to an intensity distributon.
Phase reconstruction can be performed to calculate and visualise the initial phase error of the system for each Zernike mode.
