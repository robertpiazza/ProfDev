# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:08:35 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Fourier Transforms Section 1.4.3

##complex review:
#1j*1j = (-1+0j)
#(3 + 4j) + (2 + 1j) = (5+5j)
#(3 + 4j) * (2 + 1j) = (2+11j)
#(3 + 4j) * (3 - 4j) = (25+0j) (complex conjugates multiplied)
#abs(3+4j) = 5.0 (Modulus) (recasts complex to float)

#Euler's formula: e^(i*t) = cos(t) + i*sin(t)
#Conversely: e^(-i*t) = cos(t) - i*sin(t)

##Fourier Transforms

#Fourier Tranforms data from the time domain to the frequency domain
#We can create low or high or mid-pass filters by truncating the freq data and
#applying the inverse fourier transform. Since fourier data is complex, we 
#graph it by the squared amplitude-the power spectrum transform

#Continuous vs Discrete Fourier Transform

#continuous transform (CFT) is an integral transform that is very useful in 
#mathematics, e.g. in differential equations and statistics
#We'll focus on the discrete (DFT) form as it is more relevant to data.

#DFT: X_k = (1/N)*[Summation from n=0 to N-1 of] x_n*e^(i*2*pi*k*n/N)
#Where x_n is the nth element in the signal array of N total elements
#k is the frequency, i is the imaginary number
#DFT is the same principle as the CFT (take a complex cosine/sine wave of 
#frequency k, multiply all the elements of the signal vector with the wave, 
#sum those values together, then shift k and repeat). However, instead of using 
#a continuum of frequencies, the transform is calculated only at select 
#discrete values of k

#The DFT as Matrix Multiplication

#DFT is actually equivalent to performing a matrix multiplication. 
#The matrix is called the DFT matrix and it is basically designed to contain 
#values sampled from the complex cosine/sine wave that is the heart of the 
#Fourier transform.

#The DFT measures the "similarity" between the signal and cosine and sine waves 
#of different frequencies
#fact that Fourier transforms only rotate the data has an important implication: 
#the total energy of the signal in the time domain is equal to the total energy 
#in the frequency domain

#Fast Fourier Transform

#1) the first row does not oscillate, it represents the DC-component; 
#2) the frequency goes up as you traverse down the rows, peaks in the middle, 
#then goes down again repeating some of the frequencies; 
#3) repeated frequency curves differ only in their imaginary parts (dashed 
#lines), that's because they are complex conjugates of one another.

#DFT can be calculated very efficiently on a computer. Normally, matrix 
#multiplication of the kind performed in DFT scales polynomially with the 
#number of points in the signal: O(N^2). However, because of the symmetries 
#within the DFT matrix, the multiplication can be brought down to O(N log N)
#Any algorithm which accomplishes this is a fast-fourier-transform (FFT)
#Tukey-Cooley algorithm is most well known

#The Inverse Fourier Transform

#we can obtain the DFT matrix inverse by taking the complex conjugate of each 
#element and transposing the matrix (conjugate transform)

#the Fourier transform and the inverse Fourier transform are exact 
#transformations without distortion. (barring any filtering or alterations 
#that you may perform on either and allowing for numerical precision of the 
#computer)

#Usually, FTs act on things in the time domain but could be used on any function
#(not double valued at any point)
#

#Sum of Two Sine Waves

#Using Python and NumPy, we are creating a signal x that is the sum of two sine 
#waves of different frequencies

import numpy as np
import pylab as pl
ti = 0
tf = 10.0
N = 1001
t = np.linspace(ti,tf,N) #N-1 datapoints stretching from ti to tf
x = np.sin(2*np.pi*t)+np.sin(2*np.pi*np.sqrt(2)*t) #
pl.figure(1)
pl.plot(t,x, color='#5D5166', lw=1.8)
pl.show()

#The Frequency Domain

#Take its Fourier transform of the signal x and plot it

xhat = np.fft.fft(x) # Fast Fourier Transform
xhat_sh = np.fft.fftshift(xhat) #shifting for plotting
dt = (tf-ti)/(N-1) #differential time unit
f = np.fft.fftfreq(N,dt) 
f_sh = np.fft.fftshift(f)
pl.figure(2)
pl.plot(f_sh,abs(xhat_sh), color='#5D5166', lw=1.8)
pl.xlim((-3,3))
pl.xlabel('frequency')
pl.ylabel('amplitude')
pl.show()

#plot tells us which frequencies are present in our signal and with what 
#strength-we've transformed from time to frequency domain

#two of the peaks occur at negative frequencies and they are symmetric with 
#those at positive frequencies. The power spectrum at negative frequencies 
#offer redundant information as the positive frequencies

#One may notice the peak at f = 1 is very sharp, while the peak at f ~ 1.4 is 
#broader. This blurring of the power spectrum is due to the fact that our 
#frequencies are discretized, despite appearing continuous in the graphs. One 
#of the frequencies we chose for our signal was f=sqrt(2), which is an 
#irrational number. The DFT does not have that exact frequency in its 
#discretization, so it uses several frequencies combined together to reproduce
#it. We could make the peak sharper by using smaller time steps in the signal

#Back to the Time Domain

#If we have the Fourier transform of a signal, we can reconstruct the signal 
#from it

x2 = np.fft.ifft(xhat)
pl.figure(3)
pl.plot(t,x,'-k',t,x2,'.', lw=2, color='#5D5166')
pl.xlabel('time')
pl.show()

#In reality, round-off error limits the number of times we can switch from time 
#to frequency domain and back without unacceptable errors

#remove f = 1 signal:

idx = ((f>0.97) & (f<1.03)) | ((f<-0.97) & (f>-1.03))
xhat3 = xhat
xhat3[idx] = 0
xhat3_sh = np.fft.fftshift(xhat3)
pl.figure(4)
pl.plot(f_sh,abs(xhat3_sh), color='#5D5166', lw=2)
pl.xlim((-3,3))
pl.xlabel('frequency')
pl.ylabel('amplitude')
pl.show()

#now take the inverse transform:

x3 = np.fft.ifft(xhat3)
pl.figure(5)
pl.plot(t,x3, color='#5D5166', lw=2)
pl.xlabel('time')
pl.show()

#We just applied a very rudimentary notch, or band-stop, filter to the signal.

#The DFT can transform functions that have discontinuities and/or points that 
#are non-differentiable. In general, functions with sharp points and strong 
#bends require power in high-frequency components to represent them. On the 
#other hand, functions that wend slowly require power in low-frequency 
#components

# Middle C-chord with background noise
fl = open('1-4-3 chord1.csv','rb')
data=np.loadtxt(fl,delimiter=',')
fl.close()
r = 11025.0
t = data[:,0]
x = data[:,1]
pl.figure(1);
pl.plot(t,x, color='#5D5166')
pl.xlim((t[0],t[-1]))
pl.xlabel('time')
pl.ylabel('amplitude')
pl.show()

#apply DFT

xhat = np.fft.fft(x)
f = np.fft.fftfreq(len(x),1/r)
xhat_sh = np.fft.fftshift(xhat)
f_sh = np.fft.fftshift(f)
pl.plot(f_sh/1000.0, abs(xhat_sh), color='#5D5166')
pl.setp(pl.gca(),yscale='log',xlim=(0,3),ylim=(0.001,1000))
pl.xlabel('frequency (kHz)')
pl.ylabel('amplitude')
pl.show()

#zoom in on one of the frequencies of interest


pl.plot(f_sh/1000.0, abs(xhat_sh), color='#5D5166')
pl.setp(pl.gca(),yscale='linear',xlim=(0.15,1),ylim=(0.1,180))
pl.xlabel('frequency (kHz)')
pl.ylabel('amplitude')

#Middle C has a freq of ~262 Hz, this piano needs some tuning, as its middle-C 
#is a little low at 252 Hz. Some of the other peaks are harmonics of the
#middle-C note, while others are from the E4 and G4 notes in the chord

#Filtering the Audio

#A very simple low-pass filter is the Butterworth:
#B(f) = sqrt{1/[1+abs(f/f_c)^(2n)]} - fc is the cutoff frequency and n is 
#steepness of cutoff

def butterworth(f,fc,n):
    return np.sqrt(1/(1+abs(f/fc)**(2*n)))

pl.figure(3);
pl.plot(f_sh/1000.0,butterworth(f_sh,1000,1), color='#5D5166')
pl.plot(f_sh/1000.0,butterworth(f_sh,2000,10), color='#FF971C')
pl.plot(f_sh/1000.0,butterworth(f_sh,3000,100), color='#447BB2')
pl.setp(pl.gca(),ylim=(0,1.6),xlim=(-5,5))
pl.xlabel('frequency (kHz)')
pl.ylabel('gain')
pl.legend(('fc=1000, n=1','fc=2000, n=10','fc=3000, n=100'), loc=2, frameon=False, labelspacing=0.1),
pl.show()

#Let's use the Butterworth filter on our chord clip. We set the cutoff at 
#1.2 kHz and n=20

b=butterworth(f,1200,20)
xhat_filt=b*xhat
xhat_filt_sh = np.fft.fftshift(xhat_filt)
pl.figure(4);
pl.plot(f_sh/1000.0,abs(xhat_filt_sh), color='#5D5166')
pl.setp(pl.gca(),yscale='log',xlim=(0,3),ylim=(0.001,1000))
pl.xlabel('frequency (kHz)')
pl.ylabel('amplitude')
pl.show()

#if we wanted, we could design a much more precise filter that would band-pass 
#those frequencies. Let's take the inverse transform of xhat_filt and plot a 
#portion of the resulting signal together with the original

x_filt = np.fft.ifft(xhat_filt)
pl.figure(5);
pl.plot(t,x, color='#5D5166')
pl.plot(t,x_filt, color='#FF971C')
pl.xlim((0.05,0.1))
pl.xlabel('time')
pl.ylabel('amplitude')
pl.legend(('before','after'),frameon=False,loc='lower left',labelspacing=0.1,borderpad=0)
pl.show()


pl.plot(t,x, color='#5D5166')
pl.plot(t,x_filt, color='#FF971C')
pl.xlim((0.0625,0.0675))
pl.xlabel('time')
pl.ylabel('amplitude')
pl.legend(('before','after'),frameon=False,loc='lower left',labelspacing=0.1,borderpad=0)
pl.show()

xhat = np.array([-1,-7+12.12j,5+8.66j,11,5-8.66j,-7-12.12j])/np.sqrt(6)
x_time = np.fft.ifft(xhat)
t = np.linspace(0,1,6)
pl.plot(t,x_time, color='#FF971C')
pl.xlabel('time')
pl.ylabel('amplitude')
pl.show()
print(np.sum(x_time))

#Calculate the power spectrum (amplitude squared) of the Fourier transform
print('power spectrum', np.abs(xhat)**2)

#Aliasing occurs when undersampling higher frequency data
#The Nyquist frequency is half the sampling rate, so the sampling rate needs to 
#be at least twice the largest frequency you are interested in.

#delta_f = sampling_rate/Number_of_samples and sampling_rate = 1/delta_t

#Calculating Frequencies in Python

import numpy as np
import pylab as pl
N = 50
dt = 0.01
f = np.fft.fftfreq(N,dt)
#fftfreq takes the number of points in the signal and the time between samples 
#as arguments and outputs the frequencies associated with each row of the DFT 
#matrix
pl.figure(1)
pl.plot(f,'o-', color='#5D5166')
pl.xlabel('row number')
pl.ylabel('frequency')
pl.show()

#due to aliasing, the frequencies are ordered strangely
#Python has a function to fix the sorting for you.

f2 = np.fft.fftshift(f)
pl.figure(2)
pl.plot(f2,'o-', color='#5D5166')
pl.xlabel('row number')
pl.ylabel('frequency')
pl.show()

#fftshift doesn't simply sort the frequencies, it does a "circular shift" of the
#input. That means you can also use it on the Fourier transformed data as well. 

#Inverse DFT

#Convolutions can be used to smooth data or calculate a local average
import numpy as np
import pylab as pl
np.random.seed(4781193) # set random seed to ensure same noise every time
N = 501
t = np.linspace(0,1,N)
x = 1.2*np.sin(2*np.pi*(6*t+3)*t)*pl.normpdf(t,0.5,0.2)+np.random.randn(N)
x = x - np.mean(x) # so that DC component does not dominate DFT

#The signal is a chirp--a signal that increases or decreases frequency in 
#time--modulated by a Gaussian window and with noise added. Let's create three 
#Gaussian kernels with different widths

tau=t-np.mean(t)                # center time window
g1 = pl.normpdf(tau, 0, 0.002)
g2 = pl.normpdf(tau, 0, 0.007)
g3 = pl.normpdf(tau, 0, 0.03)
#normalize Gaussian window, otherwise convolution amplifies signal, i.e. we 
#want a weighted average
g1 = g1/sum(g1)                 
g2 = g2/sum(g2)                 
g3 = g3/sum(g3)                 



pl.figure(1);
pl.plot(tau,g1, linewidth=2.5, color='#5D5166') 
pl.plot(tau,g2, linewidth=2.5, color='#FF971C') 
pl.plot(tau,g3, linewidth=2.5, color='#447BB2') 
pl.gca().set(xlim=(-0.1,0.1),ylim=(0,0.5),xlabel='time shift',ylabel='Gaussian amplitude')
pl.legend(('s=0.002','s=0.007','s=0.03'),fontsize=12)
pl.show()

#Let's use the Gaussians to perform three separate convolutions of the signal x

xg1 = np.convolve(x,g1,'same') 
# 'same' ensures the output has same number of elements as largest input
xg2 = np.convolve(x,g2,'same')
xg3 = np.convolve(x,g3,'same')

pl.figure(2);
pl.plot(t,x,color=[0.7,0.7,0.7], lw=1.8) # original signal
pl.plot(t,xg1, linewidth=2.5, color='#5D5166') 
pl.plot(t,xg2, linewidth=2.5, color='#FF971C') 
pl.plot(t,xg3, linewidth=2.5, color='#447BB2') 
pl.gca().set(ylim=(-4.5,4.5),yticks=[-4,-2,0,2,4],xlabel='time',ylabel='signal amplitude')
pl.show()

#The largest Gaussian smooths out all the fluctuations; not just the 
#noise, but also some of the chirp signal
#By convolving the signal with the Gaussian kernel, we have actually filtered 
#out some of the frequencies

f = np.fft.fftfreq(N,1./(N-1))
y = np.fft.fft(x)
yg1 = np.fft.fft(xg1)
yg2 = np.fft.fft(xg2)
yg3 = np.fft.fft(xg3)
fs = np.fft.fftshift    # alias the function as a shortcut

pl.figure(3);
pl.plot(fs(f),fs(abs(y)),color=[0.7,0.7,0.7], lw=1.8)
pl.plot(fs(f), fs(abs(yg1)), linewidth=1.8, color='#5D5166') 
pl.plot(fs(f), fs(abs(yg2)), linewidth=1.8, color='#FF971C') 
pl.plot(fs(f), fs(abs(yg3)), linewidth=1.8, color='#447BB2')
pl.gca().set(yscale='log',ylim=(0.05,300), xlim=(0,200), xlabel='frequency', ylabel='amplitude')
pl.show()

#, a convolution in one domain is equivalent to a point-by-point multiplication 
#in the other domain

h2 = np.fft.fft(g2)
yh2 = y*h2

pl.figure(4);
pl.plot(fs(f),fs(abs(yg2)), linewidth=1.8, color='#5D5166') 
pl.plot(fs(f),fs(abs(yh2)),'o', color='#5D5166') 
pl.gca().set(ylim=(0.1,300), xlim=(0,50), xlabel='frequency',ylabel='amplitude', yscale='log')
pl.ylabel('amplitude')
pl.xlabel('frequency')
pl.legend(('fft(xg2)','fft(g2)*fft(x)'),fontsize=12, loc='lower left')
pl.show()

#The real value in the convolution theorem is in how they help to interpret 
#filtering. the wide Gaussian becomes narrow in frequency space, meaning that 
#it attenuates a broad band of frequencies.

h3 = np.fft.fft(g3)
pl.figure(5);
pl.plot(fs(f),fs(abs(y)), color=[0.7,0.7,0.7])
pl.plot(fs(f),fs(abs(yg2)), fs(f),200*fs(abs(h2)),'--', linewidth=1.8, color='#FF971C')
pl.plot(fs(f),fs(abs(yg3)), fs(f),200*fs(abs(h3)),'--', linewidth=1.8, color='#447BB2')
pl.gca().set(ylim=(0.1,200), xlim=(0,80), xlabel='frequency',ylabel='amplitude')
pl.ylabel('amplitude', labelpad=0)
pl.xlabel('frequency', labelpad=0)
pl.show()

#Fourier transforms are a powerful tool both for mathematics and data. We have 
#seen some of its uses for the latter, but have only scratched the surface of 
#applications. In the time-frequency analysis mission, we extend the concept of 
#Fourier transforms to allow analyses that are time-varying


