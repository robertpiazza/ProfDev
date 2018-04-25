# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:29:39 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Time Frequency Analysis Section 1.4.4

#Learn about the Short-term Fourier Transform (STFT) and its use for 
#interpreting data, as well as the Wavelet transform (WT)

#Real world signals, however, change over time: a low frequency might be present 
#for some time then die away and a high frequency may blip for a moment.

#We want to know not only what frequencies are present in a signal, but also 
#when those frequencies are present.  the Heisenberg-Gabor limit, which places 
#a limit on how precisely we can simultaneously know the frequency content of a 
#signal and the precise moment at which the frequencies occur. 
#STFTs are "blind" to the limit and do not adjust for it at all. Wavelets 
#"adaptively" deal with it and, though they cannot escape the fundamental limit, 
#they work around it by being precise in time/imprecise in frequency at high 
#frequencies and the opposite at low frequencies.

##Stationary versus non-stationary signals

#We use NumPy to take the Fourier transforms of x1 and x2 and call them y1 and 
#y2 respectively. Also, get the frequencies for the Fourier transforms and 
#store it in the variable f. 

import numpy as np
ti = 0
tf = 2
N=2001
t = np.linspace(ti,tf,N)
x1 = 0.5*(np.sin(4*np.pi*t) + np.sin(16*np.pi*t))
x2 = 0*x1
dt = .001
print(dt)
x2[0:1000] = np.sin(4*np.pi*t[0:1000])
x2[1000:2001] = np.sin(16*np.pi*t[1000:2001])
y1 = np.fft.fft(x1)
y2 = np.fft.fft(x2)
f = np.fft.fftfreq(N,dt) 

#Or:
y1 = np.fft.fft(x1)
y2 = np.fft.fft(x2)
f = np.fft.fftfreq(2001, 0.001)

##Short-term Fourier Transform

#One way to get temporal information about a signal is to perform the Fourier 
#transform on pieces of a signal at a time. That is, use a window to "cut out" 
#a piece of the signal, take the Fourier transform of the signal inside the 
#window, then shift the window a little and repeat. This is the STFT

##Wavelet Transforms

#imagine that we could adapt our window size in the STFT to better suit our needs
#at low frequencies, we need a bigger window to have a good estimate of the 
#frequency at the cost of losing temporal precision. At higher frequencies, we 
#can use a smaller window to estimate the frequency and gain temporal precision. 
#This is exactly what wavelets do

#In the STFT, you are stuck to describing signals using cosine and sine 
#functions. Not so with wavelets. You can pick and even design your own function 
#that fits your needs. A good general rule of thumb is to pick a wavelet that 
#looks like the feature you are hunting for in the data (e.g., if your data has 
#big discontinuous jumps, then pick a wavelet that does the same

#To design your own wavelet, the only real condition is that it should sum or 
#integrate to zero, which is knowns as the admissibility condition

#The admissibility condition of a wavelet implies that its integral be zero. 
#Therefore, the DC component of the Fourier transform of any wavelet satisfying 
#the admissibility condition is 0. 

#Let's take a closer look at one particular wavelet now, called the Morlet 
#(or Gabor wavelet). 

import scipy.signal as sig
x=sig.morlet(1000,w=7.0,s=1,complete=True) #1000 points in waveform, 7 
#oscillations before tapering off, scaling factor of 1 (low #'s magnify the
#wavelet), complete indicates an offset to improve admissibility (important only 
#when w<5)

#middle ten values of array x
print(x[495:505])
#accounting for the complex plane, a Morlet wavelet is a spiral that is 
#modulated by a Gaussian window

#in 3D:
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
t = range(0,1000)
fig = pl.figure(1, figsize=(4,3))
pl.clf();
ax = fig.gca(projection='3d')
pl.plot(t, x.real, x.imag, color='#5D5166') #Purple spiral
pl.plot(t, x.real, -0.8+x.real*0, color='#FF971C') #Orange real component
pl.plot(t, 0.8+x.real*0, x.imag, color='#447BB2') #Blue imaginary component
ax.set(xticks=[], yticks=[], zticks=[], xlabel='time', ylabel='real part', zlabel='imaginary part')
pl.show()

#The purple curve shows the spiral as it moves through the real and imaginary 
#axes in time. The blue curve shows its projection of the spiral on the 
#imaginary axis and the gold curve shows the projection on the real axis. The 
#tapering spiral of the Morlet wavelet is identical to the basis functions used 
#in STFTs

#Performing a CWT with a Morlet wavelet is not identical to performing a STFT 
#with a Gaussian window. 

##Wavelet scaling versus STFT frequency

#CWT is generally more useful for scientific purposes, as evenly sampled 
#scaleograms are easier to interpret
#The DWT has the advantage of being faster to calculate and easier to take the 
#inverse of (the CWT has no unique inverse). 
#the DWT is better suited for engineering and computer science applications 
#like compression and denoising

#CWT

#Python's wavelet packages are nascent and have sparse documentation or errors
#SciPy's signal package's continuous wavelet transform, scipy.signal.cwt, does 
#not compute the transform correctly for complex-valued wavelets
#the continuous wavelet transform is straightforward and we can calculate it 
#ourselves

t = range(0,2001)
r = 1000.0                          # sampling rate
samples = 2001                        # number of samples in wavelet
w = 5                                 # controls number of oscillations in wavelet
ds = samples/(2*w*r)                  # scaling factor spacing
s = ds*np.arange(1,13)                # scaling factor
f = 2*w*ds*np.arange(1,13)*r/samples  # frequencies                
cwt_x2 = np.ones([len(s), len(t)])*1j # for storing CWT values
# ITERATE OVER SCALING FACTORS AND CONVOLVE TO CALCULATE CWT
for ind, b in enumerate(s):
    wavelet_data = sig.morlet(samples, w, b)
    cwt_x2[ind, :] = sig.convolve(x2, wavelet_data, mode='same')*np.sqrt(b)

#The code above iterates over the different sizes of a Morlet wavelet and 
#calculates the similarity between our non-stationary signal, x2, and the 
#wavelet over time using a convolution
#we are converting the parameters that feed into the morlet function into a 
#frequency
    
pl.figure(1, figsize=(4,3));
pl.clf()
pl.pcolor(t, f - 0.5, abs(cwt_x2))
pl.gca().set(ylim=(0.5, 11.5), xlabel='time', ylabel='frequency')
pl.show()
    
#It is not true one can only use wavelets that are defined in terms of frequency
#Wavelets come in all shapes and sizes. Some of them are defined in terms of an 
#algorithm.

#The window of a STFT does not have to sum to zero, only wavelets

#Wavelets increase their "frequency" by changing their scale. As the size of 
#the wavelet decreases, its frequency increases

#True statements about CWTs:
#It increases frequency resolution at low frequencies.
#It increases temporal resolution at high frequencies.
#You can make custom wavelets suitable for your application.

#TL:DR
#Remember that performing a CWT is the same as finding the similarity between the signal and a wavelet at different time shifts and scales. The DWT does the same thing, but instead of shifting and scaling the wavelet evenly, it does so on a log scale.
#The CWT has the virtue of being straightforward to calculate and interpret. However, it suffers from being costly to compute, as it encodes redundant information; in mathematical parlance, the CWT does not describe the signal in an orthogonal basis. Therefore, the inverse transform is not easy to calculate as there is no unique way of performing it.
#The DWT usually has orthogonal bases, but sometimes has bases that are only approximately orthogonal. Either way, the business of reconstruction (inverse transform) is made much easier when this is true. Moreover, by sampling scales and shifts more conservatively than the CWT, the DWT is much more efficient to calculate.

#Dyadic sampling
#The most common sampling is one based on powers of two - dyadic sampling.

#Dyadic sampling as filter banks
#each scale of a wavelet transform acts as a bandpass filter
#The width of the frequency band decreases as the scale of the wavelet 
#increases: if the scale doubles, the bandwidth halves

#Performing a wavelet transform is identical to performing a convolution with 
#the wavelet and the signal. From the convolution theorem, we know that a 
#convolution in the time domain is equivalent to performing a point-by-point 
#multiplication in the frequency domain
#That means when performing a wavelet transform, the signal's Fourier transform 
#gets multiplied with the wavelet's Fourier transform. Therefore, the wavelet 
#filters out any frequencies where its Fourier transform amplitude is small and 
#it passes frequencies where its amplitude is high.

#There is an inverse relationship between the scale and width of the transform; 
#as the scale of the wavelet increases, its Fourier transform decreases in 
#width. Hence, larger wavelets pass a narrower band of frequencies. To cover 
#the entire spectrum all the way down to zero, we would need an infinite number 
#of wavelets.
#To work around this, the DWT cuts the scaling off at a certain point, then 
#plugs in a different wavelet (known as the father wavelet or scaling function) 
#to fill the gap left by the "bank" of scaled mother wavelets

#TL:DR
#The original signal is split between a high-pass filter (h[n]) and a low-pass filter (g[n]) (other sources may reverse the h and g filters) and is also downsampled by a factor of two. The high-pass filter represents the mother wavelet and the low-pass filter represents the father wavelet. The signal passing through the low-pass filter is split between two bands and downsampled again using the same pair of filters. The process is repeated until the desired level is reached

##Downsampling
#The DWT never actually scales the wavelet directly. The wavelet stays the same 
#and, instead, the signal is downsampled. By choosing every other point in the 
#signal, as is done in dyadic sampling, the wavelet appears twice as large 
#relative to it; downsampling the signal is equivalent to scaling the wavelet

#Therefore, the DWT algorithm is as follows: 
#Take the signal and convolve it with the father wavelet and the mother wavelet. 
#Downsample the outputs. 
#Store the convolution with the mother wavelet as coefficients. 
#Take the output of the father wavelet and start at step 1 again.

#Downsample the signal x1 by a factor of 2. That is, take every other point in 
#x1 starting with x1[0]. The syntax in Python is x1[start:end:skip]. You can 
#either just enter your answer or use the print statement

import numpy as np
t = np.arange(64)
x1 = np.sin(2*np.pi*0.02*t)
print(x1[0:64:2]) #or
print(x1[0::2])

#The DWT is not simply the discretized version of the CWT:
#The CWT operates on discretized data too. The DWT, however, uses a different
# kind of sampling which makes it more efficient.

#True statements abpout a DWT:
#It is efficient to calculate.
#It increases frequency resolution at low frequencies.
#It increases temporal resolution at high frequencies.
#You can make custom wavelets suitable for your application.
#The DWT keeps the adaptive tradeoff between temporal and frequency resolution 
#that the CWT possesses. However, it also is much more efficient to calculate 
#since it uses a sparser sampling

#Using PyWavelets
#Let's run a DWT on our non-stationary signal. There is currently no DWT 
#functionality in the SciPy or NumPy packages; we will turn to a package called 
#PyWavelets.

#PyWavelets has a number of predefined wavelets for us to use. We can see the 
#entire list like this:
import pywt
print(pywt.wavelist())

#arbitrarily pick a Coiflets 2

w = pywt.Wavelet('coif2')
print(w)

#Let's plot the low-pass, father wavelet, and high-pass, mother wavelet, 
#decomposition filters on separate graphs

pl.figure(1)
hl=pl.stem(range(0,w.dec_len),w.dec_lo, linefmt='#5D5166', markerfmt='o', basefmt='#FF971C')
pl.setp(hl[0],color='#5D5166')
pl.gca().set(xlim=(-1,12))
pl.show()

pl.figure(2)
hl=pl.stem(range(0,w.dec_len),w.dec_hi, linefmt='#5D5166', markerfmt='o', basefmt='#FF971C')
pl.setp(hl[0],color='#5D5166')
pl.gca().set(xlim=(-1,12))
pl.show()

#Let's use PyWavelets to perform one level of filtering. Before we do that, 
#though, let's define a new signal that will illustrate the low- and high-pass 
#filter concepts. Our new signal is going to be a simple sine wave with some 
#Gaussian noise added to it

npts = 256
t = np.linspace(0,1,npts)
np.random.seed(24935189)
x = np.sin(2*np.pi*t) + 0.1*np.random.randn(npts)
pl.figure(3)
pl.plot(t,x, color='#5D5166')
pl.show()

#Now, apply a single level of filtering to x and plot the result

cA1, cD1 = pywt.dwt(x, wavelet=w)
t1 = np.linspace(0,1,len(cD1))
pl.figure(4)
pl.plot(t1,cA1/np.sqrt(2), color='#5D5166')
pl.plot(t1,cD1/np.sqrt(2), color='#FF971C')
pl.show()

#The purple curve shows the low-pass filtered signal, cA1, and the orange curve 
#shows the high-pass filtered signal. In other words, the purple curve is the 
#result of convolving the father wavelet with the signal and the orange curve 
#is the result of convolving the mother wavelet with the signal. 

#Note how the purple curve remained a noisy sine curve, but the curve is 
#smoother than it was before, as some noise has been filtered in the high-pass. 
#Also, notice that the low-pass signal has a larger range of values because it 
#has been scaled by sqrt(2) to normalize the energy. Recall that the signal has 
#been downsampled by a factor of two and the energy is proportional to the 
#signal squared. Let's plot the original signal together with the low-passed 
#curve

pl.figure(5)
pl.plot(t,x,color='#5D5166')
pl.plot(t1,cA1/np.sqrt(2), color='#FF971C', linewidth=1.7)
pl.show()

#The figure shows more directly that the low-passed signal (orange) is smoother 
#than the original (purple) and normalizing the low-pass filtered version by the 
#sqrt(2) aligns it to the original. We can perform an 
#additional level of wavelet transform using cA1 as the new signal.

cA2, cD2 = pywt.dwt(cA1, wavelet=w)
t2 = np.linspace(0,1,len(cD2))
pl.figure(6)
pl.plot(t2,cA2,color='#5D5166')
pl.plot(t2,cD2,color='#FF971C')
pl.show()

#The figure shows that the low-passed signal, cA2, has been filtered and scaled 
#again. curve remains noisy, but is clearly less so than the original and the 
#level 1 version.

#PyWavelets can tell us the total number of levels that are useful:

print(pywt.dwt_max_level(npts, w))
#4

#We could continue this "manual" calculation of the DWT for an additional two 
#levels, but we can also let PyWavelets do all of the levels for us with one 
#method. Before we do that, let's check the inverse DWT to make sure we get our 
#original signal back.

x_rec = pywt.idwt(cA1, cD1, wavelet=w)
pl.figure(7)
pl.plot(t,x, color='#5D5166', lw=4)
pl.plot(t,x_rec, color='#FF971C')
pl.show() 

#The yellow and purple lines overlay perfectly on each other. The two signals 
#differ only by a small floating point error

print(sum(x-x_rec))
#1.84574577844e-15

#We can also reconstruct from our second-level coefficients
x_rec_2 = pywt.idwt(cA2, cD2, wavelet=w)
# for unknown reasons, x_rec_2 is one sample longer than cA1
# the next step throws away the last sample
x_rec_2 = x_rec_2[0:-1] 
t_2 = np.linspace(0,1,len(x_rec_2))
pl.figure(8)
pl.plot(t_2,cA1, color='#5D5166', linewidth=4)
pl.plot(t_2,x_rec_2, color='#FF971C')
pl.show()

#This reproduces the first-level, low-pass filtered signal:
print('This reproduces the first-level, low-pass filtered signal')

#We mentioned earlier that DWTs were useful for denoising and compression. 
#Either cA1 or cA2 could be kept as partially denoised versions of the original 
#signal x
#, if we discarded all of the other coefficients and retained only cA2 and cD2, 
#we have effectively reduced the number of coefficients from 256 down to 144 
#(cA2 and cD2 each have 72 elements each). 

# If we also stored the type of wavelet that we used, then we can reconstruct a 
#lossy version of our signal using only cA2 and cD2. In effect, we have just 
#compressed the signal at the cost of having to compute the inverse DWT when we 
#want the lossy signal back

#We'll return to performing the full DWT on the signal

cA4,cD4,cD3,cD2,cD1 = pywt.wavedec(x, w)
#This is one way to obtain the wavelet coefficients and the remaining 
#low-passed signal. It requires us to manually write down a separate variable 
#for each level of coefficients, so you would need to know how many levels in 
#advance. You can also simply store all the coefficients into a single array:

coeffs = pywt.wavedec(x, w, level=pywt.dwt_max_level(npts, w))

#coeffs[0] stores the low-pass, filtered signal and coeffs[1:] stores all of 
#the high-pass coefficients

#Let's plot the high-pass coefficients that represent the similarity of the 
#signal with the wavelet at different shifts. To make plotting easier, we are 
#going to interpolate the coefficient values (remember, each level is taken 
#from downsampled versions of the signal and are therefore not defined at all 
#time points). 

levels = pywt.dwt_max_level(npts, w)
cA4,cD4,cD3,cD2,cD1 = pywt.wavedec(x, w, level=levels)
cD = np.zeros( (levels,npts) ) # for storing interpolated values
for i in range(0,levels):
    exec('cDtmp = cD%d'%(i+1)) # runs the command inside the string
    ttmp = (t[-1]/len(cDtmp))*np.arange(0,len(cDtmp))
    cD[i,:] = np.interp(t, (t[-1]/len(cDtmp))*(np.arange(0,len(cDtmp))+0.5), cDtmp)
pl.ylabel('level')
pl.figure(8, figsize=(4,3))
pl.pcolor(np.append(t,1), np.arange(0,levels+1)+0.5, cD)
pl.gca().set(ylim=((0.5,levels+0.5)), yticks=np.arange(0,levels)+1)
pl.show()  

#the pseudo-color graph shows the convolution of the mother wavelet with the 
#low-pass filtered signal at each level
#to see the sine wave oscillation we would need to go to higher levels (i.e.,
# lower frequencies).
#PyWavelets tolf us that only four would be useful- recall that our filters for 
#coiflets two had length 12. Our signal started out with length 256. After 
#downsampling so many levels, at some point, it's not going to be worth it to 
#keep doing the convolutions because the wavelet will be as large or larger 
#than the signal itself

import pywt
import numpy as np
data = np.loadtxt('1-4-4 Data.csv')
m=len(data)
print(m)
#number of useful levels that can be performed on data using the Symlets 2 
#wavelet ('sym2'). 
w = pywt.Wavelet('sym2')
level=pywt.dwt_max_level(m,w)
#The DWT downsamples the data at every level. Having more samples in the data 
#or choosing a smaller wavelet will allow more levels.

#we coerce PyWavelets to keep doing more levels
levels = 6
coeffs = pywt.wavedec(x, w, level=levels)
cD = np.zeros( (levels,npts) )
for i in range(1,levels+1):  # coeffs[0] is the low pass-filtered signal
    ttmp = (t[-1]/len(coeffs[i]))*(np.arange(0,len(coeffs[i]))+0.5)
    cD[levels-i,:] = np.interp(t, ttmp, coeffs[i])  # we have to reverse the order of the levels
pl.figure(10, figsize=(4,3))
pl.clf()
pl.pcolor(np.append(t,1), np.arange(0,levels+1)+0.5, cD)
pl.gca().set(ylim=((0.5,levels+0.5)), yticks=np.arange(0,levels)+1)
pl.show

#it's becoming clear that our choice of the coiflets two wavelet was a poor one. 
#Basically, it lacks the structure to discern the shape of the sine wave

# We're going to use the Daubechies 1 wavelet, which is nothing more than a -1 
#followed by a 1

w = pywt.Wavelet('db1')
levels = pywt.dwt_max_level(npts, w)
coeffs = pywt.wavedec(x, w, level=levels)
cD = np.zeros( (levels,npts) )
for i in range(1,levels+1):  # coeffs[0] is the low pass-filtered signal
    ttmp = np.linspace(0,1,len(coeffs[i]))
    cD[levels-i,:] = np.interp(t, ttmp, coeffs[i])  # we have to reverse the order of the levels
pl.figure(10, figsize=(4,3))
pl.clf()
pl.pcolor(np.append(t,1), np.arange(0,levels+1)+0.5, cD)
pl.gca().set(ylim=((0.5,levels+0.5)), yticks=np.arange(0,levels)+1)
pl.show()

#Perform a DWT of data at the maximum number of levels using the sym2 wavelet
import pywt
import numpy as np
data = np.loadtxt('1-4-4 Data.csv')
m = len(data)
w = pywt.Wavelet('sym2')
levels = pywt.dwt_max_level(m, w)
coeffs = pywt.wavedec(data, w, level=levels)

#Generally, choosing a wavelet that "looks like" the features you are searching 
#for is a good rule of thumb

#Time-frequency analysis is an important method used for analyzing signals whose 
#frequency content changes over time.The choice between using STFTs and 
#wavelets is not always an easy one. We tried to give you an idea of when you 
#might want one versus the other, but ultimately the decision varies depending 
#on the problem you are trying to solve.