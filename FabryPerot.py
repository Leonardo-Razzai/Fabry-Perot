import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit

MEDIUM_SIZE = 11
BIGGER_SIZE = 13

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

base_font = {'family': 'serif',
        'size': MEDIUM_SIZE,
        }

title_font = {
        'family': 'serif',
        'color':  'black',
        'size': BIGGER_SIZE,
        'weight' : 'bold'
        }

def Lorentzian(freq, A0, gamma, f0):
  return A0 * gamma**2 / 4 / ((freq - f0)**2 + gamma**2 / 4)

class SpectrumAnalyzer:
  def __init__(self, file_name: str) -> None:
    data = pd.read_csv(f'{file_name}.csv')
    self.Sig = -data['Sig [V]'].to_numpy()
    self.Time = data['Time [s]'].to_numpy()
    self.Tot_PW = None
    self.peaks = []
    
    self.TimeToFreq = 0
  
  def select_peaks(self, intervals):
    
    # Select Time scale
    Time = self.Time
    
    # For each peak specified, create a class Peak and append to self.peaks
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(self.Time, self.Sig)
    ax.set_xlabel('Time [s]', fontdict=base_font)
    ax.set_ylabel('Sig [V]', fontdict=base_font)
    ax.set_ylim((0, 0.3))
    ax.set_title('Transmission spectrum Fabry-Perot', fontdict=title_font)
    ax.grid()
    
    for interval in intervals:
      min_Time = np.min(interval)
      max_Time = np.max(interval)
      index = (Time > min_Time) * (Time < max_Time)
      peak_Time = Time[index]
      peak_intensity = self.Sig[index]
      self.peaks.append(Peak(peak_Time, peak_intensity))
      ax.vlines(x=[min_Time, max_Time], ymin=0, ymax=0.3, colors='crimson', linestyles='dashed')
    
    ax.vlines(x=[min_Time, max_Time], ymin=0, ymax=0.3, colors='crimson', linestyles='dashed', label='Peaks')
    ax.legend(prop=base_font)
    plt.show()

  def plot_spectrum(self):
      
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    if self.TimeToFreq != 0:
      ax.plot(self.Time * self.TimeToFreq, self.Sig, color='black', lw=0.7)
      ax.set_xlabel('Freq [GHz]', fontdict=base_font)
    else:
      ax.plot(self.Time, self.Sig, color='black', lw=0.7)
      ax.set_xlabel('Time [s]', fontdict=base_font)
    ax.set_ylabel('Sig [V]', fontdict=base_font)
    ax.set_title('Transmission spectrum Fabry-Perot', fontdict=title_font)
    ax.grid()
    return ax
  
  def Set_TimeToFreq_conv(self, conv):
    self.TimeToFreq = conv
    
  def Integrate(self, interval):
    index = (self.Time > interval[0]) * (self.Time < interval[1])
    self.Tot_PW = np.sum(self.Sig[index] * (self.Time[1]-self.Time[0]))
  
class Peak:
  def __init__(self, Time: np.ndarray, Sig: np.ndarray) -> None:
    self.Time = Time
    self.Sig = Sig
    self.Sig_corrected = Sig
    self.Integral = None
    self.peak_pos = None
    self.height = None
    self.pw_fraction = None
    
  def subtract_background(self, intervals: list, deg=1, plot=False):
    min1 = np.min(intervals[0])
    max1 = np.max(intervals[0])
    min2 = np.min(intervals[1])
    max2 = np.max(intervals[1])
    index = (self.Time > min1) * (self.Time < max1) + (self.Time > min2) * (self.Time < max2)
    x = self.Time[index]
    y = self.Sig[index]
    
    params, _ = np.polyfit(x, y, deg=deg, cov=True)
    params = np.flip(params)
    
    self.background = sum([param * self.Time**i for i, param in enumerate(params)])
    self.Sig_corrected = self.Sig - self.background
    
    if plot==True:
      ax = self.plot_raw_spectrum()
      ax.plot(self.Time, self.background, '--', color='black', label='background')
      ax.set_title('Spectrum with background shown')
      ax.legend(prop=base_font)
  
  def integrate(self):
    dx = self.Time[1] - self.Time[0]
    self.Integral = np.sum(self.Sig_corrected * dx)
  
  def smooth(self, deg=3):
    ker = np.ones(deg) / deg
    self.Sig_corrected = np.convolve(self.Sig_corrected, ker, mode='same')
  
  def fit(self, interval, name: str, plot=False, tot_pw=1):
    min = interval[0]
    max = interval[1]
    index = (self.Time > min) * (self.Time < max)
    x = self.Time[index]
    y = self.Sig_corrected[index]
    
    A0 = np.max(self.Sig)
    gamma = (self.Time[-1] - self.Time[0])/2
    f0 = (self.Time[-1] + self.Time[0])/2
    popt, pcov = curve_fit(Lorentzian, x, y, p0 = [A0, gamma, f0])
    
    self.peak_pos = popt[2]
    self.height = popt[0]
    self.pw_fraction = self.Integral/tot_pw
    
    print(f'\nPeak height = {self.height*1e3:.0f} mV, Peak Pos = {self.peak_pos*1e3:.0f} ms, PW Ratio = {self.pw_fraction:.3f}\n')
    
    if plot:
      x_fit = np.linspace(self.Time[0], self.Time[-1], 200)
      y_fit = Lorentzian(x_fit, *popt)
      
      _, ax = plt.subplots(1, 1, figsize=(10, 4))
      ax.plot(self.Time, self.Sig, 'o', ms=1, label='Data', color='royalblue')
      label = r'- $A_0$ = ' + f'{self.height*1e3:.1f} mV\n' + r'- $f_0$ = ' + f'{self.peak_pos*1e3:.1f} ms\n' + r'$PW_{ratio}$ = ' + f'{self.Integral/tot_pw:.3f}'
      ax.plot(x_fit, y_fit, '--', label='Fit:\n'+label, color='crimson')
      ax.set_xlabel('Time [s]', fontdict=base_font)
      ax.set_ylabel('Sig [V]', fontdict=base_font)
      ax.set_ylim((0, 0.3))
      ax.set_title('Transmission spectrum Fabry-Perot: '+name, fontdict=title_font)
      ax.grid()
      ax.legend(prop=base_font)
      plt.show()
    
  def plot_raw_spectrum(self):
    _, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.plot(self.Time, self.Intensity)
    ax.set_xlabel('Time [s]', fontdict=base_font)
    ax.set_ylabel('Sig [V]', fontdict=base_font)
    ax.set_title('Transmission spectrum Fabry-Perot')
    ax.grid()
    return ax

  def plot_corrected_spectrum(self, title='Transmission spectrum Fabry-Perot without background'):
    _, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.plot(self.Time, self.Intensity_corrected)
    ax.set_xlabel('Time [s]', fontdict=base_font)
    ax.set_ylabel('Sig [V]', fontdict=base_font)
    ax.set_title(title, fontdict=title_font)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    return ax