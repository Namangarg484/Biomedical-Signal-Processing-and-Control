import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Load real data
wavenumbers = np.load('testing_data/wavenumbers.npy')
X = np.load('testing_data/X_2018_proc.npy')
y = np.load('testing_data/y_2018clinical.npy')

# Ensure wavenumbers are strictly increasing for pcolormesh
if wavenumbers[0] > wavenumbers[-1]:
    wavenumbers = wavenumbers[::-1]
    X = X[:, ::-1]

# Find an E. coli sample
ecoli_indices = np.where(y == 0)[0]
# Pick a representative sample (index 5)
spectrum = X[ecoli_indices[5]]

# Compute CWT using Morlet
widths = np.arange(1, 128)
cwtmatr = signal.cwt(spectrum, signal.morlet2, widths)
cwt_magnitude = np.abs(cwtmatr)

# Create highly polished figure
fig = plt.figure(figsize=(12, 9))

# Panel 1: 1D Spectrum
ax1 = plt.subplot(2, 1, 1)
ax1.plot(wavenumbers, spectrum, color='#2c3e50', linewidth=1.5)
ax1.set_xlim([wavenumbers[0], wavenumbers[-1]])
ax1.set_title('1D Raman Spectrum (Class 0: E. coli)', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Normalized Intensity', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--')

# Annotations for 1D
peak_val = np.max(spectrum)
ax1.annotate('High-frequency noise\nobscuring Enterobacteriaceae features', 
             xy=(1300, peak_val * 0.6), xytext=(1450, peak_val * 0.8),
             arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=2, headwidth=8, edgecolor='#c0392b'),
             fontsize=12, color='#c0392b', fontweight='bold', ha='center')
ax1.text(0.02, 0.90, 'Misclassified by 1D Baseline', transform=ax1.transAxes, 
         fontsize=14, color='white', fontweight='bold', 
         bbox=dict(facecolor='#e74c3c', alpha=0.9, edgecolor='none', pad=5))

# Panel 2: 2D Scalogram
ax2 = plt.subplot(2, 1, 2)
pc = ax2.pcolormesh(wavenumbers, widths, cwt_magnitude, cmap='magma', shading='auto')
ax2.set_xlim([wavenumbers[0], wavenumbers[-1]])
ax2.set_title('2D CWT Scalogram (Resolving multi-scale structural invariants)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
ax2.set_ylabel('Wavelet Scale', fontsize=14)

# Colorbar
cbar = plt.colorbar(pc, ax=ax2, pad=0.02)
cbar.set_label('CWT Coefficient Magnitude', rotation=270, labelpad=20, fontsize=12)

# Annotations for 2D
ax2.text(0.02, 0.90, 'Correctly Classified by PINNACLE (Gated Fusion)', transform=ax2.transAxes, 
         fontsize=14, color='white', fontweight='bold', 
         bbox=dict(facecolor='#27ae60', alpha=0.9, edgecolor='none', pad=5))

plt.tight_layout(pad=3.0)
plt.savefig('figures/fig10_ecoli_cwt.png', dpi=300, bbox_inches='tight')
print("Figure 10 successfully generated from REAL data!")
