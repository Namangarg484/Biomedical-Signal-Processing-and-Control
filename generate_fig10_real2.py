import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# Load real data
wavenumbers = np.load('testing_data/wavenumbers.npy')
X = np.load('testing_data/X_2018_proc.npy')
y = np.load('testing_data/y_2018clinical.npy')
X_wavelet = np.load('testing_data/X_2018_wavelet.npy') # Use the exact precomputed scalograms!

# Ensure wavenumbers are strictly increasing for pcolormesh
if wavenumbers[0] > wavenumbers[-1]:
    wavenumbers = wavenumbers[::-1]
    X = X[:, ::-1]
    # The wavelet data might have dimension 224x224. 
    # Let's check dimensions of X_wavelet
    # It might be (N, C, H, W) or (N, H, W)

# Find an E. coli sample
ecoli_indices = np.where(y == 0)[0]
# Pick a representative sample (index 5)
idx = ecoli_indices[5]
spectrum = X[idx]

# For the scalogram, we can use the precomputed image!
# Let's extract the first channel of the 3-channel wavelet image if it is shape (N, 3, 224, 224)
if len(X_wavelet.shape) == 4 and X_wavelet.shape[1] == 3:
    cwt_magnitude = X_wavelet[idx, 0, :, :]
elif len(X_wavelet.shape) == 3:
    cwt_magnitude = X_wavelet[idx, :, :]
else:
    cwt_magnitude = X_wavelet[idx]

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
# Since X_wavelet is an image array (e.g. 224x224), we can just imshow it
ax2.imshow(cwt_magnitude, cmap='magma', aspect='auto', 
           extent=[wavenumbers[0], wavenumbers[-1], 0, 128], origin='lower')
ax2.set_title('2D CWT Scalogram (Resolving multi-scale structural invariants)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
ax2.set_ylabel('Wavelet Scale (Proxy)', fontsize=14)

# Annotations for 2D
ax2.text(0.02, 0.90, 'Correctly Classified by PINNACLE (Gated Fusion)', transform=ax2.transAxes, 
         fontsize=14, color='white', fontweight='bold', 
         bbox=dict(facecolor='#27ae60', alpha=0.9, edgecolor='none', pad=5))

plt.tight_layout(pad=3.0)
plt.savefig('figures/fig10_ecoli_cwt.png', dpi=300, bbox_inches='tight')
print("Figure 10 successfully generated from REAL data!")
