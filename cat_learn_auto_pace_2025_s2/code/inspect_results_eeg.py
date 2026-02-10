import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os

dir_data_eeg = '../data_lab_eeg'

# P2_D1.bdf
# P2_D2.bdf
# P2_D3.bdf
# P2_D4.bdf
# P2_D5.bdf
# P3_D1.bdf
# P3_D2.bdf
# P3_D3.bdf
# P3_D4.bdf
# P3_D5.bdf

f = os.path.join(dir_data_eeg, 'P2_D1.bdf')
raw = mne.io.read_raw_bdf(f, preload=True)

# triggers
events = mne.find_events(raw, stim_channel='Status', shortest_event=1)

# TRIG = {
#     "EXP_START": 10,
#     "ITI_ONSET": 11,
#     "STIM_ONSET_A": 20,
#     "STIM_ONSET_B": 21,
#     "RESP_A": 30,
#     "RESP_B": 31,
#     "FB_COR": 40,
#     "FB_INC": 41,
#     "EXP_END": 15,
# }

# get epochs for STIM_ONSET_A
epochs = mne.Epochs(raw, events, event_id=20, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)

# plot functional connectivity using coherence
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

# Define parameters for connectivity analysis
fmin, fmax = 8., 12.  # Alpha band
sfreq = raw.info['sfreq']
con_methods = ['coh']

# Compute connectivity
con, freqs, times, n_epochs, n_tapers = spectral_connectivity_epochs(
    epochs, method=con_methods, mode='fourier', sfreq=sfreq,
    fmin=fmin, fmax=fmax, faverage=True, tmin=0.0, tmax=0.8,
    mt_adaptive=False, n_jobs=1)

# Prepare labels and colors for circular plot
labels = epochs.ch_names
node_order = labels
color_dict = {label: 'skyblue' for label in labels}
node_colors = [color_dict[label] for label in node_order]
# Plot connectivity circle
fig = plot_connectivity_circle(con[:, :, 0], node_order, n_lines=300,
                               title='Alpha Band Coherence',
                               node_colors=node_colors)
plt.show()
