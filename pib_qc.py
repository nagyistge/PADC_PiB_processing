#+ echo=False
from nilearn.plotting import show, plot_stat_map, plot_anat, plot_img, plot_glass_brain, cm
from nilearn.image import iter_img
from nilearn.masking import apply_mask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
import nibabel as nib
from dipy.viz import regtools

frameTiming = pd.read_csv(frameTimingCsvFile)
# check that frameTiming has columns named frameStart and frameEnd
for col in ['frameStart','frameEnd']:
    if not col in frameTiming.columns:
        sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
frameStart = frameTiming['frameStart']
frameEnd = frameTiming['frameEnd']
# sanity checks on frameTiming spreadsheet entries
assert(len(frameTiming)>1)
assert(all(frameStart<frameEnd))
assert(all(frameStart[1:]==frameEnd[:len(frameEnd)-1]))

frameStart = frameStart.as_matrix() #tolist()
frameEnd = frameEnd.as_matrix() #tolist()

splitIndex = next((i for i,t in enumerate(frameStart) if t>=splitTime), len(frameTiming))

# Compute the time mid-way for each time frame
t = (frameStart + frameEnd)/2

#' # Realignment QC

#' ## Reference region Time Activity Curve (TAC)
#+ echo=False
masked_data = apply_mask(petrealignedfile, maskfile)
ref_TAC = np.mean(masked_data,axis=1)

plt.figure(figsize=(7,5))
plt.plot(t,ref_TAC)
plt.title('Time Activity Curve')
plt.xlabel('Time (min)', fontsize=16)
plt.ylabel('Activity', fontsize=16)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
show()

#' ## Time realignment parameters
#+ echo=False
rp = pd.read_csv(realignParamsFile,delim_whitespace=True, header=None).as_matrix()
translation = rp[1:,:3]
rotation = rp[1:,3:] * 180 / pi

plt.figure(figsize=(7,5))
plt.plot(t[splitIndex:],translation[:,0],label='x')
plt.plot(t[splitIndex:],translation[:,1],label='y')
plt.plot(t[splitIndex:],translation[:,2],label='z')
plt.legend()
plt.title('Translation over time')
plt.xlabel('Time (min)', fontsize=16)
plt.ylabel('Translation (mm)', fontsize=16)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(t[splitIndex:],rotation[:,0],label='x')
plt.plot(t[splitIndex:],rotation[:,1],label='y')
plt.plot(t[splitIndex:],rotation[:,2],label='z')
plt.legend()
plt.title('Rotation over time')
plt.xlabel('Time (min)', fontsize=16)
plt.ylabel('Rotation (degrees)', fontsize=16)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
plt.show()

#' ## Reference region superimposed on PiB 20 min mean
#+ echo=False
display = plot_anat(pet20minfile, display_mode="ortho")
display.add_overlay(maskfile, cmap=cm.red_transparent)
show()

#' ## Axial slices over time frames
#+ echo=False
count = 1
fig = plt.figure(figsize=(20,15))
for img in iter_img(petrealignedfile):
    subfig = fig.add_subplot(10,4,count)
    plot_stat_map(img, bg_img=None, display_mode="z", cut_coords=1, annotate=False, draw_cross=False, colorbar=False, axes=subfig)
    count += 1
fig.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
show()



#' # PiB-MRI coregistration QC
#+ echo=False
mri = nib.load(mrifile)
pibreg = nib.load(petregfile)

# Visualize
p = regtools.overlay_slices(pibreg.get_data(), mri.get_data(), None, 0, "PiB 20 min", "MRI")
p = regtools.overlay_slices(pibreg.get_data(), mri.get_data(), None, 1, "PiB 20 min", "MRI")
p = regtools.overlay_slices(pibreg.get_data(), mri.get_data(), None, 2, "PiB 20 min", "MRI")

display = plot_anat(petregfile)
display.add_edges(mrifile)
show()



#' # MNI QC
#+ echo=False
plot_img(pet_mnifile, title='PiB 20 min', threshold=10000, colorbar=True)
show()

plot_img(dvr_wlr_mnifile, title='DVR (WLR)', colorbar=True)
show()
plot_glass_brain(dvr_wlr_mnifile, title='DVR (WLR)', display_mode='lyrz', threshold=1.8, colorbar=True)
show()

plot_img(dvr_lrsc_mnifile, title='DVR (LRSC)', colorbar=True)
show()
plot_glass_brain(dvr_lrsc_mnifile, title='DVR (LRSC)', display_mode='lyrz', threshold=1.8, colorbar=True)
show()

plot_img(r1_wlr_mnifile, title='R1 (WLR)', colorbar=True)
show()

plot_img(r1_lrsc_mnifile, title='R1 (LRSC)', colorbar=True)
show()

plot_img(ea_mnifile, title='EA', colorbar=True)
show()

plot_img(suvr_mnifile, title='SUVR', colorbar=True)
show()
plot_glass_brain(suvr_mnifile, title='SUVR', display_mode='lyrz', threshold=2.8, colorbar=True)
show()



plot_img(dvr_wlr_pvc_mnifile, title='DVR (WLR PVC)', colorbar=True)
show()
plot_glass_brain(dvr_wlr_pvc_mnifile, title='DVR (WLR PVC)', display_mode='lyrz', threshold=1.8, colorbar=True)
show()

plot_img(dvr_lrsc_pvc_mnifile, title='DVR (LRSC PVC)', colorbar=True)
show()
plot_glass_brain(dvr_lrsc_pvc_mnifile, title='DVR (LRSC PVC)', display_mode='lyrz', threshold=1.8, colorbar=True)
show()

plot_img(r1_wlr_pvc_mnifile, title='R1 (WLR PVC)', colorbar=True)
show()

plot_img(r1_lrsc_pvc_mnifile, title='R1 (LRSC PVC)', colorbar=True)
show()

plot_img(ea_pvc_mnifile, title='EA (PVC)', colorbar=True)
show()

plot_img(suvr_pvc_mnifile, title='SUVR (PVC)', colorbar=True)
show()
plot_glass_brain(suvr_pvc_mnifile, title='SUVR (PVC)', display_mode='lyrz', threshold=2.8, colorbar=True)
show()
