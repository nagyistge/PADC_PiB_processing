# # Amyloid PET Processing for Preclinical Alzheimer's Disease Consortium
#
# The processing steps can be summarized as follows:
# 1. Time frame alignment
# 2. MRI-PET coregistration (MRIs have already been processed and anatomical regions have been defined)
# 3. Partial volume correction of time frame data
# 4. Extraction of early amyloid (EA), SUVR, DVR, R<sub>1</sub> images
# 5. ROI summary calculation
# 6. Spatial normalization of all output images to MNI space
#
# Steps 4-6 will be performed with and without partial volume correction.

# Import packages
import os, sys, logging
import pandas as pd
import numpy as np
import scipy as sp
import math
from glob import glob
from collections import OrderedDict

# nipype
import nipype.interfaces.io as nio
from nipype.interfaces import fsl, petpvc, ants
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline.plugins.callback_log import log_nodes_cb
from nipype_funcs import * # nipype_funcs.py must be in current working directory (or otherwise discoverable by python)
from nipype import config, logging
config.enable_debug_mode()
logging.update_logging(config)


# prefix for the data collection site and site name
sitePrefix = 'P05'
siteName = 'aibl'

# directory to store the workflow results
output_dir = os.path.join(os.getcwd(), os.pardir, 'results', sitePrefix+'_amyloid')

# number of parallel processes
n_procs = 20


# spreadsheet with the following columns: ID, amyloidpath, amyloidtimingpath, musemripath, muselabelpath
organization_spreadsheet = os.path.join(os.getcwd(), os.pardir, 'inputs', 'PADC_PET_MUSE.xlsx')

# columns required in the spreadsheet
required_cols = ['ID','amyloidpath','musemripath','muselabelpath']

# values to be treated as missing in the spreadsheet - do not include NA as a null value as it is a valid EMSID
NAN_VALUES = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan','']


# Amyloid PET processing parameters
# PVC smoothing parameters: PET scanner PSF FWHM (in mm)
pvc_fwhm_x = 6.7
pvc_fwhm_y = 6.7
pvc_fwhm_z = 6.7

# Smoothing parameter (in mm) used for SUVR and SRTM
smooth_fwhm = 4.25

# For trimmed mean of ROI signal, proportion to cut (exclude) from each tail of the distribution
proportiontocut = 0


if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


# Read in the organization spreadsheet and extract information
data_table = pd.read_excel(organization_spreadsheet, sheetname=siteName, keep_default_na=False, na_values=NAN_VALUES)

for col in required_cols:
    if not col in data_table.columns:
        sys.exit('Required column ' + col + ' is not present in the data organization spreadsheet ' + \
                         organization_spreadsheet + '!')

# Find all visits with amyloid PET and MUSE labels
data_table = data_table[required_cols].dropna(axis=0, how='any')

musemri_list = data_table['musemripath'].values.tolist()
muselabel_list = data_table['muselabelpath'].values.tolist()
amyloid_list = data_table['amyloidpath'].values.tolist()
id_list = data_table['ID'].values.tolist()

# Form dictionaries, with IDs as keys and paths to images as values
amyloid_dict = dict(zip(id_list, amyloid_list))
musemri_dict = dict(zip(id_list, musemri_list))
muselabel_dict = dict(zip(id_list, muselabel_list))


# ## 1. INPUTS

# We set up the nipype Nodes that will act as the inputs to our Workflows. The `infosource` Node allows for iterating over scan IDs. The remaining input Nodes allow for the retrieval of the amyloid PET, processed MRI, and label images given the scan IDs (`getpet`, `getmusemri`, `getmuselabel`, respectively), as well as the retrieval of the text files detailing the frame timing information (`getpettiming`).

# placeholder Node to enable iteration over scans
infosource = Node(interface=IdentityInterface(fields=['id']), name='infosource')
infosource.iterables = ('id', id_list)

# get full path to amyloid scan corresponding to id from spreadsheet
getpet = Node(Function(input_names=['key','dict'],
                       output_names=['pet'],
                       function=get_value),
              name='getpet')
getpet.inputs.dict = amyloid_dict

# get full path to MRI corresponding to id from spreadsheet, in same space as MUSE labels
getmusemri = Node(Function(input_names=['key','dict'],
                           output_names=['musemri'],
                           function=get_value),
                  name='getmusemri')
getmusemri.inputs.dict = musemri_dict

# get full path to MUSE label image corresponding to id from spreadsheet, in same space as MRI
getmuselabel = Node(Function(input_names=['key','dict'],
                             output_names=['muselabel'],
                             function=get_value),
                    name='getmuselabel')
getmuselabel.inputs.dict = muselabel_dict

# ## 2. MRI-PET COREGISTRATION
#
# Our goal is to perform image processing in native PET space to produce parametric images. We have chosen this approach (rather than processing in native MRI space) for two reasons:
# 1. This approach avoids the use of PET data interpolated to match the voxel size of the MRI scans for generating parametric images. Such an interpolation is undesirable because of the large difference between PET and MRI voxel sizes.
# 2. Fewer brain voxels in native PET space allows for faster computation of voxelwise kinetic parameters for the whole brain.
#
# It should be noted that PET space processing is not without disadvantages. Anatomical labels have to be interpolated to match the PET voxel size, which yields inaccuracies. While these inaccuracies are not important for reference region or ROI definitions for computing averages, they are influential for partial volume correction.
#
# We have structural MRIs that have already been preprocessed and anatomically labeled. To bring anatomical labels to PET space, we will perform coregistration of the PET and the MRI.
#
# * `reorient`, `reorientmri` and `reorientlabel`:  Apply 90, 180, or 270 degree rotations as needed about the $x,y,z$ axes to match the MNI152 orientation.
# * `pet_to_mri`: We use the image with finer spatial resolution (MRI) as the reference, and the amyloid PET 20-min average as the moving image, to perform rigid alignment with normalized mutual information cost function, using FSL's FLIRT method.
# * `invertTransform`: Since we want anatomical labels in PET space, we invert the rigid transformation.
# * `mri_to_pet` and `labels_to_pet`: We apply the inverted transformation to the MRI and anatomical labels to bring them to PET space.
#
# _Note:_ For acquisitions that do not allow for the computation of a 20-min average image, we use the mean of the entire acquisition to perform coregistration with the MRI.

# Reorient
collapse = Node(interface=fsl.ImageMaths(op_string=' -Tmean', suffix='_mean', output_type='NIFTI'), name="collapse")
reorient = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorient")

# Reorient MRI and label
reorientmri = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientmri")
reorientlabel = Node(interface=fsl.Reorient2Std(output_type='NIFTI'), name="reorientlabel")

# MRI coregistration, rigid, with normalized mutual information
pet_to_mri = Node(interface=fsl.FLIRT(cost='normmi', cost_func='normmi', dof=6,
                                      searchr_x=[-10,10], searchr_y=[-10,10], searchr_z=[-10,10],
                                      coarse_search=10, fine_search=5, force_scaling=True),
                  name="pet_to_mri")

invertTransform = Node(interface=fsl.ConvertXFM(invert_xfm=True), name="invertTransform")
mri_to_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True), name="mri_to_pet")
labels_to_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True, interp='nearestneighbour'), name="labels_to_pet")

coreg_qc = Node(interface=coreg_snapshots(), name="coreg_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','coreg_wf')
datasink.inputs.substitutions = [('_id_',''),
                                 ('_roi',''),
                                 ('_merged',''),
                                 ('_mean',''),
                                 ('flirt','coreg'),
                                 ('_reoriented',''),
                                 ('_masked','')]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

coreg_workflow = Workflow(name="coreg_workflow")
coreg_workflow.base_dir = os.path.join(output_dir,'coreg_workingdir')
coreg_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'coreg_crashdumps')}}
coreg_workflow.connect([(getmusemri, reorientmri, [('musemri','in_file')]),
                        (getmuselabel, reorientlabel, [('muselabel','in_file')]),

                        (getpet, collapse, [('pet','in_file')]),
                        (collapse, reorient, [('out_file','in_file')]),

                        (reorient, pet_to_mri, [('out_file','in_file')]),
                        (reorientmri, pet_to_mri, [('out_file','reference')]),

                        (pet_to_mri, invertTransform, [('out_matrix_file','in_file')]),

                        (reorientmri, mri_to_pet, [('out_file','in_file')]),
                        (reorient, mri_to_pet, [('out_file','reference')]),
                        (invertTransform, mri_to_pet, [('out_file','in_matrix_file')]),

                        (reorientlabel, labels_to_pet, [('out_file','in_file')]),
                        (reorient, labels_to_pet, [('out_file','reference')]),
                        (invertTransform, labels_to_pet, [('out_file','in_matrix_file')]),

                        (reorient, coreg_qc, [('out_file','petavgfile')]),
                        (mri_to_pet, coreg_qc, [('out_file','mriregfile')]),

                        # save outputs
                        (reorient, datasink, [('out_file','reorientedpet')]), # 0.75-20min average (3D image) used for MRI coregistration
                        (pet_to_mri, datasink, [('out_file','coreg_pet'),
                                                ('out_matrix_file','coreg_pet.@param')]),
                        (mri_to_pet, datasink, [('out_file','coreg_mri')]), # MRI registered onto PET
                        (labels_to_pet, datasink, [('out_file','coreg_labels')]), # anatomical labels on PET
                        (coreg_qc, datasink, [('coreg_edges','QC'),
                                              ('coreg_overlay_sagittal','QC.@sag'),
                                              ('coreg_overlay_coronal','QC.@cor'),
                                              ('coreg_overlay_axial','QC.@ax')])
                       ])

coreg_workflow.write_graph('coreg.dot', graph2use='colored', simple_form=True)

amyloid_workflow = Workflow(name="amyloid_workflow")
amyloid_workflow.base_dir = output_dir
amyloid_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'amyloid_crashdumps')}}
amyloid_workflow.connect([# PET-to-MRI registration
                      (infosource, coreg_workflow, [('id','getpet.key'),
                                                    ('id','getmusemri.key'),
                                                    ('id','getmuselabel.key')])
                     ])


# ## 3. LABELS
#
# We manipulate the anatomical label image to define the reference region for kinetic parameter computation, designate ROIs for partial volume correction, and determine composite ROIs for regional average reporting. Below, we define which MUSE label indices and their combinations form these three categories.

# ROI groupings for whole brain
whole_brain_ROI_grouping = {'whole brain':[23,30,36,37,55,56,57,58,75,76,59,60,
                            35,47,48,31,32,61,62,81,82,83,84,85,86,87,88,89,90,
                            91,92,93,94,95,104,105,136,137,146,147,178,179,
                            120,121,142,143,162,163,164,165,182,183,190,191,204,205,
                            124,125,140,141,150,151,152,153,186,187,192,193,
                            112,113,118,119,174,175,
                            106,107,176,177,194,195,198,199,148,149,168,169,
                            122,123,132,133,154,155,200,201,202,203,
                            180,181,184,185,206,207,
                            160,161,128,129,144,145,156,157,196,197,
                            108,109,114,115,134,135,116,117,170,171,
                            100,101,138,139,166,167,102,103,172,173,
                            40,41,38,39,71,72,73]}

# ROI groupings for reference region
reference_ROI_grouping = {'cerebellar GM':[38,39,71,72,73]}

# ROI groupings for PVC based on MUSE labels
pvc_ROI_groupings = {
    'background':[0],
    'ventricles and CSF':[-1,4,11,46,49,50,51,52], # includes non-WM-hypointensities. -1 is a label we will generate to approximate sulcal CSF
    'basal ganglia':[23,30,36,37,55,56,57,58,75,76], # striatum (caudate, putamen), pallidum, nucleus accumbens, substantia nigra, basal forebrain
    'thalamus':[59,60],
    'brainstem':[35], # combine with thalamus?
    'hippocampus':[47,48],
    'amygdala':[31,32],
    'cerebral WM':[61,62,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95], # includes fornix, internal capsule, corpus callosum, ventral DC
    'inferior frontal GM':[104,105,136,137,146,147,178,179],
    'lateral frontal GM':[120,121,142,143,162,163,164,165,182,183,190,191,204,205],
    'medial frontal GM':[124,125,140,141,150,151,152,153,186,187,192,193],
    'opercular frontal GM':[112,113,118,119,174,175],
    'lateral parietal GM':[106,107,176,177,194,195,198,199],
    'medial parietal GM':[148,149,168,169],
    'inferior temporal GM':[122,123],
    'lateral temporal GM':[132,133,154,155,200,201,202,203],
    'supratemporal GM':[180,181,184,185,206,207],
    'inferior occipital GM':[160,161],
    'lateral occipital GM':[128,129,144,145,156,157,196,197],
    'medial occipital GM':[108,109,114,115,134,135],
    'limbic medial temporal GM':[116,117,170,171],
    'cingulate GM':[100,101,138,139,166,167],
    'insula GM':[102,103,172,173],
    'cerebellar WM':[40,41],
    'cerebellar GM':[38,39,71,72,73]
}

# sort by ROI name
pvc_ROI_groupings = OrderedDict(sorted(pvc_ROI_groupings.items(), key=lambda roi: roi[0]))

# ROIs for report spreadsheet
ROIs = {
    '3rd Ventricle':4,'4th Ventricle':11,'Right Accumbens Area':23,'Left Accumbens Area':30,
    'Right Amygdala':31,'Left Amygdala':32,'Brain Stem':35,'Right Caudate':36,'Left Caudate':37,
    'Right Cerebellum Exterior':38,'Left Cerebellum Exterior':39,
    'Right Cerebellum White Matter':40,'Left Cerebellum White Matter':41,
    'Right Hippocampus':47,'Left Hippocampus':48,'Right Inf Lat Vent':49,'Left Inf Lat Vent':50,
    'Right Lateral Ventricle':51,'Left Lateral Ventricle':52,'Right Pallidum':55,'Left Pallidum':56,
    'Right Putamen':57,'Left Putamen':58,'Right Thalamus Proper':59,'Left Thalamus Proper':60,
    'Right Ventral DC':61,'Left Ventral DC':62,
    'Cerebellar Vermal Lobules I-V':71,'Cerebellar Vermal Lobules VI-VII':72,'Cerebellar Vermal Lobules VIII-X':73,
    'Left Basal Forebrain':75,'Right Basal Forebrain':76,'frontal lobe WM right':81,'frontal lobe WM left':82,
    'occipital lobe WM right':83,'occipital lobe WM left':84,'parietal lobe WM right':85,'parietal lobe WM left':86,
    'temporal lobe WM right':87,'temporal lobe WM left':88,'fornix right':89,'fornix left':90,
    'anterior limb of internal capsule right':91,'anterior limb of internal capsule left':92,
    'posterior limb of internal capsule inc. cerebral peduncle right':93,
    'posterior limb of internal capsule inc. cerebral peduncle left':94,
    'corpus callosum':95,'Right ACgG  anterior cingulate gyrus':100,'Left ACgG  anterior cingulate gyrus':101,
    'Right AIns  anterior insula':102,'Left AIns  anterior insula':103,
    'Right AOrG  anterior orbital gyrus':104,'Left AOrG  anterior orbital gyrus':105,
    'Right AnG   angular gyrus':106,'Left AnG   angular gyrus':107,
    'Right Calc  calcarine cortex':108,'Left Calc  calcarine cortex':109,
    'Right CO    central operculum':112,'Left CO    central operculum':113,
    'Right Cun   cuneus':114,'Left Cun   cuneus':115,
    'Right Ent   entorhinal area':116,'Left Ent   entorhinal area':117,
    'Right FO    frontal operculum':118,'Left FO    frontal operculum':119,
    'Right FRP   frontal pole':120,'Left FRP   frontal pole':121,
    'Right FuG   fusiform gyrus':122,'Left FuG   fusiform gyrus':123,
    'Right GRe   gyrus rectus':124,'Left GRe   gyrus rectus':125,
    'Right IOG   inferior occipital gyrus':128,'Left IOG   inferior occipital gyrus':129,
    'Right ITG   inferior temporal gyrus':132,'Left ITG   inferior temporal gyrus':133,
    'Right LiG   lingual gyrus':134,'Left LiG   lingual gyrus':135,
    'Right LOrG  lateral orbital gyrus':136,'Left LOrG  lateral orbital gyrus':137,
    'Right MCgG  middle cingulate gyrus':138,'Left MCgG  middle cingulate gyrus':139,
    'Right MFC   medial frontal cortex':140,'Left MFC   medial frontal cortex':141,
    'Right MFG   middle frontal gyrus':142,'Left MFG   middle frontal gyrus':143,
    'Right MOG   middle occipital gyrus':144,'Left MOG   middle occipital gyrus':145,
    'Right MOrG  medial orbital gyrus':146,'Left MOrG  medial orbital gyrus':147,
    'Right MPoG  postcentral gyrus medial segment':148,'Left MPoG  postcentral gyrus medial segment':149,
    'Right MPrG  precentral gyrus medial segment':150,'Left MPrG  precentral gyrus medial segment':151,
    'Right MSFG  superior frontal gyrus medial segment':152,'Left MSFG  superior frontal gyrus medial segment':153,
    'Right MTG   middle temporal gyrus':154,'Left MTG   middle temporal gyrus':155,
    'Right OCP   occipital pole':156,'Left OCP   occipital pole':157,
    'Right OFuG  occipital fusiform gyrus':160,'Left OFuG  occipital fusiform gyrus':161,
    'Right OpIFG opercular part of the inferior frontal gyrus':162,
    'Left OpIFG opercular part of the inferior frontal gyrus':163,
    'Right OrIFG orbital part of the inferior frontal gyrus':164,
    'Left OrIFG orbital part of the inferior frontal gyrus':165,
    'Right PCgG  posterior cingulate gyrus':166,'Left PCgG  posterior cingulate gyrus':167,
    'Right PCu   precuneus':168,'Left PCu   precuneus':169,
    'Right PHG   parahippocampal gyrus':170,'Left PHG   parahippocampal gyrus':171,
    'Right PIns  posterior insula':172,'Left PIns  posterior insula':173,
    'Right PO    parietal operculum':174,'Left PO    parietal operculum':175,
    'Right PoG   postcentral gyrus':176,'Left PoG   postcentral gyrus':177,
    'Right POrG  posterior orbital gyrus':178,'Left POrG  posterior orbital gyrus':179,
    'Right PP    planum polare':180,'Left PP    planum polare':181,
    'Right PrG   precentral gyrus':182,'Left PrG   precentral gyrus':183,
    'Right PT    planum temporale':184,'Left PT    planum temporale':185,
    'Right SCA   subcallosal area':186,'Left SCA   subcallosal area':187,
    'Right SFG   superior frontal gyrus':190,'Left SFG   superior frontal gyrus':191,
    'Right SMC   supplementary motor cortex':192,'Left SMC   supplementary motor cortex':193,
    'Right SMG   supramarginal gyrus':194,'Left SMG   supramarginal gyrus':195,
    'Right SOG   superior occipital gyrus':196,'Left SOG   superior occipital gyrus':197,
    'Right SPL   superior parietal lobule':198,'Left SPL   superior parietal lobule':199,
    'Right STG   superior temporal gyrus':200,'Left STG   superior temporal gyrus':201,
    'Right TMP   temporal pole':202,'Left TMP   temporal pole':203,
    'Right TrIFG triangular part of the inferior frontal gyrus':204,
    'Left TrIFG triangular part of the inferior frontal gyrus':205,
    'Right TTG   transverse temporal gyrus':206,'Left TTG   transverse temporal gyrus':207
}

# sort by ROI name
ROIs = OrderedDict(sorted(ROIs.items(), key=lambda roi: roi[0]))

# Composite ROIs for report spreadsheet
compositeROIs = dict(pvc_ROI_groupings,
                     **{'precuneus':[168,169],
                        'mean cortical':[100,101,104,105,106,107,112,113,118,119,120,121,124,125,
                                         128,129,132,136,137,138,139,140,141,142,143,144,145,146,147,
                                         152,153,154,155,156,157,162,163,164,165,166,167,168,169,
                                         174,175,178,179,180,181,184,185,186,187,190,191,192,193,
                                         194,195,196,197,198,199,200,201,202,203,204,205,206,207]})

# sanity check -- make sure there's no overlap across groups of labels
pvc_all_labels = [label for group in list(pvc_ROI_groupings.values()) for label in group]
assert(len(pvc_all_labels)==len(set(pvc_all_labels)))


# There are two streams of processing we will pursue. First, we generate a conservative reference region definition:
# * `reference_region`: Combines the selected MUSE labels to generate a binary mask.
# * `erode`: To minimize partial voluming effects, we slightly erode the reference region mask using a $5\times5\times5$ mm box kernel.
#
# Second, we generate the set of labels that will be used in partial volume correction. MUSE labels do not include a sulcal CSF label, but this is an important label for PVC. We approximate the sulcal CSF label as the rim around the brain. To this end, we dilate the brain mask, and subtract from it the original brain mask. We designate a label value of $-1$ to this rim, and include it with the ventricle and CSF ROI for PVC.
# * `brainmask`: Threshold the MUSE label image to get a binary brain mask.
# * `dilate`: Dilate the brain mask using a $5\times5\times5$ mm box kernel.
# * `difference`: Subtract dilated brain mask from the orginal mask to get the rim around the brain. This subtraction assigns a value of $-1$ to the rim.
# * `add`: We add the rim image to the MUSE label image. Since the MUSE label image has value $0$ where the rim image has non-zero values, the result is a label image that preserves all the MUSE labels and additionally has a "sulcal CSF" label with value $-1$.
# * `pvc_labels`: We combine the ROIs to generate a collection of binary masks. The result is a 4D volume (with all the binary 3D masks concatenated along 4th dimension). This 4D volume will be an input to the PVC methods.

# placeholder
muselabel = Node(interface=IdentityInterface(fields=['muselabel']), name="muselabel")

reference_region = Node(interface=CombineROIs(ROI_groupings=list(reference_ROI_grouping.values())),
                        name="reference_region")
whole_brain = Node(interface=CombineROIs(ROI_groupings=list(whole_brain_ROI_grouping.values())),
                   name="whole_brain")

brainmask = Node(interface=fsl.ImageMaths(op_string=' -bin', suffix='_brainmask'), name='brainmask')
dilate = Node(interface=fsl.DilateImage(operation='max', kernel_shape='box', kernel_size=4), name='dilate')
difference = Node(interface=fsl.ImageMaths(op_string=' -sub ', suffix='_diff'), name='difference')
add = Node(interface=fsl.ImageMaths(op_string=' -add ', suffix='_add'), name='add')
pvc_labels = Node(interface=CombineROIs(ROI_groupings=list(pvc_ROI_groupings.values())), name="pvc_labels")

labels_qc = Node(interface=labels_snapshots(labelnames=list(pvc_ROI_groupings.keys())), name="labels_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','labels_wf')
datasink.inputs.substitutions = [('_id_',''),
                                 ('flirt','coreg'),
                                 ('_'+'{:d}'.format(len(reference_ROI_grouping))+'combinedROIs','_refRegion'),
                                 ('_'+'{:d}'.format(len(pvc_ROI_groupings))+'combinedROIs','_pvcLabels'),
                                 ('_add',''),
                                 ('_masked','')
                                ]

labels_workflow = Workflow(name="labels_workflow")
labels_workflow.base_dir = os.path.join(output_dir,'labels_workingdir')
labels_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'labels_crashdumps')}}
labels_workflow.connect([(muselabel, reference_region, [('muselabel', 'labelImgFile')]),
                         (muselabel, whole_brain, [('muselabel', 'labelImgFile')]),

                         # Assign a value of -1 to voxels surrounding the brain
                         # this is an approximation for sulcal CSF label
                         (muselabel, brainmask, [('muselabel','in_file')]),
                         (brainmask, dilate, [('out_file','in_file')]),
                         (brainmask, difference, [('out_file','in_file')]),
                         (dilate, difference, [('out_file','in_file2')]),
                         (muselabel, add, [('muselabel','in_file')]),
                         (difference, add,[('out_file','in_file2')]),
                         (add, pvc_labels, [('out_file','labelImgFile')]),

                         (pvc_labels, labels_qc, [('roi4DMaskFile','labelfile')]),

                         (reference_region, datasink, [('roi4DMaskFile','reference_region')]),
                         (whole_brain, datasink, [('roi4DMaskFile','whole_brain')]),
                         (pvc_labels, datasink, [('roi4DMaskFile','pvc_labels')]),
                         (labels_qc, datasink, [('label_snap','QC')])
                        ])

labels_workflow.write_graph('labels.dot', graph2use='colored', simple_form=True)

amyloid_workflow.connect([# Anatomical label manipulation
                      (coreg_workflow, labels_workflow, [('labels_to_pet.out_file','muselabel.muselabel')]),
                     ])

# ## 4. PARTIAL VOLUME CORRECTION
pvc = Node(interface=petpvc.PETPVC(pvc='RBV', fwhm_x=pvc_fwhm_x, fwhm_y=pvc_fwhm_y, fwhm_z=pvc_fwhm_z),
              iterfield=['in_file'], name="pvc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','PVC_wf')
datasink.inputs.substitutions = [('_id_',''),
                                 ('_roi',''),
                                 ('_merged',''),
                                 ('mean','avg20min'),
                                 ('flirt','coreg'),
                                 ('_reoriented',''),
                                 ('_add',''),
                                 ('_0000',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

pvc_workflow = Workflow(name="pvc_workflow")
pvc_workflow.base_dir = os.path.join(output_dir,'pvc_workingdir')
pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'pvc_crashdumps')}}

pvc_workflow.connect([(pvc, datasink, [('out_file','@')])
                     ])

pvc_workflow.write_graph('pvc.dot', graph2use='colored', simple_form=True)

amyloid_workflow.connect([(coreg_workflow, pvc_workflow, [('reorient.out_file','pvc.in_file')]),
                      (labels_workflow, pvc_workflow, [('pvc_labels.roi4DMaskFile', 'pvc.mask_file')])
                     ])


# ## 5a. SUVR IMAGE
# placeholder
pet = Node(interface=IdentityInterface(fields=['pet']), name="pet")

ROImean = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean") # note that this is not a trimmed mean!
SUVR = Node(interface=fsl.ImageMaths(), name="SUVR")

ROImeans = JoinNode(interface=ROI_stats_to_spreadsheet(ROI_list=list(ROIs.values()),
                                             ROI_names=list(ROIs.keys()),
                                             additionalROIs=list(compositeROIs.values()),
                                             additionalROI_names=list(compositeROIs.keys()),
                                             xlsxFile=os.path.join(output_dir,'SUVR_40to70min_ROI.xlsx'),
                                             stat='mean',
                                             proportiontocut=proportiontocut),
                    joinsource="infosource", joinfield=['imgFileList','labelImgFileList'], name="ROImeans")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','SUVR_wf')
datasink.inputs.substitutions = [('_id_',''),
                                 ('_merged',''),
                                 ('_flirt','_coreg'),
                                 ('_mean','_avg40to70min'),
                                 ('_maths','_suvr'),
                                 ('_reoriented',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

SUVR_workflow = Workflow(name="SUVR_workflow")
SUVR_workflow.base_dir = os.path.join(output_dir,'SUVR_workingdir')
SUVR_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUVR_crashdumps')}}
SUVR_workflow.connect([(pet, ROImean, [('pet','in_file')]),
                       (pet, SUVR, [('pet','in_file')]),
                       (ROImean, SUVR, [(('out_stat',to_div_string),'op_string')]),
                       (SUVR, ROImeans, [('out_file','imgFileList')]),

                       (SUVR, datasink, [('out_file','SUVR_40to70min')])
                      ])

SUVR_workflow.write_graph('SUVR.dot', graph2use='colored', simple_form=True)

amyloid_workflow.connect([# SUVR computation
                      (coreg_workflow, SUVR_workflow, [('reorient.out_file', 'pet.pet')]),
                      (labels_workflow, SUVR_workflow, [('reference_region.roi4DMaskFile','ROImean.mask_file'),
                                                        ('muselabel.muselabel','ROImeans.labelImgFileList')])
                     ])


# ## 5b. SUVR IMAGE WITH PVC
# placeholder
pet = Node(interface=IdentityInterface(fields=['pet']), name="pet")

ROImean_pvc = Node(interface=fsl.ImageStats(op_string=' -k %s -m '), name="ROImean_pvc") # note that this is not a trimmed mean!
SUVR_pvc = Node(interface=fsl.ImageMaths(), name="SUVR_pvc")

ROImeans_pvc = JoinNode(interface=ROI_stats_to_spreadsheet(ROI_list=list(ROIs.values()),
                                             ROI_names=list(ROIs.keys()),
                                             additionalROIs=list(compositeROIs.values()),
                                             additionalROI_names=list(compositeROIs.keys()),
                                             xlsxFile=os.path.join(output_dir,'SUVR_40to70min_pvc_ROI.xlsx'),
                                             stat='mean',
                                             proportiontocut=proportiontocut),
                    joinsource="infosource", joinfield=['imgFileList','labelImgFileList'], name="ROImeans_pvc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','SUVR_wf')
datasink.inputs.substitutions = [('_id_',''),
                                 ('_merged',''),
                                 ('_flirt','_coreg'),
                                 ('_mean','_avg40to70min'),
                                 ('_maths','_suvr'),
                                 ('_reoriented',''),
                                 ('_0000',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r'')]

SUVR_pvc_workflow = Workflow(name="SUVR_pvc_workflow")
SUVR_pvc_workflow.base_dir = os.path.join(output_dir,'SUVR_workingdir')
SUVR_pvc_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'SUVR_crashdumps')}}
SUVR_pvc_workflow.connect([(pet, ROImean_pvc, [('pet','in_file')]),
                       (pet, SUVR_pvc, [('pet','in_file')]),
                       (ROImean_pvc, SUVR_pvc, [(('out_stat',to_div_string),'op_string')]),
                       (SUVR_pvc, ROImeans_pvc, [('out_file','imgFileList')]),

                       (SUVR_pvc, datasink, [('out_file','SUVR_40to70min_pvc')])
                      ])

SUVR_pvc_workflow.write_graph('SUVR_pvc.dot', graph2use='colored', simple_form=True)

amyloid_workflow.connect([# SUVR computation with PVC
                      (pvc_workflow, SUVR_pvc_workflow, [('pvc.out_file', 'pet.pet')]),
                      (labels_workflow, SUVR_pvc_workflow, [('reference_region.roi4DMaskFile','ROImean_pvc.mask_file'),
                                                            ('muselabel.muselabel','ROImeans_pvc.labelImgFileList')])
                     ])


# ## 6. MNI SPACE

template = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')

# placeholders
mri = Node(interface=IdentityInterface(fields=['mri']), name="mri")
pet = Node(interface=IdentityInterface(fields=['pet']), name="pet")

# Very quick registration to MNI template
mri_to_mni = Node(interface=fsl.FLIRT(dof=12,reference=template), name="mri_to_mni")

mergexfm = Node(interface=fsl.ConvertXFM(concat_xfm=True), name="mergexfm")

warp_pet = Node(interface=fsl.ApplyXFM(apply_xfm=True, reference=template), name='warp_pet')

warp_suvr = warp_pet.clone(name='warp_suvr')
warp_suvr_pvc = warp_pet.clone(name='warp_suvr_pvc')

# Gaussian smoothing
smooth_suvr = Node(interface=fsl.Smooth(fwhm=smooth_fwhm), name="smooth_suvr")
smooth_suvr_pvc = Node(interface=fsl.Smooth(fwhm=smooth_fwhm), name="smooth_suvr_pvc")

# Triplanar snapshots
suvr_qc = Node(interface=triplanar_snapshots(alpha=.5, x=81, y=93, z=77, vmin=0.0, vmax=4.0), name="suvr_qc")
suvr_pvc_qc = suvr_qc.clone(name="suvr_pvc_qc")

datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output','MNI_wf')
datasink.inputs.substitutions = [('_id_',''),
                                 ('_merged',''),
                                 ('_reoriented',''),
                                 ('_trans','_mni'),
                                 ('flirt','mni'),
                                 ('_0000',''),
                                 ('_masked','')
                                ]
datasink.inputs.regexp_substitutions = [(r'_\d+\.\d+to\d+\.\d+min',r''),
                                        (r'_\d+\.\d+min',r'')]

MNI_workflow = Workflow(name="MNI_workflow")
MNI_workflow.base_dir = os.path.join(output_dir,'MNI_workingdir')
MNI_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'MNI_crashdumps')}}
MNI_workflow.connect([(mri, mri_to_mni, [('mri','in_file')]),

                      (mri_to_mni, mergexfm, [('out_matrix_file','in_file2')]),

                      (pet, warp_pet, [('pet','in_file')]),
                      (mergexfm, warp_pet, [('out_file', 'in_matrix_file')]),

                      (mergexfm, warp_suvr, [('out_file','in_matrix_file')]),
                      (mergexfm, warp_suvr_pvc, [('out_file','in_matrix_file')]),

                      (warp_suvr, smooth_suvr, [('out_file','in_file')]),
                      (warp_suvr_pvc, smooth_suvr_pvc, [('out_file','in_file')]),

                      (mri_to_mni, suvr_qc, [('out_file','bgimgfile')]),
                      (mri_to_mni, suvr_pvc_qc, [('out_file','bgimgfile')]),

                      (smooth_suvr, suvr_qc, [('smoothed_file','imgfile')]),
                      (smooth_suvr_pvc, suvr_pvc_qc, [('smoothed_file','imgfile')]),

                      (mri_to_mni, datasink, [('out_file','warped_mri'),
                                              ('out_matrix_file','warped_mri.@param')]),
                      (warp_pet, datasink, [('out_file','warped_pet')]),

                      (warp_suvr, datasink, [('out_file','warped_suvr')]),
                      (smooth_suvr, datasink, [('smoothed_file','warped_suvr.@smooth')]),

                      (warp_suvr_pvc, datasink, [('out_file','warped_suvr_pvc')]),
                      (smooth_suvr_pvc, datasink, [('smoothed_file','warped_suvr_pvc.@smooth')]),

                      (suvr_qc, datasink, [('triplanar','QC@SUVR')]),
                      (suvr_pvc_qc, datasink, [('triplanar','QC@SUVR_PVC')]),
                     ])

MNI_workflow.write_graph('MNI.dot', graph2use='colored', simple_form=True)

amyloid_workflow.connect([# MNI space normalization
                      (coreg_workflow, MNI_workflow, [('reorientmri.out_file','mri.mri'),
                                                      ('reorient.out_file','pet.pet'),
                                                      ('pet_to_mri.out_matrix_file','mergexfm.in_file')]),
                      (SUVR_workflow, MNI_workflow, [('SUVR.out_file','warp_suvr.in_file')]),
                      (SUVR_pvc_workflow, MNI_workflow, [('SUVR_pvc.out_file','warp_suvr_pvc.in_file')]),
                     ])

result = amyloid_workflow.run('MultiProc', plugin_args={'n_procs': n_procs, 'status_callback': log_nodes_cb})
