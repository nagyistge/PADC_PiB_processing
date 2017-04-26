from nipype.interfaces.base import TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec, CommandLineInputSpec, CommandLine, isdefined
from nipype.utils.filemanip import split_filename

import nibabel as nib
import numpy as np
import os
import pandas as pd

def get_value(key, dict):
    return dict[key]

def to_div_string(val):
    return ' -div ' + str(val)


class SplitTimeSeriesInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='4D image file to be split', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    splitTime = traits.Float(desc='minute into the time series image at which to split the 4D image', mandatory=True)

class SplitTimeSeriesOutputSpec(TraitedSpec):
    firstImgFile = File(exists=True, desc='first of the two split images (up to but not including splitTime)')
    secondImgFile = File(exists=True, desc='second of the two split images (including splitTime and beyond)')
    firstImgStart = traits.Float(desc='start time in minutes for the first split, inclusive')
    firstImgEnd = traits.Float(desc='end time in minutes for the first split, exclusive')
    #splitTime = traits.Float(desc='possibly modified split time = end time in minutes for the first split, exclusive; also start time in minutes for the second split, inclusive')
    secondImgStart = traits.Float(desc='start time in minutes for the second split, inclusive')
    secondImgEnd = traits.Float(desc='end time in minutes for the second split, exclusive')

class SplitTimeSeries(BaseInterface):
    """
    Split a 4D (time series/dynamic) image into two 4D images

    """

    input_spec = SplitTimeSeriesInputSpec
    output_spec = SplitTimeSeriesOutputSpec

    firstImgStart = -1
    #modSplitTime = -1
    firstImgEnd = -1
    secondImgStart = -1
    secondImgEnd = -1

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        splitTime = self.inputs.splitTime

        _, base, _ = split_filename(timeSeriesImgFile)

        frameTiming = pd.read_csv(frameTimingCsvFile)
        # check that frameTiming has columns named frameStart and frameEnd
        for col in ['frameStart','frameEnd']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = frameTiming['frameStart']
        frameEnd = frameTiming['frameEnd']
        # sanity checks on frameTiming spreadsheet entries
        assert(all(frameStart<frameEnd))
        assert(all(frameStart[1:]==frameEnd[:len(frameEnd)-1]))

        frameStart = frameStart.tolist()
        frameEnd = frameEnd.tolist()

        if splitTime<frameStart[0]:
            splitTime = frameStart[0]
        elif splitTime>frameEnd[-1]:
            splitTime = frameEnd[-1]

        splitIndex = next((i for i,t in enumerate(frameStart) if t>=splitTime), len(frameTiming))

        #modSplitTime = frameStart[splitIndex]
        #self.modSplitTime = modSplitTime

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()

        # number of elements in the frameTimingTxtFile must be the same as the number of time frames in the PiB image
        assert(len(frameTiming)==img_dat.shape[3])

        # compute the first image
        if splitIndex>0:
            self.firstImgStart = frameStart[0]
            self.firstImgEnd = frameEnd[splitIndex-1]
            firstImg_dat = img_dat[:,:,:,:splitIndex]
            firstImg = nib.Nifti1Image(firstImg_dat, img.affine, img.header)
            firstImgFile = base+'_'+'{:02.2f}'.format(self.firstImgStart)+'to'+'{:02.2f}'.format(self.firstImgEnd)+'min.nii.gz'
            #firstImgFile = base+'_'+str(self.firstImgStart)+'to'+str(self.modSplitTime)+'min.nii'
            nib.save(firstImg,firstImgFile)
        else:
            firstImgFile = None

        # compute the second image
        if splitIndex<img_dat.shape[3]:
            self.secondImgStart = frameStart[splitIndex]
            self.secondImgEnd = frameEnd[-1]
            secondImg_dat = img_dat[:,:,:,splitIndex:]
            secondImg = nib.Nifti1Image(secondImg_dat, img.affine, img.header)
            secondImgFile = base+'_'+'{:02.2f}'.format(self.secondImgStart)+'to'+'{:02.2f}'.format(self.secondImgEnd)+'min.nii.gz'
            #secondImgFile = base+'_'+str(self.modSplitTime)+'to'+str(self.secondImgEnd)+'min.nii'
            nib.save(secondImg,secondImgFile)
        else:
            secondImgFile = None

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['firstImgStart'] = self.firstImgStart
        #outputs['splitTime'] = self.modSplitTime
        outputs['firstImgEnd'] = self.firstImgEnd
        outputs['secondImgStart'] = self.secondImgStart
        outputs['secondImgEnd'] = self.secondImgEnd
        outputs['firstImgFile'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.firstImgStart)+'to'+'{:02.2f}'.format(self.firstImgEnd)+'min.nii.gz')
        outputs['secondImgFile'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.secondImgStart)+'to'+'{:02.2f}'.format(self.secondImgEnd)+'min.nii.gz')
        #outputs['firstImgFile'] = os.path.abspath(base+'_'+str(self.firstImgStart)+'to'+str(self.modSplitTime)+'min.nii')
        #outputs['secondImgFile'] = os.path.abspath(base+'_'+str(self.modSplitTime)+'to'+str(self.secondImgEnd)+'min.nii')

        return outputs


class ExtractTimeInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='4D image file from which to extract', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    startTime = traits.Float(desc='minute into the time series image at which to begin, inclusive', mandatory=True)
    endTime = traits.Float(desc='minute into the time series image at which to stop, exclusive', mandatory=True)

class ExtractTimeOutputSpec(TraitedSpec):
    extractImgFile = File(exists=True, desc='extracted 4D image between the specified start and end times')
    startTime = traits.Float(desc='possibly modified start time')
    endTime = traits.Float(desc='possibly modified end time')

class ExtractTime(BaseInterface):
    """
    Extract a 4D (time series/dynamic) image from a larger 4D image

    """

    input_spec = ExtractTimeInputSpec
    output_spec = ExtractTimeOutputSpec

    modStartTime = -1
    modEndTime = -1

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        startTime = self.inputs.startTime
        endTime = self.inputs.endTime

        _, base, _ = split_filename(timeSeriesImgFile)

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

        frameStart = frameStart.tolist()
        frameEnd = frameEnd.tolist()

        if startTime < frameStart[0]:
            startTime = frameStart[0]
        elif startTime > frameEnd[-1]:
            sys.exit('Start time is beyond the time covered by the time series data!')

        if endTime > frameEnd[-1]:
            endTime = frameEnd[-1]
        elif endTime < frameStart[0]:
            sys.exit('End time is prior to the time covered by the time series data!')

        # make sure that start time is earlier than end time
        assert(startTime<endTime)

        # find the first time frame with frameStart at or shortest after the specified start time (this frame will be included in the average)
        startIndex = next((i for i,t in enumerate(frameStart) if t>=startTime), len(frameTiming)-1)
        # find the first time frame with frameEnd shortest after the specified end time (this frame will not be included in the average)
        endIndex = next((i for i,t in enumerate(frameEnd) if t>endTime), len(frameTiming))

        # another sanity check, mainly to make sure that startIndex!=endIndex
        assert(startIndex<endIndex)

        # get the actual start and end times for the 3D mean image to be computed
        self.modStartTime = frameStart[startIndex]
        self.modEndTime = frameEnd[endIndex-1]

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()

        # number of elements in the frameTimingTxtFile must be the same as the number of time frames in the PiB image
        assert(len(frameTiming)==img_dat.shape[3])

        # extract image
        extractImg_dat = img_dat[:,:,:,startIndex:endIndex]
        extractImg = nib.Nifti1Image(extractImg_dat, img.affine, img.header)
        extractImgFile = base+'_'+'{:02.2f}'.format(self.modStartTime)+'to'+'{:02.2f}'.format(self.modEndTime)+'min.nii'
        nib.save(extractImg,extractImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['startTime'] = self.modStartTime
        outputs['endTime'] = self.modEndTime
        outputs['extractImgFile'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modStartTime)+'to'+'{:02.2f}'.format(self.modEndTime)+'min.nii')

        return outputs



class DynamicMeanInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='4D image file to average temporally', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    startTime = traits.Float(desc='minute into the time series image at which to begin computing the mean image, inclusive', mandatory=True)
    endTime = traits.Float(desc='minute into the time series image at which to stop computing the mean image, exclusive', mandatory=True)

class DynamicMeanOutputSpec(TraitedSpec):
    meanImgFile = File(exists=True, desc='3D mean of the 4D image between the specified start and end times')
    startTime = traits.Float(desc='possibly modified start time')
    endTime = traits.Float(desc='possibly modified end time')

class DynamicMean(BaseInterface):
    """
    Compute the 3D mean of a 4D (time series/dynamic) image

    """

    input_spec = DynamicMeanInputSpec
    output_spec = DynamicMeanOutputSpec

    modStartTime = -1
    modEndTime = -1

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        startTime = self.inputs.startTime
        endTime = self.inputs.endTime

        _, base, _ = split_filename(timeSeriesImgFile)

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

        frameStart = frameStart.tolist()
        frameEnd = frameEnd.tolist()

        if startTime < frameStart[0]:
            startTime = frameStart[0]
        elif startTime > frameEnd[-1]:
            sys.exit('Start time is beyond the time covered by the time series data!')

        if endTime > frameEnd[-1]:
            endTime = frameEnd[-1]
        elif endTime < frameStart[0]:
            sys.exit('End time is prior to the time covered by the time series data!')

        # make sure that start time is earlier than end time
        assert(startTime<endTime)

        # find the first time frame with frameStart at or shortest after the specified start time (this frame will be included in the average)
        startIndex = next((i for i,t in enumerate(frameStart) if t>=startTime), len(frameTiming)-1)
        # find the first time frame with frameEnd shortest after the specified end time (this frame will not be included in the average)
        endIndex = next((i for i,t in enumerate(frameEnd) if t>endTime), len(frameTiming))

        # another sanity check, mainly to make sure that startIndex!=endIndex
        assert(startIndex<endIndex)

        # get the actual start and end times for the 3D mean image to be computed
        self.modStartTime = frameStart[startIndex]
        self.modEndTime = frameEnd[endIndex-1]

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()

        # number of elements in the frameTimingTxtFile must be the same as the number of time frames in the PiB image
        assert(len(frameTiming)==img_dat.shape[3])

        # compute the mean image
        meanImg_dat = np.mean(img_dat[:,:,:,startIndex:endIndex], axis=3)
        meanImg = nib.Nifti1Image(meanImg_dat, img.affine, img.header)
        meanImgFile = base+'_'+'{:02.2f}'.format(self.modStartTime)+'to'+'{:02.2f}'.format(self.modEndTime)+'min_mean.nii.gz'
        #meanImgFile = base+'_'+str(self.modStartTime)+'to'+str(self.modEndTime)+'min_mean.nii'
        nib.save(meanImg,meanImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['startTime'] = self.modStartTime
        outputs['endTime'] = self.modEndTime
        outputs['meanImgFile'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modStartTime)+'to'+'{:02.2f}'.format(self.modEndTime)+'min_mean.nii.gz')
        #outputs['meanImgFile'] = os.path.abspath(base+'_'+str(self.modStartTime)+'to'+str(self.modEndTime)+'min_mean.nii')

        return outputs


class CombineROIsInputSpec(BaseInterfaceInputSpec):
    labelImgFile = File(exists=True, desc='Label image file containing ROIs to be combined', mandatory=True)
    ROI_groupings = traits.List(desc='list of lists of integers', mandatory=True)

class CombineROIsOutputSpec(TraitedSpec):
    roi4DMaskFile = File(exists=True, desc='4D image volume, each corresponding to a combined ROI')

class CombineROIs(BaseInterface):
    """
    Combine multiple ROIs and write resulting mask to image.
    If multiple ROI combinations are provided, the result will be a 4D mask image,
    with each 3D image representing a separate ROI combination mask.

    """

    input_spec = CombineROIsInputSpec
    output_spec = CombineROIsOutputSpec

    def _run_interface(self, runtime):
        labelImgFile = self.inputs.labelImgFile
        ROI_groupings = self.inputs.ROI_groupings

        _, base, _ = split_filename(labelImgFile)

        labelimage = nib.load(labelImgFile)
        labelimage_dat = labelimage.get_data()

        ROI4Dmask_shape = list(labelimage_dat.shape)
        ROI4Dmask_shape.append(len(ROI_groupings))
        ROI4Dmask_dat = np.zeros(ROI4Dmask_shape)

        for n, ROI_grouping in enumerate(ROI_groupings):
            ROI_mask = labelimage_dat==ROI_grouping[0]
            if len(ROI_grouping)>1:
                for ROI in ROI_grouping[1:]:
                    ROI_mask = ROI_mask | (labelimage_dat==ROI)
            ROI4Dmask_dat[:,:,:,n] = ROI_mask

        # Save 4D mask
        ROI4Dmask = nib.Nifti1Image(ROI4Dmask_dat, labelimage.affine, labelimage.header)
        ROI4Dmaskfile = base+'_'+'{:d}'.format(len(ROI_groupings))+'combinedROIs.nii.gz'
        nib.save(ROI4Dmask,ROI4Dmaskfile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.labelImgFile
        _, base, _ = split_filename(fname)

        outputs['roi4DMaskFile'] = os.path.abspath(base+'_'+'{:d}'.format(len(self.inputs.ROI_groupings))+'combinedROIs.nii.gz')

        return outputs


class Pad4DImageInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='Dynamic image to be padded', mandatory=True)
    padsize = traits.Int(desc='Each time frame will be padded on each size by padsize', mandatory=True)

class Pad4DImageOutputSpec(TraitedSpec):
    paddedImgFile = File(exists=True, desc='Padded dynamic image')

class Pad4DImage(BaseInterface):
    """
    Pad each timeframe on each of the 6 sides (top, bottom, left, right, front, back) with the nearest slice
    """

    input_spec = Pad4DImageInputSpec
    output_spec = Pad4DImageOutputSpec

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        padsize = self.inputs.padsize
        _, base, _ = split_filename(timeSeriesImgFile)

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()
        [rows,cols,slices,comps] = img_dat.shape

        padded_dat = np.zeros((rows+2*padsize,cols+2*padsize,slices+2*padsize,comps))

        for l in range(comps):
            padded_dat[padsize:(rows+padsize),padsize:(cols+padsize),padsize:(slices+padsize),l] = img_dat[:,:,:,l]
            for x in range(padsize):
                padded_dat[x,:,:,l] = padded_dat[padsize,:,:,l]
                padded_dat[-(x+1),:,:,l] = padded_dat[-(padsize+1),:,:,l]
            for y in range(padsize):
                padded_dat[:,x,:,l] = padded_dat[:,padsize,:,l]
                padded_dat[:,-(x+1),:,l] = padded_dat[:,-(padsize+1),:,l]
            for z in range(padsize):
                padded_dat[:,:,x,l] = padded_dat[:,:,padsize,l]
                padded_dat[:,:,-(x+1),l] = padded_dat[:,:,-(padsize+1),l]

        # Save results
        paddedImg = nib.Nifti1Image(padded_dat, img.affine)
        paddedImgFile = base+'_padded.nii'
        nib.save(paddedImg,paddedImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['paddedImgFile'] = os.path.abspath(base+'_padded.nii')

        return outputs

class Unpad4DImageInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='Dynamic image unpad', mandatory=True)
    padsize = traits.Int(desc='Each time frame will be unpadded on each size by padsize', mandatory=True)

class Unpad4DImageOutputSpec(TraitedSpec):
    unpaddedImgFile = File(exists=True, desc='Unpadded dynamic image')

class Unpad4DImage(BaseInterface):
    """
    Pad each timeframe on each of the 6 sides (top, bottom, left, right, front, back) with the nearest slice
    """

    input_spec = Unpad4DImageInputSpec
    output_spec = Unpad4DImageOutputSpec

    def _run_interface(self, runtime):
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        padsize = self.inputs.padsize
        _, base, _ = split_filename(timeSeriesImgFile)

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()
        [rows,cols,slices,comps] = img_dat.shape

        unpadded_dat = np.zeros((rows-2*padsize,cols-2*padsize,slices-2*padsize,comps))

        for l in range(comps):
            unpadded_dat[:,:,:,l] = img_dat[padsize:(rows-padsize),padsize:(cols-padsize),padsize:(slices-padsize),l]

        # Save results
        unpaddedImg = nib.Nifti1Image(unpadded_dat, img.affine)
        unpaddedImgFile = base+'_unpadded.nii.gz'
        nib.save(unpaddedImg,unpaddedImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['unpaddedImgFile'] = os.path.abspath(base+'_unpadded.nii.gz')

        return outputs

class SRTMInputSpec(BaseInterfaceInputSpec):
    timeSeriesImgFile = File(exists=True, desc='Dynamic PET image', mandatory=True)
    refRegionMaskFile = File(exists=True, desc='Reference region mask', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    endTime = traits.Float(desc='minute into the time series image at which to stop computing the parametric images, exclusive', mandatory=True)
    proportiontocut = traits.Float(0.05,desc='proportion to cut from each tail of the distribution before computing the mean signal in the reference region')
    fwhm = traits.Float(desc='Full width at half max (in mm) for Gaussian smoothing',mandatory=True)

class SRTMOutputSpec(TraitedSpec):
    endTime = traits.Float(desc='possibly modified end time')
    DVRImgFile_wlr = File(exists=True, desc='DVR image computed using weighted linear regression (wlr)')
    R1ImgFile_wlr = File(exists=True, desc='R1 image computed using weighted linear regression (wlr)')
    DVRImgFile_lrsc = File(exists=True, desc='DVR image computed using linear regression with spatial constraint (lrsc)')
    R1ImgFile_lrsc = File(exists=True, desc='R1 image computed using linear regression with spatial constraint (lrsc)')

class SRTM(BaseInterface):
    """
    Simplified Reference Tissue Model (SRTM)

    """

    input_spec = SRTMInputSpec
    output_spec = SRTMOutputSpec

    modEndTime = -1

    def _run_interface(self, runtime):
        import math
        import numpy.matlib as mat
        import scipy as sp
        import statsmodels.api as sm
        from tqdm import tqdm

        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        refRegionMaskFile = self.inputs.refRegionMaskFile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        endTime = self.inputs.endTime
        proportiontocut = self.inputs.proportiontocut
        fwhm = self.inputs.fwhm

        _, base, _ = split_filename(timeSeriesImgFile)

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

        if endTime > frameEnd[-1]:
            endTime = frameEnd[-1]
        elif endTime < frameStart[0]:
            sys.exit('End time is prior to the time covered by the time series data!')

        # find the first time frame with frameEnd shortest after the specified end time (this frame will not be included in the average)
        endIndex = next((i for i,t in enumerate(frameEnd) if t>endTime), len(frameTiming))

        # get the actual end time for the 4D image to be used
        self.modEndTime = frameEnd[endIndex-1]

        # Compute the time mid-way for each time frame
        t = (frameStart[0:endIndex] + frameEnd[0:endIndex])/2
        # Compute the duration of each time frame
        delta = frameEnd[0:endIndex] - frameStart[0:endIndex]

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()[:,:,:,0:endIndex]
        [rows,cols,slices,comps] = img_dat.shape
        voxSize = img.header.get_zooms()[0:3]
        sigma_mm = fwhm / (2*math.sqrt(2*math.log(2)))
        sigma = [sigma_mm / v for v in voxSize]

        mip = np.amax(img_dat,axis=3)
        mask = mip>=1 # don't process voxels that don't have at least one count
        numVox = np.sum(mask)

        # load reference region mask
        ref = nib.load(refRegionMaskFile)
        ref_dat = ref.get_data().astype(bool)

        # make sure that the reference region mask is in alignment with PET
        Cref = np.zeros(comps) # Time activity curve (TAC) of reference region
        Ctfilt = np.zeros(img_dat.shape) # Spatially smoothed image
        for l in range(comps):
            timeframe = img_dat[:,:,:,l]
            tmp = timeframe[ref_dat]
            Cref[l] = sp.stats.trim_mean(tmp[np.isfinite(tmp)], proportiontocut)
            Ctfilt[:,:,:,l] = sp.ndimage.gaussian_filter(img_dat[:,:,:,l],sigma=sigma,order=0)

        # Integrals etc.
        # Numerical integration of TAC
        intCref = sp.integrate.cumtrapz(Cref,t,initial=0)
        # Numerical integration of Ct
        intCt_ = sp.integrate.cumtrapz(img_dat,t,axis=3,initial=0)

        # STEP 1: weighted linear regression (wlr) [Zhou 2003 p. 978]
        m = 3
        W = mat.diag(delta)
        B0_wlr = np.zeros((rows,cols,slices,m))
        var_B0_wlr = np.zeros((rows,cols,slices))
        B1_wlr = np.zeros((rows,cols,slices,m))
        var_B1_wlr = np.zeros((rows,cols,slices))
        for i in tqdm(range(rows)):
            for j in range(cols):
                for k in range(slices):
                    if mask[i,j,k]:
                        Ct = img_dat[i,j,k,:]
                        #Ct_smoothed = Ctfilt[i,j,k,:]
                        intCt = intCt_[i,j,k,:]

                        #X = np.mat(np.column_stack((intCref, Cref, -Ct_smoothed)))
                        X = np.mat(np.column_stack((intCref, Cref, -Ct)))
                        y = np.mat(intCt).T
                        b0 = sp.linalg.solve(X.T * W * X, X.T * W * y)
                        #b0 = np.linalg.solve(np.matmul(np.matmul(np.transpose(X),W),X) ,
                        #                     np.matmul(np.matmul(np.transpose(X),W),y) )
                        #mod_wls = sm.WLS(intCt, X, weights = delta)
                        #res_wls = mod_wls.fit()
                        #B0_wlr[i,j,k,:] = res_wls.params
                        #var_B0_wlr[i,j,k] = res_wls.scale
                        #residual = intCt - np.matmul(X,b0)
                        #var_b0 = np.matmul(np.matmul(np.transpose(residual), W), residual) / (comps-m)
                        residual = y - X * b0
                        var_b0 = residual.T * W * residual / (comps-m)
                        B0_wlr[i,j,k,:] = b0.T
                        var_B0_wlr[i,j,k] = var_b0

                        XR1 = np.mat(np.column_stack((Cref, intCref, -intCt)))
                        yR1 = np.mat(Ct).T
                        b1 = sp.linalg.solve(XR1.T * W * XR1, XR1.T * W * yR1)
                        #mod_wls_R1 = sm.WLS(Ct, XR1, weights = delta)
                        #res_wls_R1 = mod_wls_R1.fit()
                        #B1_wlr[i,j,k,:] = res_wls_R1.params
                        #var_B1_wlr[i,j,k] = res_wls_R1.scale
                        residual = yR1 - XR1 * b1
                        var_b1 = residual.T * W * residual / (comps-m)
                        B1_wlr[i,j,k,:] = b1.T
                        var_B1_wlr[i,j,k] = var_b1

        dvr_wlr = B0_wlr[:,:,:,0]
        r1_wlr = B1_wlr[:,:,:,0]

        # Preparation for Step 2
        # Apply spatially smooth initial parameter estimates
        B0_sc = np.zeros(B0_wlr.shape)
        B1_sc = np.zeros(B1_wlr.shape)
        for l in range(m):
            B0_sc[:,:,:,l] = sp.ndimage.gaussian_filter(B0_wlr[:,:,:,l],sigma=sigma,order=0)
            B1_sc[:,:,:,l] = sp.ndimage.gaussian_filter(B1_wlr[:,:,:,l],sigma=sigma,order=0)

        H0 = np.zeros(B0_wlr.shape)
        H1 = np.zeros(B1_wlr.shape)
        for i in tqdm(range(rows)):
            for j in range(cols):
                for k in range(slices):
                    if mask[i,j,k]:
                        H0[i,j,k,:] = m * var_B0_wlr[i,j,k] / np.square(B0_wlr[i,j,k,:] - B0_sc[i,j,k,:])
                        H1[i,j,k,:] = m * var_B1_wlr[i,j,k] / np.square(B1_wlr[i,j,k,:] - B1_sc[i,j,k,:])

        # Apply spatial smoothing to H0 and H1
        HH0 = np.zeros(H0.shape)
        HH1 = np.zeros(H1.shape)
        for l in range(m):
            HH0[:,:,:,l] = sp.ndimage.gaussian_filter(H0[:,:,:,l],sigma=sigma,order=0)
            HH1[:,:,:,l] = sp.ndimage.gaussian_filter(H1[:,:,:,l],sigma=sigma,order=0)

        # STEP 2: ridge regression [Zhou 2003 p. 978]
        B0_lrsc = np.zeros((rows,cols,slices,m))
        B1_lrsc = np.zeros((rows,cols,slices,m))
        for i in tqdm(range(rows)):
            for j in range(cols):
                for k in range(slices):
                    if mask[i,j,k]:
                        Ct = img_dat[i,j,k,:]
                        Ct_smoothed = Ctfilt[i,j,k,:]
                        intCt = intCt_[i,j,k,:]

                        X = np.mat(np.column_stack((intCref, Cref, -Ct_smoothed)))
                        y = np.mat(intCt).T
                        b0 = sp.linalg.solve(X.T * W * X + mat.diag(HH0[i,j,k,:]),
                                             X.T * W * y + mat.diag(HH0[i,j,k,:]) * np.mat(B0_sc[i,j,k,:]).T)
                        B0_lrsc[i,j,k,:] = b0.T

                        XR1 = np.mat(np.column_stack((Cref, intCref, -intCt)))
                        yR1 = np.mat(Ct).T
                        b1 = sp.linalg.solve(XR1.T * W * XR1 + mat.diag(HH1[i,j,k,:]),
                                             XR1.T * W * yR1 + mat.diag(HH1[i,j,k,:]) * np.mat(B1_sc[i,j,k,:]).T)
                        #(np.matmul(np.matmul(np.transpose(XR1),W),XR1) + np.diag(HH1[i,j,k,:]),
                        #                     np.matmul(np.matmul(np.transpose(XR1),W),yR1) + np.matmul(np.diag(HH1[i,j,k,:]),B1_sc[i,j,k,:]))
                        B1_lrsc[i,j,k,:] = b1.T

        dvr_lrsc = B0_lrsc[:,:,:,0]
        r1_lrsc = B1_lrsc[:,:,:,0]

        # Save results
        DVRImg_wlr = nib.Nifti1Image(dvr_wlr, img.affine, img.header)
        DVRImgFile_wlr = base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_DVR_wlr.nii.gz'
        R1Img_wlr = nib.Nifti1Image(r1_wlr, img.affine, img.header)
        R1ImgFile_wlr = base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_R1_wlr.nii.gz'
        DVRImg_lrsc = nib.Nifti1Image(dvr_lrsc, img.affine, img.header)
        DVRImgFile_lrsc = base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_DVR_lrsc.nii.gz'
        R1Img_lrsc = nib.Nifti1Image(r1_lrsc, img.affine, img.header)
        R1ImgFile_lrsc = base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_R1_lrsc.nii.gz'

        nib.save(DVRImg_wlr,DVRImgFile_wlr)
        nib.save(R1Img_wlr,R1ImgFile_wlr)
        nib.save(DVRImg_lrsc,DVRImgFile_lrsc)
        nib.save(R1Img_lrsc,R1ImgFile_lrsc)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        outputs['endTime'] = self.modEndTime
        outputs['DVRImgFile_wlr'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_DVR_wlr.nii.gz')
        outputs['R1ImgFile_wlr'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_R1_wlr.nii.gz')
        outputs['DVRImgFile_lrsc'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_DVR_lrsc.nii.gz')
        outputs['R1ImgFile_lrsc'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_R1_lrsc.nii.gz')

        return outputs

class ROI_means_to_spreadsheetInputSpec(BaseInterfaceInputSpec):
    imgFileList = traits.List(desc='Image list', mandatory=True)
    labelImgFileList = traits.List(desc='Label image list', mandatory=True)
    xlsxFile = File(exists=False, desc='xlsx file. If it doesn''t exist, it will be created', mandatory=True)
    ROI_list = traits.List(desc='list of ROI indices for which means will be computed (should match the label indices in the label image)', mandatory=True)
    ROI_names = traits.List(desc='list of equal size to ROI_list that lists the corresponding ROI names', mandatory=True)
    additionalROIs = traits.List(desc='list of lists of integers')
    additionalROI_names = traits.List(desc='names corresponding to additional ROIs')
    proportiontocut = traits.Float(0.05,desc='proportion to cut from each tail of the distribution before computing the mean signal in the reference region')

class ROI_means_to_spreadsheetOutputSpec(TraitedSpec):
    xlsxFile = File(exists=True, desc='xlsx file')

class ROI_means_to_spreadsheet(BaseInterface):
    """
    Compute ROI means and write to spreadsheet

    """

    input_spec = ROI_means_to_spreadsheetInputSpec
    output_spec = ROI_means_to_spreadsheetOutputSpec

    def _run_interface(self, runtime):
        from scipy import stats
        import xlsxwriter

        imgFileList = self.inputs.imgFileList
        labelImgFileList = self.inputs.labelImgFileList
        ROI_list = self.inputs.ROI_list
        ROI_names = self.inputs.ROI_names
        additionalROIs = self.inputs.additionalROIs
        additionalROI_names = self.inputs.additionalROI_names
        proportiontocut = self.inputs.proportiontocut
        xlsxfile = self.inputs.xlsxFile

        assert(len(ROI_list)==len(ROI_names))
        assert(len(additionalROIs)==len(additionalROI_names))
        assert(len(imgFileList)==len(labelImgFileList))


        # Excel worksheet
        workbook = xlsxwriter.Workbook(xlsxfile)
        worksheet = workbook.add_worksheet('ROI means')

        row = 0
        col = 0
        worksheet.write(row,col,'image path')
        col += 1
        worksheet.write(row,col,'label image path')
        col += 1
        worksheet.write(row,col,'blsaid')
        col += 1
        worksheet.write(row,col,'blsavi')
        for ROI in ROI_list + additionalROIs:
            col += 1
            worksheet.write(row,col,str(ROI))

        row = 1
        col = 0
        worksheet.write(row,col,'image path')
        col += 1
        worksheet.write(row,col,'label image path')
        col += 1
        worksheet.write(row,col,'blsaid')
        col += 1
        worksheet.write(row,col,'blsavi')
        for ROI_name in ROI_names + additionalROI_names:
            col += 1
            worksheet.write(row,col,ROI_name)

        row = 2

        for i, imagefile in enumerate(imgFileList):
            image = nib.load(imagefile)
            image_dat = image.get_data()

            labelimagefile = labelImgFileList[i]
            labelimage = nib.load(labelimagefile)
            labelimage_dat = labelimage.get_data()

            #ROI_list = np.unique(labelimage_dat)
            bn = os.path.basename(imagefile)
            col = 0
            worksheet.write(row,col,imagefile)
            col += 1
            worksheet.write(row,col,labelimagefile)
            col += 1
            worksheet.write(row,col,bn[6:10])
            col += 1
            worksheet.write(row,col,bn[11:13]+'.'+bn[14])

            for ROI in ROI_list:
                ROI_mask = labelimage_dat==ROI
                if ROI_mask.sum()>0:
                    #ROI_mean = image_dat[ROI_mask].mean()
                    ROI_mean = stats.trim_mean(image_dat[ROI_mask], proportiontocut)
                else:
                    ROI_mean = ''
                col += 1
                worksheet.write(row,col,ROI_mean)


            for compositeROI in additionalROIs:
                ROI_mask = labelimage_dat==compositeROI[0]
                if len(compositeROI)>1:
                    for compositeROImember in compositeROI[1:]:
                        ROI_mask = ROI_mask | (labelimage_dat==compositeROImember)
                if ROI_mask.sum()>0:
                    #ROI_mean = image_dat[ROI_mask].mean()
                    ROI_mean = stats.trim_mean(image_dat[ROI_mask], proportiontocut)
                else:
                    ROI_mean = ''
                col += 1
                worksheet.write(row,col,ROI_mean)

            row += 1

        workbook.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs['xlsxFile'] = self.inputs.xlsxFile

        return outputs


class PrepareReportInputSpec(CommandLineInputSpec):
    pyscript = File(desc="Python script to make into a report", exists=True, mandatory=True, position=0, argstr="%s")
    make_pdf = traits.Bool(desc="Make PDF (if false, an HTML report will be generated -- default)", position=1, argstr="-f pdf")

class PrepareReportOutputSpec(TraitedSpec):
    report = File(desc="Report", exists=True)

class PrepareReport(CommandLine):
    _cmd = 'pypublish'
    input_spec = PrepareReportInputSpec
    output_spec = PrepareReportOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()

        fname = self.inputs.pyscript
        pth, base, _ = split_filename(fname)

        if self.inputs.make_pdf:
            outputs['report'] = os.path.join(pth,base + ".pdf")
        else:
            outputs['report'] = os.path.join(pth,base + ".html")

        return outputs


class GeneratePyScriptInputSpec(BaseInterfaceInputSpec):
    petrealignedfile = File(desc="Realigned PET file in native space", exists=True, mandatory=True)
    pet20minfile = File(desc="PET 20 min mean file in native space", exists=True, mandatory=True)
    realignParamsFile = File(desc="PET time frame realignment parameter file (from SPM Realign)", exists=True, mandatory=True)
    maskfile = File(desc="Reference region mask in PET space", exists=True, mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    splitTime = traits.Float(desc='minute into the time series image at which to split the 4D image', mandatory=True)

    petregfile = File(desc="PET 20 min mean file in MRI space", exists=True, mandatory=True)
    mrifile = File(desc="MRI file in native space", exists=True, mandatory=True)

    pet_mnifile = File(desc="PET 20 min mean file in MNI space", exists=True, mandatory=True)
    dvr_wlr_mnifile = File(desc="PiB DVR (WLR) file in MNI space", exists=True, mandatory=False)
    r1_wlr_mnifile = File(desc="PiB R1 (WLR) file in MNI space", exists=True, mandatory=False)
    dvr_lrsc_mnifile = File(desc="PiB DVR (LRSC) file in MNI space", exists=True, mandatory=False)
    r1_lrsc_mnifile = File(desc="PiB R1 (LRSC) file in MNI space", exists=True, mandatory=False)
    dvr_wlr_pvc_mnifile = File(desc="PiB DVR (WLR, PVC) file in MNI space", exists=True, mandatory=False)
    r1_wlr_pvc_mnifile = File(desc="PiB R1 (WLR, PVC) file in MNI space", exists=True, mandatory=False)
    dvr_lrsc_pvc_mnifile = File(desc="PiB DVR (LRSC, PVC) file in MNI space", exists=True, mandatory=False)
    r1_lrsc_pvc_mnifile = File(desc="PiB R1 (LRSC, PVC) file in MNI space", exists=True, mandatory=False)
    suvr_mnifile = File(desc="PET SUVR file in MNI space", exists=True, mandatory=True)
    suvr_pvc_mnifile = File(desc="PET SUVR (PVC) file in MNI space", exists=True, mandatory=True)
    ea_mnifile = File(desc="PiB EA file in MNI space", exists=True, mandatory=False)
    ea_pvc_mnifile = File(desc="PiB EA (PVC) file in MNI space", exists=True, mandatory=False)

    prototypescript = File(desc="Prototype script that will be appended", exists=True, mandatory=True)

class GeneratePyScriptOutputSpec(TraitedSpec):
    pyscript = File(desc="Python script", exists=True)

class GeneratePyScript(BaseInterface):
    input_spec = GeneratePyScriptInputSpec
    output_spec = GeneratePyScriptOutputSpec

    def _run_interface(self, runtime):
        _, base, _ = split_filename(self.inputs.pet_mnifile)

        pyscript = base + '_qc.py'

        f = open(pyscript,'w')

        print("#' # Inputs", file=f)
        print("petrealignedfile = '" + self.inputs.petrealignedfile + "'", file=f)
        print("pet20minfile = '" + self.inputs.pet20minfile + "'", file=f)
        print("realignParamsFile = '" + self.inputs.realignParamsFile + "'", file=f)
        print("maskfile = '" + self.inputs.maskfile + "'", file=f)
        print("frameTimingCsvFile = '" + self.inputs.frameTimingCsvFile + "'", file=f)
        print("splitTime = " + str(self.inputs.splitTime), file=f)

        print("petregfile = '" + self.inputs.petregfile + "'", file=f)
        print("mrifile = '" + self.inputs.mrifile + "'", file=f)

        print("pet_mnifile = '" + self.inputs.pet_mnifile + "'", file=f)
        if isdefined(self.inputs.dvr_wlr_mnifile):
            print("dvr_wlr_mnifile = '" + self.inputs.dvr_wlr_mnifile + "'", file=f)
        if isdefined(self.inputs.r1_wlr_mnifile):
                print("r1_wlr_mnifile = '" + self.inputs.r1_wlr_mnifile + "'", file=f)
        if isdefined(self.inputs.dvr_lrsc_mnifile):
            print("dvr_lrsc_mnifile = '" + self.inputs.dvr_lrsc_mnifile + "'", file=f)
        if isdefined(self.inputs.r1_lrsc_mnifile):
            print("r1_lrsc_mnifile = '" + self.inputs.r1_lrsc_mnifile + "'", file=f)
        if isdefined(self.inputs.dvr_wlr_pvc_mnifile):
            print("dvr_wlr_pvc_mnifile = '" + self.inputs.dvr_wlr_pvc_mnifile + "'", file=f)
        if isdefined(self.inputs.r1_wlr_pvc_mnifile):
            print("r1_wlr_pvc_mnifile = '" + self.inputs.r1_wlr_pvc_mnifile + "'", file=f)
        if isdefined(self.inputs.dvr_lrsc_pvc_mnifile):
            print("dvr_lrsc_pvc_mnifile = '" + self.inputs.dvr_lrsc_pvc_mnifile + "'", file=f)
        if isdefined(self.inputs.r1_lrsc_pvc_mnifile):
            print("r1_lrsc_pvc_mnifile = '" + self.inputs.r1_lrsc_pvc_mnifile + "'", file=f)
        print("suvr_mnifile = '" + self.inputs.suvr_mnifile + "'", file=f)
        print("suvr_pvc_mnifile = '" + self.inputs.suvr_pvc_mnifile + "'", file=f)
        if isdefined(self.inputs.ea_mnifile):
            print("ea_mnifile = '" + self.inputs.ea_mnifile + "'", file=f)
        if isdefined(self.inputs.ea_pvc_mnifile):
            print("ea_pvc_mnifile = '" + self.inputs.ea_pvc_mnifile + "'", file=f)

        print("", file=f)

        for line in open(self.inputs.prototypescript):
            print(line, file=f, end='')

        f.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, base, _ = split_filename(self.inputs.pet_mnifile)

        outputs['pyscript'] = os.path.abspath(base+'_qc.py')

        return outputs
