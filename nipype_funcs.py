from nipype.interfaces.base import TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec, CommandLineInputSpec, CommandLine, isdefined
from nipype.utils.filemanip import split_filename

import nibabel as nib
import numpy as np
import os, sys
import pandas as pd
import re

def get_value(key, dict):
    return dict[key]

def to_div_string(val):
    return ' -div ' + str(val)

def reverse_list(l):
    return l[::-1]

def get_base_filename(pth):
    from nipype.utils.filemanip import split_filename
    _, base, _ = split_filename(pth)
    return base + '_'

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

        # check that frameTiming has the required columns
        for col in ['Duration of time frame (min)','Elapsed time (min)']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = list(frameTiming['Elapsed time (min)'] - frameTiming['Duration of time frame (min)'])
        frameEnd = list(frameTiming['Elapsed time (min)'])

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
        # check that frameTiming has the required columns
        for col in ['Duration of time frame (min)','Elapsed time (min)']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = list(frameTiming['Elapsed time (min)'] - frameTiming['Duration of time frame (min)'])
        frameEnd = list(frameTiming['Elapsed time (min)'])

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
        # check that frameTiming has the required columns
        for col in ['Duration of time frame (min)','Elapsed time (min)']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = list(frameTiming['Elapsed time (min)'] - frameTiming['Duration of time frame (min)'])
        frameEnd = list(frameTiming['Elapsed time (min)'])

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
        meanImg = nib.Nifti1Image(np.squeeze(meanImg_dat), img.affine, img.header)
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
    ROI_groupings = traits.List(traits.List(traits.Int(), minlen=1),
                                minlen=1, desc='list of lists of integers',
                                mandatory=True)

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
        ROI4Dmask = nib.Nifti1Image(np.squeeze(ROI4Dmask_dat), labelimage.affine, labelimage.header)
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
                padded_dat[:,y,:,l] = padded_dat[:,padsize,:,l]
                padded_dat[:,-(y+1),:,l] = padded_dat[:,-(padsize+1),:,l]
            for z in range(padsize):
                padded_dat[:,:,z,l] = padded_dat[:,:,padsize,l]
                padded_dat[:,:,-(z+1),l] = padded_dat[:,:,-(padsize+1),l]

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
    startTime = traits.Float(0, desc='minute into the time series image at which to start computing the parametric images, inclusive', mandatory=False)
    endTime = traits.Float(desc='minute into the time series image at which to stop computing the parametric images, exclusive', mandatory=True)
    proportiontocut = traits.Range(low=0.0, high=0.25,
                                   desc='proportion to cut from each tail of the distribution before computing the mean signal in the reference region, in range 0.00-0.25')
                     #traits.Float(0, desc='proportion to cut from each tail of the distribution before computing the mean signal in the reference region')
    fwhm = traits.Float(desc='Full width at half max (in mm) for Gaussian smoothing',mandatory=True)

class SRTMOutputSpec(TraitedSpec):
    startTime = traits.Float(desc='possibly modified start time')
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

    def _run_interface(self, runtime):
        import numpy.matlib as mat
        from scipy import stats, ndimage, integrate, linalg
        import math

        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        refRegionMaskFile = self.inputs.refRegionMaskFile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        startTime = self.inputs.startTime
        endTime = self.inputs.endTime
        proportiontocut = self.inputs.proportiontocut
        fwhm = self.inputs.fwhm

        _, base, _ = split_filename(timeSeriesImgFile)

        frameTiming = pd.read_csv(frameTimingCsvFile)
        # check that frameTiming has the required columns
        for col in ['Duration of time frame (min)','Elapsed time (min)']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = frameTiming['Elapsed time (min)'] - frameTiming['Duration of time frame (min)']
        frameEnd = frameTiming['Elapsed time (min)']

        frameStart = frameStart.as_matrix() #tolist()
        frameEnd = frameEnd.as_matrix() #tolist()

        if startTime < frameStart[0]:
            startTime = frameStart[0]
        elif startTime > frameEnd[-1]:
            sys.exit('Start time is beyond the time covered by the time series data!')

        if endTime > frameEnd[-1]:
            endTime = frameEnd[-1]
        elif endTime < frameStart[0]:
            sys.exit('End time is prior to the time covered by the time series data!')

        # find the first time frame with frameStart at or shortest after the specified start time (this frame will be included in the average)
        startIndex = next((i for i,t in enumerate(frameStart) if t>=startTime), len(frameTiming)-1)
        # find the first time frame with frameEnd shortest after the specified end time (this frame will not be included in the average)
        endIndex = next((i for i,t in enumerate(frameEnd) if t>endTime), len(frameTiming))

        # another sanity check, mainly to make sure that startIndex!=endIndex
        assert(startIndex<endIndex)

        # get the actual start and end times for the 4D image to be used
        self.modStartTime = frameStart[startIndex]
        self.modEndTime = frameEnd[endIndex-1]

        # Compute the time mid-way for each time frame
        t = (frameStart[startIndex:endIndex] + frameEnd[startIndex:endIndex])/2
        # Compute the duration of each time frame
        delta = frameEnd[startIndex:endIndex] - frameStart[startIndex:endIndex]

        # load 4D image
        img = nib.load(timeSeriesImgFile)
        img_dat = img.get_data()[:,:,:,startIndex:endIndex]
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
            Cref[l] = stats.trim_mean(tmp[np.isfinite(tmp)], proportiontocut)
            Ctfilt[:,:,:,l] = ndimage.gaussian_filter(img_dat[:,:,:,l],sigma=sigma,order=0)

        # Integrals etc.
        # Numerical integration of TAC
        intCref = integrate.cumtrapz(Cref,t,initial=0)
        # account for ignored frames at the beginning:
        #   assume signal increased linearly with time until first included frame,
        #   starting at 0 at time 0.
        #   So we compute the are of this right triangle and add to the trapezoid integral
        intCref = intCref + t[0] * Cref[0] / 2

        # Numerical integration of Ct
        intCt_ = integrate.cumtrapz(img_dat,t,axis=3,initial=0)
        # account for ignored frames at the beginning:
        #   assume signal increased linearly with time until first included frame,
        #   starting at 0 at time 0.
        #   So we compute the are of this right triangle and add to the trapezoid integral
        intCt_ = intCt_ + t[0] * img_dat[:,:,:,:1] / 2

        # STEP 1: weighted linear regression (wlr) [Zhou 2003 p. 978]
        m = 3
        W = mat.diag(delta)
        B0_wlr = np.zeros((rows,cols,slices,m))
        var_B0_wlr = np.zeros((rows,cols,slices))
        B1_wlr = np.zeros((rows,cols,slices,m))
        var_B1_wlr = np.zeros((rows,cols,slices))

        # This for-loop will be more efficient if iteration is performed only
        # over voxels within mask
        for i in range(rows):
            for j in range(cols):
                for k in range(slices):
                    if mask[i,j,k]:
                        Ct = img_dat[i,j,k,:]
                        intCt = intCt_[i,j,k,:]

                        X = np.mat(np.column_stack((intCref, Cref, -Ct)))
                        y = np.mat(intCt).T
                        b0 = linalg.solve(X.T * W * X, X.T * W * y)
                        residual = y - X * b0
                        var_b0 = residual.T * W * residual / (comps-m)
                        B0_wlr[i,j,k,:] = b0.T
                        var_B0_wlr[i,j,k] = var_b0

                        XR1 = np.mat(np.column_stack((Cref, intCref, -intCt)))
                        yR1 = np.mat(Ct).T
                        b1 = linalg.solve(XR1.T * W * XR1, XR1.T * W * yR1)
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
            B0_sc[:,:,:,l] = ndimage.gaussian_filter(B0_wlr[:,:,:,l],sigma=sigma,order=0)
            B1_sc[:,:,:,l] = ndimage.gaussian_filter(B1_wlr[:,:,:,l],sigma=sigma,order=0)

        H0 = np.zeros(B0_wlr.shape)
        H1 = np.zeros(B1_wlr.shape)
        for i in range(rows):
            for j in range(cols):
                for k in range(slices):
                    if mask[i,j,k]:
                        H0[i,j,k,:] = m * var_B0_wlr[i,j,k] / np.square(B0_wlr[i,j,k,:] - B0_sc[i,j,k,:])
                        H1[i,j,k,:] = m * var_B1_wlr[i,j,k] / np.square(B1_wlr[i,j,k,:] - B1_sc[i,j,k,:])

        # Apply spatial smoothing to H0 and H1
        HH0 = np.zeros(H0.shape)
        HH1 = np.zeros(H1.shape)
        for l in range(m):
            HH0[:,:,:,l] = ndimage.gaussian_filter(H0[:,:,:,l],sigma=sigma,order=0)
            HH1[:,:,:,l] = ndimage.gaussian_filter(H1[:,:,:,l],sigma=sigma,order=0)

        # STEP 2: ridge regression [Zhou 2003 p. 978]
        B0_lrsc = np.zeros((rows,cols,slices,m))
        B1_lrsc = np.zeros((rows,cols,slices,m))

        # This for-loop will be more efficient if iteration is performed only
        # over voxels within mask
        for i in range(rows):
            for j in range(cols):
                for k in range(slices):
                    if mask[i,j,k]:
                        Ct = img_dat[i,j,k,:]
                        Ct_smoothed = Ctfilt[i,j,k,:]
                        intCt = intCt_[i,j,k,:]

                        X = np.mat(np.column_stack((intCref, Cref, -Ct_smoothed)))
                        y = np.mat(intCt).T
                        b0 = linalg.solve(X.T * W * X + mat.diag(HH0[i,j,k,:]),
                                             X.T * W * y + mat.diag(HH0[i,j,k,:]) * np.mat(B0_sc[i,j,k,:]).T)
                        B0_lrsc[i,j,k,:] = b0.T

                        XR1 = np.mat(np.column_stack((Cref, intCref, -intCt)))
                        yR1 = np.mat(Ct).T
                        b1 = linalg.solve(XR1.T * W * XR1 + mat.diag(HH1[i,j,k,:]),
                                             XR1.T * W * yR1 + mat.diag(HH1[i,j,k,:]) * np.mat(B1_sc[i,j,k,:]).T)
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

        outputs['startTime'] = self.modStartTime
        outputs['endTime'] = self.modEndTime
        outputs['DVRImgFile_wlr'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_DVR_wlr.nii.gz')
        outputs['R1ImgFile_wlr'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_R1_wlr.nii.gz')
        outputs['DVRImgFile_lrsc'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_DVR_lrsc.nii.gz')
        outputs['R1ImgFile_lrsc'] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_R1_lrsc.nii.gz')

        return outputs


class ROI_stats_to_spreadsheetInputSpec(BaseInterfaceInputSpec):
    imgFileList = traits.List(File(exists=True), minlen=1,
                              desc='Image list', mandatory=True)
    labelImgFileList = traits.List(File(exists=True), minlen=1,
                                   desc='Label image list', mandatory=True)
    xlsxFile = File(exists=False, desc='xlsx file. If it doesn''t exist, it will be created',
                    mandatory=True)
    ROI_list = traits.List(traits.Int(), minlen=1,
                           desc='list of ROI indices for which stats will be computed (should match the label indices in the label image)',
                           mandatory=True)
    ROI_names = traits.List(traits.String(), minlen=1,
                            desc='list of equal size to ROI_list that lists the corresponding ROI names',
                            mandatory=True)
    additionalROIs = traits.List(traits.List(traits.Int()), desc='list of lists of integers')
    additionalROI_names = traits.List(traits.String(),
                                      desc='names corresponding to additional ROIs')
    stat = traits.Enum('mean','Q1','median','Q3','min','max',
                       desc='one of: mean, Q1, median, Q3, min, max',
                       mandatory=True)
    proportiontocut = traits.Range(low=0.0, high=0.25,
                                   desc='proportion to cut from each tail of the distribution before computing the mean signal in the reference region, in range 0.00-0.25')

class ROI_stats_to_spreadsheetOutputSpec(TraitedSpec):
    xlsxFile = File(exists=True, desc='xlsx file')

class ROI_stats_to_spreadsheet(BaseInterface):
    """
    Compute ROI statistics and write to spreadsheet

    """

    input_spec = ROI_stats_to_spreadsheetInputSpec
    output_spec = ROI_stats_to_spreadsheetOutputSpec

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
        stat = self.inputs.stat

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
        worksheet.write(row,col,'id')
        for ROI in ROI_list + additionalROIs:
            col += 1
            worksheet.write(row,col,str(ROI))

        row = 1
        col = 0
        worksheet.write(row,col,'image path')
        col += 1
        worksheet.write(row,col,'label image path')
        col += 1
        worksheet.write(row,col,'id')
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

            m = re.search(r'.+/_id_([^/]+)/.+', imagefile)
            if m:
                worksheet.write(row,col,m.group(1))
            else:
                m = re.search(r'.+/_id_([^/]+)/.+', labelimagefile)
                if m:
                    worksheet.write(row,col,m.group(1))


            for ROI in ROI_list:
                ROI_mask = labelimage_dat==ROI
                if ROI_mask.sum()>0:
                    if stat=="mean":
                        #ROI_stat = image_dat[ROI_mask].mean()
                        ROI_stat = stats.trim_mean(image_dat[ROI_mask], proportiontocut)
                    elif stat=="Q1":
                        ROI_stat = np.percentile(image_dat[ROI_mask], 25)
                    elif stat=="median":
                        ROI_stat = np.percentile(image_dat[ROI_mask], 50)
                    elif stat=="Q3":
                        ROI_stat = np.percentile(image_dat[ROI_mask], 75)
                    elif stat=="min":
                        ROI_stat = np.min(image_dat[ROI_mask])
                    elif stat=="max":
                        ROI_stat = np.max(image_dat[ROI_mask])
                    else:
                        ROI_stat=''
                    if np.isnan(ROI_stat):
                        ROI_stat = ''
                else:
                    ROI_stat = ''
                col += 1
                worksheet.write(row,col,ROI_stat)


            for compositeROI in additionalROIs:
                ROI_mask = labelimage_dat==compositeROI[0]
                if len(compositeROI)>1:
                    for compositeROImember in compositeROI[1:]:
                        ROI_mask = ROI_mask | (labelimage_dat==compositeROImember)
                if ROI_mask.sum()>0:
                    if stat=="mean":
                        #ROI_stat = image_dat[ROI_mask].mean()
                        ROI_stat = stats.trim_mean(image_dat[ROI_mask], proportiontocut)
                    elif stat=="Q1":
                        ROI_stat = np.percentile(image_dat[ROI_mask], 25)
                    elif stat=="median":
                        ROI_stat = np.percentile(image_dat[ROI_mask], 50)
                    elif stat=="Q3":
                        ROI_stat = np.percentile(image_dat[ROI_mask], 75)
                    elif stat=="min":
                        ROI_stat = np.min(image_dat[ROI_mask])
                    elif stat=="max":
                        ROI_stat = np.max(image_dat[ROI_mask])
                    else:
                        ROI_stat=''
                    if np.isnan(ROI_stat):
                        ROI_stat = ''
                else:
                    ROI_stat = ''
                col += 1
                worksheet.write(row,col,ROI_stat)

            row += 1

        workbook.close()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs['xlsxFile'] = self.inputs.xlsxFile

        return outputs


class realign_snapshotsInputSpec(BaseInterfaceInputSpec):
    petrealignedfile = File(exists=True, desc='Realigned 4D PET file', mandatory=True)
    realignParamsFile = File(exists=True, desc='Realignment parameters text file', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    splitTime = traits.Float(desc='minute beyond which time frames were realigned, inclusive', mandatory=True)

class realign_snapshotsOutputSpec(TraitedSpec):
    realign_param_plot = File(exists=True, desc='Realignment parameter plot')
    realigned_img_snap = File(exists=True, desc='Realigned time frame snapshot')

class realign_snapshots(BaseInterface):
    input_spec = realign_snapshotsInputSpec
    output_spec = realign_snapshotsOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt
        from math import pi

        petrealignedfile = self.inputs.petrealignedfile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        splitTime = self.inputs.splitTime
        realignParamsFile = self.inputs.realignParamsFile


        frameTiming = pd.read_csv(frameTimingCsvFile)
        # check that frameTiming has the required columns
        for col in ['Duration of time frame (min)','Elapsed time (min)']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = frameTiming['Elapsed time (min)'] - frameTiming['Duration of time frame (min)']
        frameEnd = frameTiming['Elapsed time (min)']

        frameStart = frameStart.as_matrix() #tolist()
        frameEnd = frameEnd.as_matrix() #tolist()

        splitIndex = next((i for i,t in enumerate(frameStart) if t>=splitTime), len(frameTiming))

        # Compute the time mid-way for each time frame, to be used for plotting purposes
        t = (frameStart + frameEnd)/2


        # Time realignment parameters
        rp = pd.read_csv(realignParamsFile, delim_whitespace=True, header=None).as_matrix()
        translation = rp[1:,:3]
        rotation = rp[1:,3:] * 180 / pi

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))
        axes[0].plot(t[splitIndex:],translation[:,0],label='x')
        axes[0].plot(t[splitIndex:],translation[:,1],label='y')
        axes[0].plot(t[splitIndex:],translation[:,2],label='z')
        axes[0].legend(loc=0)
        axes[0].set_title('Translation over time')
        axes[0].set_xlabel('Time (min)', fontsize=16)
        axes[0].set_ylabel('Translation (mm)', fontsize=16)

        axes[1].plot(t[splitIndex:],rotation[:,0],label='x')
        axes[1].plot(t[splitIndex:],rotation[:,1],label='y')
        axes[1].plot(t[splitIndex:],rotation[:,2],label='z')
        axes[1].legend(loc=0)
        axes[1].set_title('Rotation over time')
        axes[1].set_xlabel('Time (min)', fontsize=16)
        axes[1].set_ylabel('Rotation (degrees)', fontsize=16)

        fig.tight_layout()

        _, base, _ = split_filename(realignParamsFile)
        fig.savefig(base+'_plot.png', format='png')


        # visualize time frames of the realigned scan
        petrealigned = nib.load(petrealignedfile)
        I = petrealigned.get_data()
        imdim = I.shape
        vmin, vmax = np.percentile(I,[1,99])
        voxsize = petrealigned.header.get_zooms()

        # Right hemisphere is on the right hand side
        nx = int(np.ceil(np.sqrt(imdim[3])))
        fig, axes = plt.subplots(nrows=nx, ncols=nx, figsize=(16,16))
        x = y = 0
        for tt in range(nx ** 2):
            if tt < imdim[3]:
                axes[x,y].imshow(np.fliplr(I[:,:,imdim[2]//2,tt]).T, aspect=voxsize[1]/voxsize[0], cmap='hot', vmin=vmin, vmax=vmax)
                axes[x,y].set_title('#'+str(tt)+': '+'{:.2f}'.format(frameStart[tt])+'-'+'{:.2f}'.format(frameEnd[tt])+' min')
            axes[x,y].set_axis_off()
            y += 1
            if y>=np.ceil(np.sqrt(imdim[3])):
                y = 0
                x += 1

        fig.tight_layout()

        _, base, _ = split_filename(petrealignedfile)
        fig.savefig(base+'_snap.png', format='png')

        return runtime

    def _list_outputs(self):
        petrealignedfile = self.inputs.petrealignedfile
        realignParamsFile = self.inputs.realignParamsFile

        outputs = self._outputs().get()

        _, base, _ = split_filename(realignParamsFile)
        outputs['realign_param_plot'] = os.path.abspath(base+'_plot.png')

        _, base, _ = split_filename(petrealignedfile)
        outputs['realigned_img_snap'] = os.path.abspath(base+'_snap.png')

        return outputs


class coreg_snapshotsInputSpec(BaseInterfaceInputSpec):
    mriregfile = File(exists=True, desc='MRI image registered to PET', mandatory=True)
    petavgfile = File(exists=True, desc='PET average image', mandatory=True)

class coreg_snapshotsOutputSpec(TraitedSpec):
    coreg_edges = File(exists=True, desc='PET with coregistered MRI edges')
    coreg_overlay_sagittal = File(exists=True, desc='Overlay of PET and coregistered MRI, sagittal')
    coreg_overlay_coronal = File(exists=True, desc='Overlay of PET and coregistered MRI, coronal')
    coreg_overlay_axial = File(exists=True, desc='Overlay of PET and coregistered MRI, axial')

class coreg_snapshots(BaseInterface):
    input_spec = coreg_snapshotsInputSpec
    output_spec = coreg_snapshotsOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt
        from dipy.viz import regtools
        from nilearn.plotting import plot_anat

        petavgfile = self.inputs.petavgfile
        mriregfile = self.inputs.mriregfile

        mrireg = nib.load(mriregfile)
        petavg = nib.load(petavgfile)

        _, base, _ = split_filename(petavgfile)

        # Visualize the overlaid PiB 20-min average and the coregistered MRI
        p = regtools.overlay_slices(petavg.get_data(), mrireg.get_data(),
                                    None, 0, "PET", "Coregistered MRI",
                                    fname=base+'_coreg_overlay_sagittal.png')
        p = regtools.overlay_slices(petavg.get_data(), mrireg.get_data(),
                                    None, 1, "PET", "Coregistered MRI",
                                    fname=base+'_coreg_overlay_coronal.png')
        p = regtools.overlay_slices(petavg.get_data(), mrireg.get_data(),
                                    None, 2, "PET", "Coregistered MRI",
                                    fname=base+'_coreg_overlay_axial.png')

        fig = plt.figure(figsize=(15,5))
        display = plot_anat(petavgfile,figure=fig)
        display.add_edges(mriregfile)
        display.title('MRI edges on PET')
        fig.savefig(base+'_coreg_edges.png', format='png')

        return runtime

    def _list_outputs(self):
        petavgfile = self.inputs.petavgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(petavgfile)
        outputs['coreg_edges'] = os.path.abspath(base+'_coreg_edges.png')
        outputs['coreg_overlay_sagittal'] = os.path.abspath(base+'_coreg_overlay_sagittal.png')
        outputs['coreg_overlay_coronal'] = os.path.abspath(base+'_coreg_overlay_coronal.png')
        outputs['coreg_overlay_axial'] = os.path.abspath(base+'_coreg_overlay_axial.png')

        return outputs



class labels_snapshotsInputSpec(BaseInterfaceInputSpec):
    labelfile = File(exists=True, desc='4D label image', mandatory=True)
    labelnames = traits.List(traits.String(), minlen=1,
                             desc='list of equal size to the fourth dimension of the label image',
                             mandatory=True)

class labels_snapshotsOutputSpec(TraitedSpec):
    label_snap = File(exists=True, desc='Label image snapshot')

class labels_snapshots(BaseInterface):
    input_spec = labels_snapshotsInputSpec
    output_spec = labels_snapshotsOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt

        labelfile = self.inputs.labelfile
        labelnames = self.inputs.labelnames

        label = nib.load(labelfile)
        I = label.get_data()
        imdim = I.shape
        voxsize = label.header.get_zooms()

        # Right hemisphere is on the right hand side
        fig, axes = plt.subplots(imdim[3],3,figsize=(10,60))
        for tt in range(imdim[3]):

            sli = np.argmax(np.sum(I[:,:,:,tt],axis=(0,1)))
            axes[tt,0].imshow(np.fliplr(I[:,:,sli,tt]).T, aspect=voxsize[1]/voxsize[0])
            axes[tt,0].set_axis_off()

            sli = np.argmax(np.sum(I[:,:,:,tt],axis=(0,2)))
            axes[tt,1].imshow(np.fliplr(I[:,sli,:,tt]).T, aspect=voxsize[2]/voxsize[0])
            axes[tt,1].set_axis_off()
            axes[tt,1].set_title(labelnames[tt])

            sli = np.argmax(np.sum(I[:,:,:,tt],axis=(1,2)))
            axes[tt,2].imshow(np.fliplr(I[sli,:,:,tt]).T, aspect=voxsize[2]/voxsize[1])
            axes[tt,2].set_axis_off()
        fig.tight_layout()

        _, base, _ = split_filename(labelfile)
        fig.savefig(base+'_snap.png', format='png')

        return runtime

    def _list_outputs(self):
        labelfile = self.inputs.labelfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(labelfile)
        outputs['label_snap'] = os.path.abspath(base+'_snap.png')

        return outputs



class refReg_snapshotsInputSpec(BaseInterfaceInputSpec):
    petavgfile = File(exists=True, desc='PET average image', mandatory=True)
    petrealignedfile = File(exists=True, desc='Realigned 4D PET file', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    maskfile = File(exists=True, desc='Mask file', mandatory=True)

class refReg_snapshotsOutputSpec(TraitedSpec):
    maskOverlay_axial = File(exists=True, desc='Mask overlaid on PET, axial')
    maskOverlay_coronal = File(exists=True, desc='Mask overlaid on PET, coronal')
    maskOverlay_sagittal = File(exists=True, desc='Mask overlaid on PET, sagittal')
    mask_TAC = File(exists=True, desc='Reference region time activity curve')

class refReg_snapshots(BaseInterface):
    input_spec = refReg_snapshotsInputSpec
    output_spec = refReg_snapshotsOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_anat, cm
        from nilearn.masking import apply_mask

        petavgfile = self.inputs.petavgfile
        petrealignedfile = self.inputs.petrealignedfile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        maskfile = self.inputs.maskfile

        frameTiming = pd.read_csv(frameTimingCsvFile)
        # check that frameTiming has the required columns
        for col in ['Duration of time frame (min)','Elapsed time (min)']:
            if not col in frameTiming.columns:
                sys.exit('Required column '+col+' is not present in the frame timing spreadsheet '+frameTimingCsvFile+'!')
        frameStart = frameTiming['Elapsed time (min)'] - frameTiming['Duration of time frame (min)']
        frameEnd = frameTiming['Elapsed time (min)']

        frameStart = frameStart.as_matrix() #tolist()
        frameEnd = frameEnd.as_matrix() #tolist()

        # Compute the time mid-way for each time frame, to be used for plotting purposes
        t = (frameStart + frameEnd)/2


        _, base, _ = split_filename(petavgfile)
        fig = plt.figure(figsize=(16,2))
        display = plot_anat(petavgfile,figure=fig, display_mode="z", cut_coords=10)
        #display.add_edges(maskfile)
        display.add_overlay(maskfile, cmap=cm.red_transparent)
        display.title('Reference region on PET - axial')
        fig.savefig(base+'_maskOverlay_axial.png', format='png')

        fig = plt.figure(figsize=(16,2))
        display = plot_anat(petavgfile,figure=fig, display_mode="y", cut_coords=10, annotate=False)
        #display.add_edges(maskfile)
        display.add_overlay(maskfile, cmap=cm.red_transparent)
        display.title('Reference region on PET - coronal')
        fig.savefig(base+'_maskOverlay_coronal.png', format='png')

        fig = plt.figure(figsize=(16,2))
        display = plot_anat(petavgfile,figure=fig, display_mode="x", cut_coords=10, annotate=False)
        #display.add_edges(maskfile)
        display.add_overlay(maskfile, cmap=cm.red_transparent)
        display.title('Reference region on PET - sagittal')
        fig.savefig(base+'_maskOverlay_sagittal.png', format='png')

        # Reference region Time Activity Curve (TAC)
        masked_data = apply_mask(petrealignedfile, maskfile)
        ref_TAC = np.mean(masked_data,axis=1)

        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(t,ref_TAC)
        ax.set_title('Reference region Time Activity Curve')
        ax.set_xlabel('Time (min)', fontsize=16)
        ax.set_ylabel('Activity', fontsize=16)
        fig.tight_layout()

        _, base, _ = split_filename(petrealignedfile)
        fig.savefig(base+'_mask_TAC.png', format='png')

        return runtime

    def _list_outputs(self):
        petrealignedfile = self.inputs.petrealignedfile
        petavgfile = self.inputs.petavgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(petavgfile)
        outputs['maskOverlay_axial'] = os.path.abspath(base+'_maskOverlay_axial.png')
        outputs['maskOverlay_coronal'] = os.path.abspath(base+'_maskOverlay_coronal.png')
        outputs['maskOverlay_sagittal'] = os.path.abspath(base+'_maskOverlay_sagittal.png')

        _, base, _ = split_filename(petrealignedfile)
        outputs['mask_TAC'] = os.path.abspath(base+'_mask_TAC.png')

        return outputs


class triplanar_snapshotsInputSpec(BaseInterfaceInputSpec):
    imgfile = File(exists=True, desc='Image file', mandatory=True)
    bgimgfile = File(exists=True, desc='Background image file', mandatory=False)
    vmin = traits.Float(desc='vmin', mandatory=False)
    vmax = traits.Float(desc='vmax', mandatory=False)
    cmap = traits.String(desc='cmap', mandatory=False)
    alpha = traits.Range(low=0.0, high=1.0, desc='alpha', mandatory=False) # higher alpha means more bg visibility
    x = traits.Range(low=0, desc='x cut', mandatory=False)
    y = traits.Range(low=0, desc='y cut', mandatory=False)
    z = traits.Range(low=0, desc='z cut', mandatory=False)

class triplanar_snapshotsOutputSpec(TraitedSpec):
    triplanar = File(exists=True, desc='Triplanar snapshot')

class triplanar_snapshots(BaseInterface):
    input_spec = triplanar_snapshotsInputSpec
    output_spec = triplanar_snapshotsOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt

        imgfile = self.inputs.imgfile
        bgimgfile = self.inputs.bgimgfile
        vmin = self.inputs.vmin
        vmax = self.inputs.vmax
        cmap = self.inputs.cmap
        alpha = self.inputs.alpha
        x = self.inputs.x
        y = self.inputs.y
        z = self.inputs.z

        if not isdefined(cmap):
            cmap = 'jet'
        if not isdefined(alpha):
            if not isdefined(bgimgfile):
                alpha = 0
            else:
                alpha = 0.5

        img = nib.load(imgfile)
        I = img.get_data()
        imdim = I.shape
        voxsize = img.header.get_zooms()

        if not isdefined(vmin):
            vmin = np.percentile(np.abs(I),5)
        if not isdefined(vmax):
            vmax = np.percentile(np.abs(I),100)
        if not isdefined(x):
            x = imdim[0]//2
        if not isdefined(y):
            y = imdim[1]//2
        if not isdefined(z):
            z = imdim[2]//2

        if isdefined(bgimgfile):
            bgimg = nib.load(bgimgfile)
            bgI = bgimg.get_data()
            bgimdim = bgI.shape
            bgvoxsize = bgimg.header.get_zooms()
            assert(imdim==bgimdim)
            assert(voxsize==bgvoxsize)

            # trim to remove 0 voxels
            trimmask = bgI>0
            tmp = np.argwhere(trimmask)
            (xstart, ystart, zstart), (xstop, ystop, zstop) = tmp.min(0), tmp.max(0) + 1
            bgI = bgI[xstart:xstop, ystart:ystop, zstart:zstop]
            I = I[xstart:xstop, ystart:ystop, zstart:zstop]
            imdim = I.shape
            bgimdim = bgI.shape
            x = x - xstart
            y = y - ystart
            z = z - zstart

            mask = bgI==0
            bgI = np.ma.array(bgI, mask=mask)
            I = np.ma.array(I, mask=mask)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,5), facecolor='black')
        axes[0].imshow(np.fliplr(I[:,:,z]).T,
                       aspect=voxsize[1]/voxsize[0],
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)

        axes[1].imshow(np.fliplr(I[:,y,:]).T,
                       aspect=voxsize[2]/voxsize[0],
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)

        im = axes[2].imshow(np.fliplr(I[x,:,:]).T,
                            aspect=voxsize[2]/voxsize[1],
                            cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)

        if isdefined(bgimgfile):
            axes[0].imshow(np.fliplr(bgI[:,:,z]).T,
                           aspect=voxsize[1]/voxsize[0],
                           cmap='gray',alpha=alpha)
            axes[1].imshow(np.fliplr(bgI[:,y,:]).T,
                           aspect=voxsize[2]/voxsize[0],
                           cmap='gray',alpha=alpha)
            axes[2].imshow(np.fliplr(bgI[x,:,:]).T,
                           aspect=voxsize[2]/voxsize[1],
                           cmap='gray',alpha=alpha)

        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[2].set_axis_off()

        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=20) # colorbar legend font size

        # colorbar ticks' and labels' color
        cbar.ax.yaxis.set_tick_params(color='w')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')

        # smooth colorbar without lines in it
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")

        _, base, _ = split_filename(imgfile)
        fig.savefig(base+'_snap.png', format='png', facecolor=fig.get_facecolor())

        return runtime

    def _list_outputs(self):
        imgfile = self.inputs.imgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(imgfile)
        outputs['triplanar'] = os.path.abspath(base+'_snap.png')

        return outputs


class mosaicInputSpec(BaseInterfaceInputSpec):
    imgfile = File(exists=True, desc='Image file', mandatory=True)
    bgimgfile = File(exists=True, desc='Background image file', mandatory=False)
    vmin = traits.Float(desc='vmin')
    vmax = traits.Float(desc='vmax')
    cmap = traits.String(desc='cmap', mandatory=False)
    alpha = traits.Range(low=0.0, high=1.0, desc='alpha', mandatory=False) # higher alpha means more bg visibility

class mosaicOutputSpec(TraitedSpec):
    mosaic = File(exists=True, desc='Mosaic snapshot')

class mosaic(BaseInterface):
    input_spec = mosaicInputSpec
    output_spec = mosaicOutputSpec

    def _run_interface(self, runtime):
        import matplotlib.pyplot as plt

        imgfile = self.inputs.imgfile
        bgimgfile = self.inputs.bgimgfile
        vmin = self.inputs.vmin
        vmax = self.inputs.vmax
        cmap = self.inputs.cmap
        alpha = self.inputs.alpha

        if not isdefined(cmap):
            cmap = 'jet'
        if not isdefined(alpha):
            alpha = 0

        img = nib.load(imgfile)
        I = img.get_data()
        imdim = I.shape
        voxsize = img.header.get_zooms()

        if isdefined(bgimgfile):
            bgimg = nib.load(bgimgfile)
            bgI = bgimg.get_data()
            bgimdim = bgI.shape
            bgvoxsize = bgimg.header.get_zooms()
            assert(imdim==bgimdim)
            assert(voxsize==bgvoxsize)
            mask = bgI==0
            bgI = np.ma.array(bgI, mask=mask)
            I = np.ma.array(I, mask=mask)

        nx = int(np.ceil(np.sqrt(imdim[2])))
        fig, axes = plt.subplots(nrows=nx, ncols=nx, figsize=(nx*3,nx*3), facecolor='black')
        x = y = 0
        for z in range(nx ** 2):
            if z < imdim[2]:
                if isdefined(bgimgfile):
                    axes[x,y].imshow(np.fliplr(bgI[:,:,z]).T, aspect=voxsize[1]/voxsize[0], cmap='gray', alpha=alpha)
                im = axes[x,y].imshow(np.fliplr(I[:,:,z]).T, aspect=voxsize[1]/voxsize[0], cmap=cmap, vmin=vmin, vmax=vmax, alpha=1-alpha)
            axes[x,y].set_axis_off()
            axes[x,y].set_adjustable('box-forced')
            y += 1
            if y>=np.ceil(np.sqrt(imdim[2])):
                y = 0
                x += 1

        cax = fig.add_axes([0.1, 0.03, 0.8, 0.03])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=30) # colorbar legend font size

        # colorbar ticks' and labels' color
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
        cbar.ax.xaxis.set_tick_params(color='w')

        # smooth colorbar without lines in it
        cbar.solids.set_rasterized(True)
        cbar.solids.set_edgecolor("face")
        cbar.set_alpha(1)
        cbar.draw_all() # recent addition

        fig.tight_layout()

        _, base, _ = split_filename(imgfile)
        fig.savefig(base+'_mosaic.png', format='png', facecolor=fig.get_facecolor())

        return runtime

    def _list_outputs(self):
        imgfile = self.inputs.imgfile

        outputs = self._outputs().get()

        _, base, _ = split_filename(imgfile)
        outputs['mosaic'] = os.path.abspath(base+'_mosaic.png')

        return outputs
