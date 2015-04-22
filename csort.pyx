#!python
#cython: boundscheck=False, wraparound = False, initializedcheck = False, nonecheck = False, overflowcheck = False, overflowcheck.fold = False, embedsignature = False, unraisable_tracebacks = False 

from __future__ import division
import numpy as np
#import cv2
#import scipy as sp
#import sys
#sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')

cimport numpy as np
cimport cython




#from scipy.ndimage import label
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

#################


@cython.boundscheck(False)
@cython.wraparound(False)
def countSort(np.ndarray[np.int32_t, ndim=1] inputimg):
    '''
    countSort(np.ndarray[DTYPE_t, ndim=1] inputimg) -> sorted index of image
    '''
    cdef int total = 0
    cdef int oldcount
    cdef int inputimgsize1d =  inputimg.shape[0]*inputimg.shape[1]
    cdef int arraysize = 256
    cdef int ii = 0
#    create array for counting the intensity values
    cdef np.ndarray countarray = np.zeros([arraysize], dtype = np.int) 
#    cdef np.ndarray inputimg1d = np.zeros([1, inputimgsize1d], dtype = np.int)
#    cdef np.ndarray indexarray1d = np.zeros([1,inputimgsize1d], dtype = np.int)
#    cdef np.ndarray indexarray = np.zeros([inputimg.shape[0],inputimg.shape[1] ], dtype = np.int)
    cdef np.ndarray sortedarray1d = np.zeros([inputimgsize1d], dtype = np.int)
#    cdef np.ndarray sortedarray1d = np.zeros([1,inputimgsize1d], dtype = np.int)
    
#    cdef np.ndarray sortedarray = np.zeros([inputimg.shape[0],inputimg.shape[1] ], dtype = np.int)

#    inputimg1d = inputimg.reshape((1,inputimgsize1d))
#    inputimg1d = inputimg.reshape((1,inputimgsize1d))
    
#    create histogram of intensities
    
    for ii in inputimg:
#        print ii
        countarray[ii] += 1

#    cumulative hitogram
#    total = 0
#    for ii in np.arange(arraysize-1,-1,-1):# reverse sorting        
    for ii in range(arraysize): # forwards sorting
        oldcount = countarray[ii]
        countarray[ii] = total
        total += oldcount
        
#   do sorting and return value and index arrays   
#    for ii in inputimg[::-1]: # reverse sorting
    for ii in inputimg: # forward sorting
        
        sortedarray1d[countarray[ii]] = ii
#        indexarray1d[ind] = countarray[ii]        
        countarray[ii] += 1
        
#    reshape arrays
#    sortedarray = sortedarray1d.reshape((inputimg.shape[0],inputimg.shape[1] ))
    
    return(sortedarray1d)




   
###############################################################################
#@cython.boundscheck(False)
#def countSort(np.ndarray[DTYPE_t, ndim=2] inputimg):
#    '''
#    countSort(np.ndarray[DTYPE_t, ndim=2] inputimg) -> sorted index of image
#    '''
#    cdef int total
#    cdef int oldcount
#    cdef int inputimgsize1d =  inputimg.shape[0]*inputimg.shape[1]
#    cdef int arraysize = 256
##    create array for counting the intensity values
#    cdef np.ndarray countarray = np.zeros([1, arraysize], dtype = np.int]) 
#    cdef np.ndarray inputimg1d = np.zeros([1, inputimgsize1d], dtype = np.int)
#    cdef np.ndarray indexarray1d = np.zeros([1,inputimgsize1d], dtype = np.int)
#    cdef np.ndarray indexarray = np.zeros([inputimg.shape[0],inputimg.shape[1] ], dtype = np.int)
#    cdef np.ndarray sortedarray1d = np.zeros([1,inputimgsize1d], dtype = np.int)
#    cdef np.ndarray sortedarray = np.zeros([inputimg.shape[0],inputimg.shape[1] ], dtype = np.int)
#
#    inputimg1d = inputimg.reshape((1,inputimgsize1d))
#    
##    create histogram of intensities
#    
#    for ii in inputimg1d:
#        countarray[ii] += 1
#
##    cumulative hitogram
#    total = 0
#    for ii in np.arange(arraysize-1,-1,-1):
#        oldcount = countarray[ii]
#        countarray[ii] = total
#        total += oldcount
#        
##   do sorting and return value and index arrays   
#    for ind,ii in enumerate(a[::-1]):
#        sortedarray1d[countarray[ii]] = ii
##        indexarray1d[ind] = countarray[ii]        
#        countarray[ii] += 1
#        
##    reshape arrays
#    sortedarray = sortedarray1d.reshape((inputimg.shape[0],inputimg.shape[1] ))
#    
#    return(sortedarray)


