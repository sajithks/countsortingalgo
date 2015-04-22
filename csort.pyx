from __future__ import division
import numpy as np
import cv2
import scipy as sp
import sys
sys.path.append('/Users/sajithks/Documents/project_cell_tracking/phase images from elf/ecoli_det_seg')
import createbacteria  as cb
import skimage.morphology as skmorph
from scipy.ndimage import label
cimport numpy as np
cimport cython




#from scipy.ndimage import label
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t

#################


@cython.boundscheck(False)
def cenMoment(np.ndarray[DTYPE_t, ndim=2] boundcoord, p, q):
    '''
    cenMoment(np.ndarray[DTYPE_t, ndim=2] boundcoord, p, q) -> p,q th order central moment
    '''
    return(np.sum((boundcoord[:,0]-boundcoord[:,0].mean())**p * (boundcoord[:,1]-boundcoord[:,1].mean())**q))
    
@cython.boundscheck(False)
def ellipseParam(np.ndarray[DTYPE_t, ndim=2] boundcoord):
    
    cdef float area = 0.0
    cdef float perimeter = 0.0
    cdef float orientation = 0.0
    cdef float majoraxis = 0.0
    cdef float minoraxis = 0.0
    cdef float m00, m11, m02, m20, a, b, circumference  
    cdef int maxindex, minindex
    
    cdef np.ndarray params = np.zeros([5,1], dtype = np.float)
    cdef np.ndarray eigvals = np.zeros([1, 2], dtype=np.float)    
    
    m00 = cenMoment(boundcoord, 0, 0)
    m11 = cenMoment(boundcoord, 1, 1)
    m02 = cenMoment(boundcoord, 0, 2)
    m20 = cenMoment(boundcoord, 2, 0)

    eigvals, eigvec = np.linalg.eig([[m02,m11],[m11,m20]])
    
    maxindex = np.argmax(eigvals)
    minindex = np.argmin(eigvals)
    
    majoraxis = 4*(eigvals[maxindex]/m00)**.5
    minoraxis = 4*(eigvals[minindex]/m00)**.5
    
    a = majoraxis/2
    b = minoraxis/2
    
    area = np.pi*a*b
    circumference = np.pi*(3*(a+b)- np.sqrt(10*a*b + 3*(a**2 + b**2)))
    orientation = 0.5*np.math.atan2(2*m11,(m20-m02))  

    
    params[0] = majoraxis
    params[1] = minoraxis
    params[2] = area
    params[3] = circumference
    params[4] = orientation
    
#    orientation = 0.5*np.math.atan2((m20-m02),2*m11)  
    
#    orientation =  np.math.degrees(np.arctan(2*m11/(m20-m02))/2)
    return(params)
##    assert data.dtype == DTYPE


#%% find eigen image of smoothed hessian
@cython.boundscheck(False)
def findEigImage(np.ndarray[np.float64_t, ndim=2]smoothimg):

    cdef int ii, jj
    
    cdef np.ndarray Ix = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float)    
    cdef np.ndarray Iy = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float)    
    cdef np.ndarray Ixx = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float)    
    cdef np.ndarray Ixy = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float)    
    cdef np.ndarray Iyy = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float)    
    cdef np.ndarray eigh = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float) 
    cdef np.ndarray etemp = np.zeros([smoothimg.shape[0], smoothimg.shape[1]], dtype=np.float)    
       
    cdef np.ndarray temp2 = np.zeros((2,1), dtype=np.float)    
    
    smoothimg = np.float64(smoothimg)
    Iy, Ix =np.gradient(smoothimg)
    Ixy, Ixx = np.gradient(Ix)
    Iyy, Ixy = np.gradient(Iy)
    
    for ii in range(Ix.shape[0]):
        for jj in range(Ix.shape[1]):
            temp2 = np.linalg.eigvalsh([[Ixx[ii,jj],Ixy[ii,jj]],[Ixy[ii,jj], Iyy[ii,jj]]])
            temp2.sort()
            eigh[ii,jj] = temp2[0]
      
    
    #cb.myshow2(eigh1)
    etemp = np.copy(eigh)
    etemp = etemp-etemp.min()
    etemp = etemp/etemp.max()
    etemp = np.uint8(etemp*255)
    return(etemp)
    
@cython.boundscheck(False)
def findMaxOccurance(data):
    '''
    findMaxOccurance(data)-->maxoccur
    finds the element which repeats maximum
    '''
    uniqdata = np.unique(data)
    uniqdata = uniqdata[uniqdata>0]
    sizelist = []
    for ii in range(uniqdata.shape[0]):
        sizelist.append(np.sum(data==uniqdata[ii]))
    maxoccur = uniqdata[np.argmax(sizelist)]
    return(maxoccur)



# test part

@cython.boundscheck(False)
def segImage(np.ndarray[np.float64_t, ndim=2]etemp, np.ndarray[np.float64_t, ndim=2]signsum2):

    # constants
    cdef int LEVELS = 15
    # variables
    
    cdef int ii, jj, peakval, startinten, stopinten, ncc, cellcnt
    #arrays
    cdef np.ndarray histo    
    cdef np.ndarray labelimgarray = np.zeros((etemp.shape[0], etemp.shape[1], LEVELS), dtype=np.float)
    
    cdef np.ndarray inten
    #find histogram peak to detect start and end intensities
    histo, inten = np.histogram(etemp,255)
    peakval = inten[np.argwhere(histo == histo.max())[0][0]]
    startinten = peakval - LEVELS
    stopinten = peakval
    
    
    
    
    for ii in range(LEVELS):
        labelimgarray[:,:,ii], ncc = label((etemp>=(ii+startinten)),np.ones((3,3))) 
    
    #    cb.myshow2(((etemp>ii))*1+cellreg*2)
    #cb.myshow2(threshimg)
    #%%
    cdef np.ndarray cellreg = np.zeros((etemp.shape[0], etemp.shape[1]), dtype=np.int)    
    cdef np.ndarray cellreglab = np.zeros((etemp.shape[0], etemp.shape[1]), dtype=np.int)    
    cdef np.ndarray uniqout 
    
    cellreg = (signsum2<signsum2.min()*.8)
    
    cellreg = skmorph.remove_small_objects(cellreg,5)
    cellreglab, cellcnt = label(cellreg, np.ones((3,3)))
    
    uniqout = np.unique(np.concatenate((cellreglab[0,:], cellreglab[-1,:], cellreglab[:,0],cellreglab[:,-1])))
    for ii in range(uniqout.shape[0]):
        cellreglab[cellreglab==uniqout[ii]] = 0
    
    ##cb.myshow2(cellreglab)
    #
    cdef np.ndarray cellab 

    cellab = np.unique(cellreglab)
    cellab = cellab[cellab>0]
    
   
    labwtthreslist =[]
    
    #mainseeds = []    
    cdef np.ndarray indivlabwtthresh
    cdef np.ndarray uniqseeds
    cdef np.ndarray params    
    cdef np.ndarray visitedarray = np.zeros((cellab.shape[0],1), dtype = np.int)
#    cdef np.ndarray labwtthreslist = np.zeros((cellab.shape[0],4), dtype = np.int)
    
    cdef np.ndarray singlereg = np.zeros((etemp.shape[0], etemp.shape[1]), dtype=np.int) 
    cdef np.ndarray threshlab = np.zeros((etemp.shape[0], etemp.shape[1]), dtype=np.float)
    
    
    cdef float currentweight,indivarea,elipsarea,indivperim, elipsperim,residual,residuarearatio,residuperimratio
    cdef float convexity,weightval
    cdef int threshval, indivthreslab,vseed,togetherimgsum
    
    
    
    for seed in range(cellab.shape[0]):
        currentseed = cellab[seed]    
        if(visitedarray[seed] == 0):        
            visitedarray[seed] = 1 
            indivlabwtthresh = np.zeros((1,3))
            currentweight = 0
    #        mainseeds.append(currentseed)
    #        visitedarray[currentseed -1] = 1
            #for loop here 
            for threshval in range(LEVELS):
                threshlab = labelimgarray[:,:,threshval]
                togetherimgsum = np.sum(threshlab*(cellreglab == currentseed))
                
                if(togetherimgsum>1):
        #            threshlab, nlab = label(threshimg <= threshval, np.ones((3,3))) #change threshold here
                    indivthreslab = cb.findMaxOccurance(labelimgarray[:,:,threshval]*(cellreglab == currentseed))
                    singlereg = labelimgarray[:,:,threshval] == indivthreslab
                    
                    if(np.sum(singlereg)>100 and np.sum(singlereg)<5000) :           
                    
                        uniqseeds = np.unique(singlereg*cellreglab)
                        uniqseeds = uniqseeds[uniqseeds>0]
                        
                        
                        #weight calculation
        #                ellipseimg, param = cb.fitEllipseImage(singlereg)
                        params = ellipseParam(np.int64(np.argwhere(singlereg==1)))                
                        
                        indivarea = np.sum(np.float32(singlereg))
                        elipsarea = params[2]                
                                                                
                        indivperim = np.sum(np.float32(singlereg - cv2.erode(np.uint8(singlereg) ,np.ones((3,3))) ))                    
                        elipsperim = params[3]
                                                                                
                        residual = np.abs(indivarea - elipsarea ) 
                        residuarearatio = np.min([indivarea,  elipsarea] )/np.max([indivarea,  elipsarea] )
                            
                        residuperimratio = np.min([indivperim,  elipsperim] )/np.max([indivperim,  elipsperim] )              
                        convexity = indivarea/cb.findConvexarea(singlereg)
                        weightval = 0.5*residuperimratio + 0.5*convexity
#                        print weightval
        #                weightval = convexity
                        if params[1] >6 and params[1]<35 and params[0] <400 and params[0]>25 and weightval> currentweight:
                            currentweight = weightval
                            indivlabwtthresh[0,0] = currentseed
                            indivlabwtthresh[0,1] = currentweight
                            indivlabwtthresh[0,2] = threshval  
                                                
            if(indivlabwtthresh[0][0] != 0 ):
                labwtthreslist.append(indivlabwtthresh)
                for vseed in range(uniqseeds.shape[0]):
                    visitedarray[np.argwhere(cellab==uniqseeds[vseed])[0][0]] = 1

    #        print 'array size ',np.size(labwtthreslist),' ccnt',ccnt
            print seed ,' of ', cellab.shape[0]
    #        ccnt += 1
    
    #%%
    #start = time.time()
    cdef np.ndarray outimg = np.zeros_like(etemp)
    cdef np.ndarray newsegment = np.zeros_like(etemp)
    cdef int existlab
    cdef float existarea,newsegarea
    
    for ii in range(np.shape(labwtthreslist)[0]):
        newsegment = labelimgarray[:, :, np.int64(labwtthreslist[ii][0][2])] == cb.findMaxOccurance(labelimgarray[:, :, np.int64(labwtthreslist[ii][0][2])] * (cellreglab == np.int64(labwtthreslist[ii][0][0]) ) ) 
        newsegment = sp.ndimage.binary_fill_holes(newsegment)    
        if(np.sum((outimg>0)*newsegment)>0):    
#            tlabel, tnum = label(outimg, np.ones((3,3)))
            existlab = cb.findMaxOccurance(outimg*newsegment)
            existarea = np.sum(outimg == existlab)
            newsegarea = np.sum(newsegment)
            if(newsegarea>existarea):
                outimg[outimg==existlab] = 0
                outimg = outimg + ii*newsegment
        else:        
            outimg = outimg + ii*newsegment
    #    print ii,' of ',np.shape(labwtthreslist)[0]
    
    return(outimg)

