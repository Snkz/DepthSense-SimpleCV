import DepthSense as ds
import numpy as np
from SimpleCV import Image

class DS325:
    ''' DepthSense camera class for simple cv '''

    def __init__(self, dim=[320,240]):
        # TODO: Pass in dim to init (currently defaults to 640, 480 for camera)
        # TODO: Allow for enabling camera/depth/accel independantly
        ''' The maps returned by the ds mod are not transposed, the maps
        used ''' 
        ds.initDepthSense()

    def saveMap(self, name, file_name):
        ''' Save the specified map to file file_name '''

        ds.saveMap(name, file_name);
        return

    def getDepth(self):
        ''' Return a simple cv compatiable 8bit depth image '''

        depth = ds.getDepthMap()
        np.clip(depth, 0, 2**10 - 1, depth)
        depth >>=2
        depth = depth.astype(np.uint8)
        iD = Image(depth.transpose())
        return iD.invert()

    def getNormal(self):
        ''' Return a simple cv compatiable 8bit depth image '''

        normal = ds.getNormalMap()
        #np.clip(normal, 0, 2**10 - 1, normal)
        #normal >>=2
        #normal = normal.astype(np.uint8)
        #normal = normal[:,:,::-1]
        return Image(normal.transpose([1,0,2]))

    def getBlob(self, i, j, thresh_high, thresh_low):
        ''' Return a simple cv compatiable 8bit depth image that contains only 
        the blob found at index i,j with depth values that are at most 
        +thresh_high or at least -thresh_low relative to the depth value at 
        i, j'''

        #blob = ds.getBlobAt(i,j, thresh_high, thresh_low)
        #np.clip(blob, 0, 2**10 - 1, blob)
        #blob >>=2
        #blob = blob.astype(np.uint8)
        #iB = Image(blob.transpose())
        #return iB.invert()


    def getConvolvedDepth(self, kern, rep, bias):
        ''' Return a simple cv compatiable 8bit depth map that has had 
        the specified kernel applied rep times '''

        conv = ds.convolveDepthMap(kern, rep, bias)
        np.clip(conv, 0, 2**10 - 1, conv)
        conv >>=2
        conv = conv.astype(np.uint8)
        iE = Image(conv.transpose())
        return iE.invert()

    def getDepthFull(self):
        ''' Return the pure 16bit depth map as a numpy array '''
         
        depth = ds.getDepthMap()
        depth = depth/125
        iD = Image(depth.transpose())
        return iD

    def getConvolvedDepthFull(self, kern, rep, bias):
        ''' Return a pure numpy array of the convolved in the depthmap ''' 

        edge = ds.convolveDepthMap(kern, rep, bias)
        edge = edge/125
        iE = Image(edge.transpose())
        return iE

    def getVertex(self):
        ''' Return a vertex map for points in the depth map as a numpy array'''

        return ds.getVertices()

    def getVertexFP(self):
        ''' Return a floating point vertex map for points in the depth map as 
        a numpy array'''

        return ds.getVerticesFP()

    def getImage(self):
        ''' Return a simple cv compatiable 8bit colour image ''' 

        image = ds.getColourMap()
        image = image[:,:,::-1]
        return Image(image.transpose([1,0,2]))

    def getGreyScale(self):
        ''' Return a simple cv compatiable 8bit colour image ''' 

        grey = ds.getGreyScaleMap()
        return Image(grey.transpose())

    def getConvolvedImage(self, kern, rep, bias):
        ''' Return a simple cv compatiable 8bit greyscaled image that has had 
        the specified kernel applied rep times with bias supplied'''

        conv = ds.convolveColourMap(kern, rep, bias)
        iE = Image(conv.transpose())
        return iE.invert()

    def getDepthColoured(self):
        ''' Return a simple cv compatiable 8bit colour image ''' 

        depthc = ds.getDepthColouredMap()
        depthc = depthc[:,:,::-1]
        return Image(depthc.transpose([1,0,2]))


    def getAcceleration(self):
        ''' Return the current acceleration of the device (x,y,z) measured in 
        g force as a numpy array '''

        return ds.getAcceleration()

    def getUV(self):
        ''' Return a uv map, the map represents a conversion from depth 
        indicies to scaled colour indicies from 0 - 1 '''

        return ds.getUVMap()

    def getSync(self):
        ''' Return a simplecv compatiable synced map with dimensions of the 
        depth map and colours from the colour map. Indexes here match indexes 
        in the vertex map '''

        sync = ds.getSyncMap()
        sync = sync[:,:,::-1]
        return Image(sync.transpose([1,0,2]))

    def getSyncFull(self):
        ''' Return a colour synced depth map, like above indicies here work on
        both the depth map and vertex map '''

        return ds.getSyncMap()

