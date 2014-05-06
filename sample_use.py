#!/bin/python2
from SimpleCV import *
from ds325 import DS325

depthsense = DS325()
while True:

    # THESE THREE ARE SIMPLECV IMAGE 
    iD = depthsense.getDepth() 
    #iD.show()

    iS = depthsense.getImage()
    #iS.show()

    iY = depthsense.getSync()
    #iY.show()
    
    iH = depthsense.getDepthColoured()
    #iH.show() 
    
    iG = depthsense.getGreyScale()
    #iG.show() 

    iE = depthsense.getConvolvedDepth("sobx", 1)
    #iE.show()
    
    iC = depthsense.getConvolvedImage("edge", 1, 0)
    #iC.show()

    # THESE THREE ARE NOT SIMPLECV IMAGES BY DEFAULT 
    vertex = depthsense.getVertex()
    # vertex map does not get returned as an image as it makes no sense
    iV = Image(vertex.transpose([1,0,2]))
    #iV.show()

    # accel is not returned as a simplecv image
    iA = depthsense.getAcceleration()
    # uv map is not returned as a simplecv image
    iU = depthsense.getUV()


    #iS.sideBySide(iV).show()
    iC.sideBySide(iG).show()
    #iD.sideBySide(iY).show()
    #iH.sideBySide(iE).show()

