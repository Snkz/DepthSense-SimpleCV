#!/bin/python2

import DepthSense as ds
import numpy as np
from SimpleCV import *
c = 0
ds.initDepthSense()
while True:
    depth = ds.getDepthMap()
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>=2
    depth = depth.astype(np.uint8)
    iD = Image(depth.transpose())
    #iD.show()


    vertex = ds.getVertices()
    iV = Image(vertex.transpose([1,0,2]))
    #iV.show()


    image = ds.getColourMap()
    image = image[:,:,::-1]
    iS = Image(image.transpose([1,0,2]))
    iV.sideBySide(iD).show()

    #iS.show()
    ##dblobs = iD.findBlobs()
    ##if dblobs:
    ##    dblobs.draw()
    #c+=1
    #iD.show()
    #if c > 10000:
    #    ds.killDepthSense()
    #    break

