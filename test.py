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
    depth = depth.astype(np.uint8).transpose()
    iD = Image(depth)

    image = ds.getColourMap()
    image = image[:,:,::-1]
    iS = Image(image.transpose([1,0,2]))
    iS.sideBySide(iD).show()
 
    c+=1
    if c > 10000:
        break

