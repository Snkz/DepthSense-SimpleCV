#!/bin/python2
from ds325 import DS325
import sys
from SimpleCV import *
from threading import Thread 
from numpy import copy
#c = 0
vertex = None
def listen():
    while True:
        data = sys.stdin.readline()
        if "save" in data:
            depthsense.saveMap("vertex", "vertex_data.asc")


# dum
depthsense = DS325()
t = Thread(target=listen)
t.daemon = True
t.start()
while True:
    #iB = depthsense.getBlob(100,100, 5, 5)

    #iC = depthsense.getImage()
    #iD = depthsense.getDepth() 
    #iE = depthsense.getConvolvedDepth("lapl", 1, 0.5)

    #iDH = depthsense.getDepthFull()
    #iEH = depthsense.getConvolvedDepthFull("edgh", 1, 0.5)

    #iH = depthsense.getDepthColoured()

    #iH.show()
    #iDH.sideBySide(iEH).show()
    #iD.sideBySide(iE).show()
    #iC.sideBySide(iE).show()
    vertex = depthsense.getVertex()
    iV = Image(vertex.transpose([1,0,2]))
    iV.show()

