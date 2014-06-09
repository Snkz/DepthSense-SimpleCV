/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            depthsense.cxx
 * author:          Abdi Dahir
 * modified:        May 9 2014
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:
 * 
 * Python hooks happen here. This is the main file.
 * -----------------------------------------------------------------------------
 */

// Python Module includes
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <python2.7/Python.h>
#include <python2.7/numpy/arrayobject.h>

// MS completly untested
#ifdef _MSC_VER
#include <windows.h>
#endif

// C includes
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

// C++ includes
#include <exception>
#include <iostream>
#include <fstream>
//#include <thread>

// Application includes
#include "initdepthsense.h"
#include "imageproccessing.h"

// internal map copies
uint8_t colourMapClone[640*480*3];
int16_t depthMapClone[320*240];
int16_t vertexMapClone[320*240*3];
float accelMapClone[3];
float uvMapClone[320*240*2];
float vertexFMapClone[320*240*3];
uint8_t syncMapClone[320*240*3];
int16_t nPrintMap[320*240*3];
int16_t vPrintMap[320*240*3];

uint8_t depthColouredMapClone[320*240*3];
int16_t dConvolveResultClone[320*240];
uint8_t cConvolveResultClone[640*480];
uint8_t greyResultClone[640*480];
int16_t normalResultClone[320*240*3];

using namespace std;

// Minor processing (kind of hard to move)
static void saveMap(char *map, char* file)
{
    (void) map;
    //TODO: Make this a not shitty and specific function
    //TODO: memcpy and loop based on name in map? 
    ofstream f;
    f.open(file);
    cout << "Writing to file: " << file << endl;
    int16_t vx; int16_t vy; int16_t vz;
    int16_t nx; int16_t ny; int16_t nz;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            nx = nPrintMap[i*dW*3 + j*3 + 0];
            ny = nPrintMap[i*dW*3 + j*3 + 1];
            nz = nPrintMap[i*dW*3 + j*3 + 2];

            vx = vPrintMap[i*dW*3 + j*3 + 0];
            vy = vPrintMap[i*dW*3 + j*3 + 1];
            vz = vPrintMap[i*dW*3 + j*3 + 2];

            if (vz != 32001) {
                f << vx << "," << vy << "," << vz << endl;
                f << nx << "," << ny << "," << nz << endl;
            }
        }
    }

    cout << "Complete!" << endl;
    f.close();
}

void getMedian(int ci, int cj, uint8_t *colx, uint8_t *coly, uint8_t *colz) 
{
    int blkthresh = 4;
    int blklim = 25;
    int blkcount = 0;
    int count = 0;

    uint8_t cx;
    uint8_t cy;
    uint8_t cz;

    uint32_t tcx = 0;
    uint32_t tcy = 0;
    uint32_t tcz = 0;

    // note this meth is a lie
    for(int i=-1; i < 1; i++) {
        for(int j=-1; j < 1; j++) {
            if (ci - i > 0 && ci - i < cH && cj - j > 0 && cj - j < cW) {

                cx = colourMapClone[(ci - i)*cW*3 + (cj - j)*3 + 0];
                cy = colourMapClone[(ci - i)*cW*3 + (cj - j)*3 + 1];
                cz = colourMapClone[(ci - i)*cW*3 + (cj - j)*3 + 2];

                if ( cx < blklim && cy < blklim && cz < blklim) {
                    blkcount++;
                    continue;
                } 

                count++;
                tcx += cx;
                tcy += cy;
                tcz += cz;
            }
        }
    }

    if (blkcount > blkthresh || count == 0) {
        *colx = 0;
        *coly = 0;
        *colz = 0;
        return;
    }

    *colx = (uint8_t)(tcx/(count));
    *coly = (uint8_t)(tcy/(count));
    *colz = (uint8_t)(tcz/(count));
}

void buildSyncMap()
{
    int ci, cj;
    uint8_t *colx = (uint8_t *) malloc(sizeof(uint8_t));
    uint8_t *coly = (uint8_t *) malloc(sizeof(uint8_t));
    uint8_t *colz = (uint8_t *) malloc(sizeof(uint8_t));
    float uvx;
    float uvy;

    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            uvx = uvMapClone[i*dW*2 + j*2 + 0];    
            uvy = uvMapClone[i*dW*2 + j*2 + 1];    

            *colx = 0;
            *coly = 0;
            *colz = 0;
            
            if((uvx > 0 && uvx < 1 && uvy > 0 && uvy < 1) && 
                (depthMapClone[i*dW + j] < 32000)){
                ci = (int) (uvy * ((float) cH));
                cj = (int) (uvx * ((float) cW));

                getMedian(ci, cj, colx, coly, colz);
            }
          
            syncMapClone[i*dW*3 + j*3 + 0] = *colx;
            syncMapClone[i*dW*3 + j*3 + 1] = *coly;
            syncMapClone[i*dW*3 + j*3 + 2] = *colz;

        }
    }
}

void buildDepthColoured() 
{
    float hue;
    float sat;
    float val;

    float frac; float p; float q; float t;

    int hueIndex; float hueRound;

    uint8_t r; uint8_t g; uint8_t b;

    uint32_t colour;

    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            //TODO: Make this not complete shit
            hue = (log2(((float)depthCMap[i*dW + j]) / 32001.0) / -15.0);
            //hue = ((float)depthCMap[i*dW + j]) / 32001.0;
            sat = 0.6;
            val = 1.0;
            
            hueRound = hue * 6.0;
            hueIndex = (int) (((int)floor(hueRound)) % 6);
            frac = hueRound - floor(hueRound);

            p = val * (1.0 - sat);
            q = val * (1.0 - frac * sat);
            t = val * (1.0 - (1.0 - frac) * sat);

            if (hueIndex == 0) {
                r = (uint8_t) (val * 255);
                g = (uint8_t) (t * 255);
                b = (uint8_t) (p * 255);
            } else if (hueIndex == 1) {
                r = (uint8_t) (q * 255);
                g = (uint8_t) (val * 255);
                b = (uint8_t) (p * 255);
            } else if (hueIndex == 2) {
                r = (uint8_t) (p * 255);
                g = (uint8_t) (val * 255);
                b = (uint8_t) (t * 255);
            } else if (hueIndex == 3) {
                r = (uint8_t) (p * 255);
                g = (uint8_t) (q * 255);
                b = (uint8_t) (val * 255);
            } else if (hueIndex == 4) {
                r = (uint8_t) (t * 255);
                g = (uint8_t) (p * 255);
                b = (uint8_t) (val * 255);
            } else if (hueIndex == 5) {
                r = (uint8_t) (val * 255);
                g = (uint8_t) (p * 255);
                b = (uint8_t) (q * 255);
            } else  {
                r = 255;
                g = 255;
                b = 255;
            }

            depthColouredMap[i*dW*3 + j*3 + 0] = r;
            depthColouredMap[i*dW*3 + j*3 + 1] = g;
            depthColouredMap[i*dW*3 + j*3 + 2] = b;

            //cout << hueIndex << " " << r << " " << g << " " << b << endl;
            //cout << "fracts" << " " << p << " " << q << " " << t << " " << hue << " " << frac << endl;

        }
    }
}

// Python Callbacks
static PyObject *getColour(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {cH, cW, 3};

    memcpy(colourMapClone, colourFullMap, cshmsz*3);
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, colourMapClone);
}

static PyObject *getDepth(PyObject *self, PyObject *args)
{
    npy_intp dims[2] = {dH, dW};

    memcpy(depthMapClone, depthFullMap, dshmsz);
    return PyArray_SimpleNewFromData(2, dims, NPY_INT16, depthMapClone);
}

static PyObject *getGreyScale(PyObject *self, PyObject *args)
{

    npy_intp dims[2] = {cH, cW};

    memcpy(greyColourMap, colourFullMap, cshmsz*3);
    memset(greyResult, 0, cshmsz);
    toGreyScale(0.2126, 0.7152, 0.0722);
    memcpy(greyResultClone, greyResult, cshmsz);

    return PyArray_SimpleNewFromData(2, dims, NPY_UINT8, greyResultClone);
}

/* TODO: extract this bad boy */
static PyObject *getDepthColoured(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {dH, dW, 3};

    memcpy(depthCMap, depthFullMap, dshmsz);
    memset(depthColouredMap, 0, hshmsz*3);

    buildDepthColoured(); 
    memcpy(depthColouredMapClone, depthColouredMap, hshmsz*3);
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, depthColouredMapClone);
}

static PyObject *getAccel(PyObject *self, PyObject *args)
{
    npy_intp dims[1] = {3};

    memcpy(accelMapClone, accelFullMap, 3*sizeof(float));
    return PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, accelMapClone);
}

static PyObject *getVertex(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {dH, dW, 3};
    memcpy(vertexMapClone, vertexFullMap, vshmsz*3);
    return PyArray_SimpleNewFromData(3, dims, NPY_INT16, vertexMapClone);
}

static PyObject *getVertexFP(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {dH, dW, 3};
    memcpy(vertexFMapClone, vertexFFullMap, ushmsz*3);
    return PyArray_SimpleNewFromData(3, dims, NPY_FLOAT32, vertexFMapClone);
}

static PyObject *getUV(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {dH, dW, 2};
    memcpy(uvMapClone, uvFullMap, ushmsz*2);
    return PyArray_SimpleNewFromData(3, dims, NPY_FLOAT32, uvMapClone);
}

static PyObject *getSync(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {dH, dW, 3};

    memcpy(uvMapClone, uvFullMap, ushmsz*2);
    memcpy(colourMapClone, colourFullMap, cshmsz*3);
    memcpy(depthMapClone, depthFullMap, dshmsz);
    
    buildSyncMap();
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, syncMapClone);
}


static PyObject *initDS(PyObject *self, PyObject *args)
{
    initds();
    return Py_None;
}

static PyObject *killDS(PyObject *self, PyObject *args)
{
    killds();
    return Py_None;
}

/* TODO: Make this actually work, refer to checkHood.cxx */
static PyObject *getBlob(PyObject *self, PyObject *args)
{
    int i;
    int j;
    double thresh_high;
    double thresh_low;

    if (!PyArg_ParseTuple(args, "iidd", &i, &j,  &thresh_high, &thresh_low))
        return NULL;

    //npy_intp dims[2] = {dH, dW};

    //memcpy(blobMap, depthFullMap, dshmsz);
    // SHUTDOWN findBlob(i, j, thresh_high, thresh_low); 
    //memcpy(blobResultClone, blobResult, dshmsz);
    //return PyArray_SimpleNewFromData(2, dims, NPY_INT16, blobResultClone);
    Py_RETURN_NONE;
    
}

static PyObject *convolveDepth(PyObject *self, PyObject *args)
{
    char *kern;
    int repeat;
    double bias;

    if (!PyArg_ParseTuple(args, "sid", &kern, &repeat, &bias))
        return NULL;

    memcpy(dConvolveMap, depthFullMap, dshmsz);
   
    for(int i = 0; i < repeat; i++) {
        applyKernelDepth(kern, dW, dH, bias); 
        memcpy(dConvolveMap, dConvolveResult, dshmsz);
    } 

    npy_intp dims[2] = {dH, dW};
    memcpy(dConvolveResultClone, dConvolveResult, dshmsz);
    return PyArray_SimpleNewFromData(2, dims, NPY_INT16, dConvolveResultClone);
}

static PyObject *convolveColour(PyObject *self, PyObject *args)
{
    char *kern;
    int repeat;
    double bias;

    if (!PyArg_ParseTuple(args, "sid", &kern, &repeat, &bias))
        return NULL;

    memcpy(greyColourMap, colourFullMap, cshmsz*3);
    memset(greyResult, 0, cshmsz);
    toGreyScale(0.2126, 0.7152, 0.0722);
    memcpy(cConvolveMap, greyResult, cshmsz);
    
    for(int i = 0; i < repeat; i++) {
        applyKernelColour(kern, cW, cH, bias); 
        memcpy(cConvolveMap, cConvolveResult, cshmsz);
    } 

    npy_intp dims[2] = {cH, cW};
    memcpy(cConvolveResultClone, cConvolveResult, cshmsz);
    return PyArray_SimpleNewFromData(2, dims, NPY_UINT8, cConvolveResultClone);
}

/* TODO: Make this only vertex map */
static PyObject *saveMap(PyObject *self, PyObject *args)
{
    char *map;
    char *file;

    if (!PyArg_ParseTuple(args, "ss", &map, &file))
        return NULL;

    /* TODO: choose if this part can become general or not */
    memcpy(vPrintMap, vertexFullMap, vshmsz*3);
    memcpy(nPrintMap, normalResult, dshmsz*3); 
    saveMap(map, file);

    Py_RETURN_NONE;
}

static PyObject *getNormal(PyObject *self, PyObject *args)
{
    memcpy(normalMap, vertexFullMap, vshmsz*3);
    double bias = 0.5;
    char * kern = (char*)"placeholder";
    computeDifferential(kern, bias); // applies sobel kernel to each plane 
    crossMaps(); // cross product on three planes, store in normalMap result

    npy_intp dims[3] = {dH, dW, 3};
    memcpy(normalResultClone, normalResult, dshmsz*3); 
    return PyArray_SimpleNewFromData(3, dims, NPY_INT16, normalResultClone);
}

static PyMethodDef DepthSenseMethods[] = {
    // GET MAPS
    {"getDepthMap",  getDepth, METH_VARARGS, "Get Depth Map"},
    {"getDepthColouredMap",  getDepthColoured, METH_VARARGS, "Get Depth Coloured Map"},
    {"getColourMap",  getColour, METH_VARARGS, "Get Colour Map"},
    {"getGreyScaleMap",  getGreyScale, METH_VARARGS, "Get Grey Scale Colour Map"},
    {"getVertices",  getVertex, METH_VARARGS, "Get Vertex Map"},
    {"getVerticesFP",  getVertexFP, METH_VARARGS, "Get Floating Point Vertex Map"},
    {"getNormalMap",  getNormal, METH_VARARGS, "Get Normal Map"},
    {"getUVMap",  getUV, METH_VARARGS, "Get UV Map"},
    {"getSyncMap",  getSync, METH_VARARGS, "Get Colour Overlay Map"},
    {"getAcceleration",  getAccel, METH_VARARGS, "Get Acceleration"},
    // CREATE MODULE
    {"initDepthSense",  initDS, METH_VARARGS, "Init DepthSense"},
    {"killDepthSense",  killDS, METH_VARARGS, "Kill DepthSense"},
    // PROCESS MAPS
    {"getBlobAt",  getBlob, METH_VARARGS, "Find blob at location in the depth map"},
    {"convolveDepthMap",  convolveDepth, METH_VARARGS, "Apply specified kernel to the depth map"},
    {"convolveColourMap",  convolveColour, METH_VARARGS, "Apply specified kernel to the image"},
    // SAVE MAPS
    {"saveMap",  saveMap, METH_VARARGS, "Save the specified map"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initDepthSense(void)
{
    (void) Py_InitModule("DepthSense", DepthSenseMethods);
    // Clean up forked process, attach it to the python exit hook
    (void) Py_AtExit(killds);
    import_array();
}

int main(int argc, char* argv[])
{

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName((char *)"DepthSense");

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initDepthSense();

    //initds(); //for testing

    return 0;
}
