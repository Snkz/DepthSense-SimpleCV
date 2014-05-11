/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            depthsense.cxx
 * author:          Abdi Dahir
 * modified:        May 9 2014
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:
 * 
 * Python and DepthSense hooks happen here. This is the main file.
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
#include <sys/wait.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

// C++ includes
#include <vector>
#include <exception>
#include <iostream>
#include <fstream>
//#include <thread>

// DepthSense SDK includes
#include <DepthSense.hxx>

// Application includes
#include "imagebuffers.h"
#include "imageproccessing.h"

using namespace DepthSense;
using namespace std;

// depth sense node inits
static Context g_context;
static DepthNode g_dnode;
static ColorNode g_cnode;
static AudioNode g_anode;

static bool g_bDeviceFound = false;

// unecassary frame counters
static uint32_t g_aFrames = 0;
static uint32_t g_cFrames = 0;
static uint32_t g_dFrames = 0;

int16_t *depthMap; 
int16_t *depthFullMap; 

int16_t *vertexMap; 
int16_t *vertexFullMap; 

uint8_t *colourMap; 
uint8_t *colourFullMap; 

float *uvMap; 
float *uvFullMap; 

float *vertexFMap; 
float *vertexFFullMap; 

float *accelMap; 
float *accelFullMap; 

// clean up
int child_pid = 0;

// can't write atomic op but i can atleast do a swap
static void dptrSwap (int16_t **pa, int16_t **pb){
        int16_t *temp = *pa;
        *pa = *pb;
        *pb = temp;
}

static void cptrSwap (uint8_t **pa, uint8_t **pb){
        uint8_t *temp = *pa;
        *pa = *pb;
        *pb = temp;
}

static void aptrSwap (float **pa, float **pb){
        float *temp = *pa;
        *pa = *pb;
        *pb = temp;
}

static void vptrSwap (int16_t **pa, int16_t **pb){
        int16_t *temp = *pa;
        *pa = *pb;
        *pb = temp;
}

/*----------------------------------------------------------------------------*/
// New audio sample event handler
static void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    //printf("A#%u: %d\n",g_aFrames,data.audioData.size());
    g_aFrames++;
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
static void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{
    //printf("C#%u: %d\n",g_cFrames,data.colorMap.size());
    memcpy(colourMap, data.colorMap, 3*cshmsz);
    cptrSwap(&colourMap, &colourFullMap);
    g_cFrames++;
}

/*----------------------------------------------------------------------------*/
// New depth sample event handler
static void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    // Depth
    memcpy(depthMap, data.depthMap, dshmsz);
    dptrSwap(&depthMap, &depthFullMap);

    // Verticies
    Vertex vertex;
    FPVertex fvertex;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            vertex = data.vertices[i*dW + j];
            fvertex = data.verticesFloatingPoint[i*dW + j];

            vertexMap[i*dW*3 + j*3 + 0] = vertex.x;
            vertexMap[i*dW*3 + j*3 + 1] = vertex.y;
            vertexMap[i*dW*3 + j*3 + 2] = vertex.z;

            vertexFMap[i*dW*3 + j*3 + 0] = fvertex.x;
            vertexFMap[i*dW*3 + j*3 + 1] = fvertex.y;
            vertexFMap[i*dW*3 + j*3 + 2] = fvertex.z;
            //cout << vertex.x << vertex.y << vertex.z << endl;
            //cout << fvertex.x << fvertex.y << fvertex.z << endl;

        }
    }
    vptrSwap(&vertexMap, &vertexFullMap);
    aptrSwap(&vertexFMap, &vertexFFullMap);

    // uv
    UV uv;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            uv = data.uvMap[i*dW + j];
            uvMap[i*dW*2 + j*2 + 0] = uv.u;
            uvMap[i*dW*2 + j*2 + 1] = uv.v;
            //cout << uv.u << uv.v << endl;

        }
    }
    aptrSwap(&uvMap, &uvFullMap);


    // Acceleration
    accelMap[0] = data.acceleration.x;
    accelMap[1] = data.acceleration.y;
    accelMap[2] = data.acceleration.z;
    aptrSwap(&accelMap, &accelFullMap);

    g_dFrames++;
}

/*----------------------------------------------------------------------------*/
static void configureAudioNode()
{
    g_anode.newSampleReceivedEvent().connect(&onNewAudioSample);

    AudioNode::Configuration config = g_anode.getConfiguration();
    config.sampleRate = 44100;

    try
    {
        g_context.requestControl(g_anode,0);

        g_anode.setConfiguration(config);

        g_anode.setInputMixerLevel(0.5f);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
static void configureDepthNode()
{
    g_dnode.newSampleReceivedEvent().connect(&onNewDepthSample);

    DepthNode::Configuration config = g_dnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_QVGA;
    config.framerate = 30;
    config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;
    config.saturation = true;

    try
    {
        g_context.requestControl(g_dnode,0);
        g_dnode.setConfidenceThreshold(100);

        g_dnode.setEnableDepthMap(true);
        g_dnode.setEnableVertices(true);
        g_dnode.setEnableVerticesFloatingPoint(true);
        g_dnode.setEnableAccelerometer(true);
        g_dnode.setEnableUvMap(true);

        g_dnode.setConfiguration(config);

    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
static void configureColorNode()
{

    // connect new color sample handler
    g_cnode.newSampleReceivedEvent().connect(&onNewColorSample);

    ColorNode::Configuration config = g_cnode.getConfiguration();
    config.frameFormat = FRAME_FORMAT_VGA;
    config.compression = COMPRESSION_TYPE_MJPEG;
    config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;
    config.framerate = 30;

    g_cnode.setEnableColorMap(true);

    try
    {
        g_context.requestControl(g_cnode,0);

        g_cnode.setConfiguration(config);
        g_cnode.setBrightness(0);
        g_cnode.setContrast(5);
        g_cnode.setSaturation(5);
        g_cnode.setHue(0);
        g_cnode.setGamma(3);
        g_cnode.setWhiteBalance(4650);
        g_cnode.setSharpness(5);
        g_cnode.setWhiteBalanceAuto(true);


    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
static void configureNode(Node node)
{
    if ((node.is<DepthNode>())&&(!g_dnode.isSet()))
    {
        g_dnode = node.as<DepthNode>();
        configureDepthNode();
        g_context.registerNode(node);
    }

    if ((node.is<ColorNode>())&&(!g_cnode.isSet()))
    {
        g_cnode = node.as<ColorNode>();
        configureColorNode();
        g_context.registerNode(node);
    }

    if ((node.is<AudioNode>())&&(!g_anode.isSet()))
    {
        g_anode = node.as<AudioNode>();
        configureAudioNode();
        // Audio seems to take up bandwith on usb3.0 devices ... we'll make this a param
        //g_context.registerNode(node);
    }
}

/*----------------------------------------------------------------------------*/
static void onNodeConnected(Device device, Device::NodeAddedData data)
{
    configureNode(data.node);
}

/*----------------------------------------------------------------------------*/
static void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
    if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == g_anode))
        g_anode.unset();
    if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == g_cnode))
        g_cnode.unset();
    if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == g_dnode))
        g_dnode.unset();
    printf("Node disconnected\n");
}

/*----------------------------------------------------------------------------*/
static void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
    if (!g_bDeviceFound)
    {
        data.device.nodeAddedEvent().connect(&onNodeConnected);
        data.device.nodeRemovedEvent().connect(&onNodeDisconnected);
        g_bDeviceFound = true;
    }
}

/*----------------------------------------------------------------------------*/
static void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    g_bDeviceFound = false;
    printf("Device disconnected\n");
}

/*----------------------------------------------------------------------------*/
/*                         Data processors                                    */
/*----------------------------------------------------------------------------*/

extern "C" {
    static void killds()
    {
        if (child_pid !=0) {
            kill(child_pid, SIGTERM);
            munmap(depthMap, dshmsz);
            munmap(depthFullMap, dshmsz);
            munmap(colourMap, cshmsz*3);
            munmap(colourFullMap, cshmsz*3);
            munmap(vertexMap, vshmsz*3);
            munmap(vertexFullMap, vshmsz*3);
            munmap(vertexFMap, ushmsz*3);
            munmap(vertexFFullMap, ushmsz*3);
            munmap(uvMap, ushmsz*2);
            munmap(uvMap, ushmsz*2);
            munmap(uvFullMap, ushmsz*2);
        }

    }
}

static void * initmap(int sz) 
{
    void * map;     
    if ((map = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED) {
        perror("mmap: cannot alloc shmem;");
        exit(1);
    }

    return map;
}

static void initds()
{
    // shared mem double buffers
    depthMap = (int16_t *) initmap(dshmsz); 
    depthFullMap = (int16_t *) initmap(dshmsz); 

    accelMap = (float *) initmap(3*sizeof(float)); 
    accelFullMap = (float *) initmap(3*sizeof(float)); 

    colourMap = (uint8_t *) initmap(cshmsz*3); 
    colourFullMap = (uint8_t *) initmap(cshmsz*3); 

    vertexMap = (int16_t *) initmap(vshmsz*3); 
    vertexFullMap = (int16_t *) initmap(vshmsz*3); 
    
    uvMap = (float *) initmap(ushmsz*2); 
    uvFullMap = (float *) initmap(ushmsz*2); 

    vertexFMap = (float *) initmap(ushmsz*3); 
    vertexFFullMap = (float *) initmap(ushmsz*3); 


    // kerns
    child_pid = fork();

    // child goes into loop
    if (child_pid == 0) {
        g_context = Context::create("localhost");
        g_context.deviceAddedEvent().connect(&onDeviceConnected);
        g_context.deviceRemovedEvent().connect(&onDeviceDisconnected);

        // Get the list of currently connected devices
        vector<Device> da = g_context.getDevices();

        // We are only interested in the first device
        if (da.size() >= 1)
        {
            g_bDeviceFound = true;

            da[0].nodeAddedEvent().connect(&onNodeConnected);
            da[0].nodeRemovedEvent().connect(&onNodeDisconnected);

            vector<Node> na = da[0].getNodes();

            //printf("Found %u nodes\n",na.size());

            for (int n = 0; n < (int)na.size();n++)
                configureNode(na[n]);
        }

        g_context.startNodes();
        g_context.run();
        g_context.stopNodes();

        if (g_cnode.isSet()) g_context.unregisterNode(g_cnode);
        if (g_dnode.isSet()) g_context.unregisterNode(g_dnode);
        if (g_anode.isSet()) g_context.unregisterNode(g_anode);

        exit(EXIT_SUCCESS);
    }

}

/* TODO: Make this only vertex map */
static void saveMap(char *map, char* file)
{
    (void) map;
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

/*----------------------------------------------------------------------------*/
/*                       Python Callbacks                                     */
/*----------------------------------------------------------------------------*/
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
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            //colour = ((uint32_t)depthCMap[i*dW + j]) * 524; // convert approx 2^15 bits of data to 2^24 bits  
            //colour = 16777216.0*(((double)depthCMap[i*dW + j])/31999.0); // 2^24 * zval/~2^15(zrange)
            //cout << depth << " " << depthCMap[i*dW + j] << endl;

            //depthColouredMap[i*dW*3 + j*3 + 0] = (uint8_t) ((colour << (32 - 8*1)) >> (32 - 8));
            //depthColouredMap[i*dW*3 + j*3 + 1] = (uint8_t) ((colour << (32 - 8*2)) >> (32 - 8));
            //depthColouredMap[i*dW*3 + j*3 + 2] = (uint8_t) ((colour << (32 - 8*3)) >> (32 - 8));

            depthColouredMap[i*dW*3 + j*3 + 0] = (uint8_t) (((depthCMap[i*dW + j] << (16 - 5*1)) >> (16 - 5)) << 3);
            depthColouredMap[i*dW*3 + j*3 + 1] = (uint8_t) (((depthCMap[i*dW + j] << (16 - 5*2)) >> (16 - 5)) << 3);
            depthColouredMap[i*dW*3 + j*3 + 2] = (uint8_t) (((depthCMap[i*dW + j] << (16 - 5*3)) >> (16 - 5)) << 3);

        }
    }

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


static PyObject *initDepthS(PyObject *self, PyObject *args)
{
    initds();
    return Py_None;
}

static PyObject *killDepthS(PyObject *self, PyObject *args)
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

    npy_intp dims[2] = {dH, dW};

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
    {"initDepthSense",  initDepthS, METH_VARARGS, "Init DepthSense"},
    {"killDepthSense",  killDepthS, METH_VARARGS, "Kill DepthSense"},
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
