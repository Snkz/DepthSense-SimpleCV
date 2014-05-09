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
#include <list>
//#include <thread>

// DepthSense SDK includes
#include <DepthSense.hxx>

// Application includes
#include "imagebuffers.h"

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


// kernels
static int edgeKern[9] = { 0,  1,  0, 
                           1, -4,  1, 
                           0,  1,  0 };

static int sharpKern[9] = {  0, -1,  0, 
                            -1,  5, -1, 
                             0, -1,  0 };

static int ogKern[9] = { 0, 0, 0, 
                         0, 1, 0, 
                         0, 0, 0 };

static int embossKern[9] = { -2, -1,  0, 
                             -1,  1,  1, 
                              0,  1,  2 };

static int edgeHighKern[9] = { -1, -1, -1, 
                               -1,  8, -1, 
                               -1, -1, -1 };

static int blurKern[9] = { 1,  2,  1,  // needs to be handled differently
                           2,  4,  2, 
                           1,  2,  1 };

static int sobelYKern[9] = {  1,  2,  1, 
                              0,  0,  0, 
                             -1, -2, -1 };

static int sobelXKern[9] = { -1, 0, 1, 
                             -2, 0, 2, 
                             -1, 0, 1 };

static int scharrXKern[9] = {  3, 0, -3 , 
                              10, 0, -10, 
                               3, 0, -3  };

static int scharrYKern[9] = {  3,  10,  3, 
                               0,   0,  0, 
                              -3, -10, -3  };

static int lapKern[9] = {   1,  -2,   1,  
                           -2,   4,  -2, 
                            1,  -2,   1 };

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

/*
 * Using (assumed to be) up-to-date depth/uv/colour maps build a colour map
 * with the resoloution of the depth map with pixels that exist in both the 
 * depth and colour map exclusively (that info is provided by the uv map)
 */
static void buildSyncMap()
{
    int ci, cj;
    uint8_t colx;
    uint8_t coly;
    uint8_t colz;
    float uvx;
    float uvy;

    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            uvx = uvMapClone[i*dW*2 + j*2 + 0];    
            uvy = uvMapClone[i*dW*2 + j*2 + 1];    
            colx = 0;
            coly = 0;
            colz = 0;
            
            if((uvx > 0 && uvx < 1 && uvy > 0 && uvy < 1) && 
                (depthMapClone[i*dW + j] < 32000)){
                ci = (int) (uvy * ((float) cH));
                cj = (int) (uvx * ((float) cW));
                colx = colourMapClone[ci*cW*3 + cj*3 + 0];
                coly = colourMapClone[ci*cW*3 + cj*3 + 1];
                colz = colourMapClone[ci*cW*3 + cj*3 + 2];
            }
          
            syncMapClone[i*dW*3 + j*3 + 0] = colx;
            syncMapClone[i*dW*3 + j*3 + 1] = coly;
            syncMapClone[i*dW*3 + j*3 + 2] = colz;

        }
    }
}

static void pickKern(char* kern, int kernel[9]) {

    if (strncmp(kern, "edge", 4) == 0) 
        memcpy(kernel, edgeKern, 9*sizeof(int) );
    if (strncmp(kern, "shrp", 4) == 0) 
        memcpy(kernel, sharpKern, 9*sizeof(int) );
    if (strncmp(kern, "iden", 4) == 0) 
        memcpy(kernel, ogKern, 9*sizeof(int) );
    if (strncmp(kern, "blur", 4) == 0) 
        memcpy(kernel, blurKern, 9*sizeof(int) );
    if (strncmp(kern, "sobx", 4) == 0) 
        memcpy(kernel, sobelXKern, 9*sizeof(int) );
    if (strncmp(kern, "soby", 4) == 0) 
        memcpy(kernel, sobelYKern, 9*sizeof(int) );
    if (strncmp(kern, "scrx", 4) == 0) 
        memcpy(kernel, scharrXKern, 9*sizeof(int) );
    if (strncmp(kern, "scry", 4) == 0) 
        memcpy(kernel, scharrYKern, 9*sizeof(int) );
    if (strncmp(kern, "embs", 4) == 0) 
        memcpy(kernel, embossKern, 9*sizeof(int) );
    if (strncmp(kern, "edgh", 4) == 0) 
        memcpy(kernel, edgeHighKern, 9*sizeof(int) );
    if (strncmp(kern, "lapl", 4) == 0) 
        memcpy(kernel, lapKern, 9*sizeof(int) );
}

/* TODO:
 * Three convoloution implementations, must be way to factor this nicely 
 * (unique array types make it difficult) 
 */

static int convolve(int i, int j, int kern[9], char *kernel, int W, int H, double bias) {
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)diffMap[i*W + j];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)diffMap[(i-1)*W + j];
    else
        edge = edge + kern[0*w + 1] * (int)diffMap[(i-0)*W + j]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)diffMap[(i+1)*W + j];
    else
        edge = edge + kern[2*w + 1] * (int)diffMap[(i+0)*W + j]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)diffMap[i*W + (j-1)]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)diffMap[i*W + (j-0)]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)diffMap[i*W + (j+1)]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)diffMap[i*W + (j+0)]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)diffMap[(i-1)*W + (j-1)]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)diffMap[(i-0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)diffMap[(i-1)*W + (j+1)]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)diffMap[(i-0)*W + (j+0)]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)diffMap[(i+1)*W + (j-1)]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)diffMap[(i+0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)diffMap[(i+1)*W + (j+1)]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)diffMap[(i+0)*W + (j+0)]; // extend
    
    edge = (edge * ((double)1 - bias)) + ((double)32000 * (bias));

    // clamp
    if (edge < 0)
        edge = 0;
    
    if (edge > 31999)
        edge = 31999;

    return edge;

}

static void applyKernel3D(char *kern, double bias) 
{
    int kernel[9]; pickKern(kern, kernel);
    memset(diffResult, 32002, dshmsz);
    
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            diffResult[i*dW + j] = convolve(i,j, kernel, kern, dW, dH, bias);
        }
    }

}

static void computeDifferential(char *kern, double bias) 
{
    memset(dxMap, 0, dshmsz);
    memset(dyMap, 0, dshmsz);
    memset(dzMap, 0, dshmsz);

    (void) kern;

    int16_t x; int16_t y; int16_t z;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            x = normalMap[i*dW*3 + j*3 + 0];
            y = normalMap[i*dW*3 + j*3 + 1];
            z = normalMap[i*dW*3 + j*3 + 2];
            if (z != 32001) {
                dxMap[i*dW + j] = x;
                dyMap[i*dW + j] = y;
                dzMap[i*dW + j] = z;
            }
        }
    }
 
    // compute dxMap
    memcpy(diffMap, dzMap, dshmsz);
    applyKernel3D((char*)"sobx", bias);
    memcpy(dxMap, diffResult, dshmsz);

    // compute dyMap
    memcpy(diffMap, dzMap, dshmsz);
    applyKernel3D((char*)"soby", bias);
    memcpy(dyMap, diffResult, dshmsz);

    // compute dzMap
    //memcpy(diffMap, dzMap, dshmsz);
    //applyKernel3D(kern, bias); 
    //memcpy(dzMap, diffResult, dshmsz);

    
}

static void buildNormal() 
{
    int16_t x; int16_t y; int16_t z;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            x = dxMap[i*dW + j];
            y = dyMap[i*dW + j];
            z = dzMap[i*dW + j];
            
            normalResult[i*dW*3 + j*3 + 0] = (x);
            normalResult[i*dW*3 + j*3 + 1] = (y);
            normalResult[i*dW*3 + j*3 + 2] = (255);
        }
    }
}

static int convolveDepth(int i, int j, int kern[9], char *kernel, int W, int H) 
{
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)dConvolveMap[i*W + j];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)dConvolveMap[(i-1)*W + j];
    else
        edge = edge + kern[0*w + 1] * (int)dConvolveMap[(i-0)*W + j]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)dConvolveMap[(i+1)*W + j];
    else
        edge = edge + kern[2*w + 1] * (int)dConvolveMap[(i+0)*W + j]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)dConvolveMap[i*W + (j-1)]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)dConvolveMap[i*W + (j-0)]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)dConvolveMap[i*W + (j+1)]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)dConvolveMap[i*W + (j+0)]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)dConvolveMap[(i-1)*W + (j-1)]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)dConvolveMap[(i-0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)dConvolveMap[(i-1)*W + (j+1)]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)dConvolveMap[(i-0)*W + (j+0)]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)dConvolveMap[(i+1)*W + (j-1)]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)dConvolveMap[(i+0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)dConvolveMap[(i+1)*W + (j+1)]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)dConvolveMap[(i+0)*W + (j+0)]; // extend

    
    if (strncmp(kernel, "blur", 4) == 0) 
        edge = edge/(4+2+2+1+1+1+1);
    
    edge = edge/2 + (32000/2);

    // clamp
    if (edge < 0)
        edge = 0;
    
    if (edge > 31999)
        edge = 31999;

    return edge;

}

static int convolveColour(int i, int j, int kern[9], char *kernel, int W, int H, double bias) 
{
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)cConvolveMap[i*W + j];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)cConvolveMap[(i-1)*W + j];
    else
        edge = edge + kern[0*w + 1] * (int)cConvolveMap[(i-0)*W + j]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)cConvolveMap[(i+1)*W + j];
    else
        edge = edge + kern[2*w + 1] * (int)cConvolveMap[(i+0)*W + j]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)cConvolveMap[i*W + (j-1)]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)cConvolveMap[i*W + (j-0)]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)cConvolveMap[i*W + (j+1)]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)cConvolveMap[i*W + (j+0)]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)cConvolveMap[(i-1)*W + (j-1)]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)cConvolveMap[(i-0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)cConvolveMap[(i-1)*W + (j+1)]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)cConvolveMap[(i-0)*W + (j+0)]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)cConvolveMap[(i+1)*W + (j-1)]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)cConvolveMap[(i+0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)cConvolveMap[(i+1)*W + (j+1)]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)cConvolveMap[(i+0)*W + (j+0)]; // extend
    

    if (strncmp(kernel, "blur", 4) == 0) 
        edge = edge/(4+2+2+1+1+1+1);

    edge = (edge * ((double)1 - bias)) + ((double)32000 * (bias));

    // clamp
    if (edge < 0)
        edge = 0;
    
    if (edge > 31999)
        edge = 31999;

    return edge;

}

static void applyKernelDepth(char *kern, int W, int H) 
{
    int kernel[9]; pickKern(kern, kernel);
    memset(dConvolveResult, 32002, sizeof(dConvolveResult));
    for(int i=0; i < H; i++) {
        for(int j=0; j < W; j++) {
            dConvolveResult[i*W + j] = convolveDepth(i,j, kernel, kern, W, H);
        }
    }

}

static void applyKernelColour(char *kern, int W, int H, double bias) 
{
    int kernel[9]; pickKern(kern, kernel);
    memset(cConvolveResult, 0, sizeof(cConvolveResult));

    // saftey
    if (bias > 1)
        bias = 0;

    for(int i=0; i < H; i++) {
        for(int j=0; j < W; j++) {
            cConvolveResult[i*W + j] = convolveColour(i,j, kernel, kern, W, H, bias);
        }
    }

}


static void toGreyScale(double rweight, double gweight, double bweight) 
{
    uint8_t red; uint8_t green; uint8_t blue;
    for(int i=0; i < cH; i++) {
        for(int j=0; j < cW; j++) {
            red =   greyColourMap[i*cW*3 + j*3 + 0];
            green = greyColourMap[i*cW*3 + j*3 + 1];
            blue =  greyColourMap[i*cW*3 + j*3 + 2];
            greyResult[i*cW + j] = (uint8_t)(red*rweight + green*gweight + blue*bweight);
        }
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
    int16_t x; int16_t y; int16_t z;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            x = printMap[i*dW*3 + j*3 + 0];
            y = printMap[i*dW*3 + j*3 + 1];
            z = printMap[i*dW*3 + j*3 + 2];
            if (z != 32001)
                f << x << "," << y << "," << z << endl;
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

static PyObject *getDepthColoured(PyObject *self, PyObject *args)
{
    npy_intp dims[3] = {dH, dW, 3};

    memcpy(depthCMap, depthFullMap, dshmsz);
    uint32_t colour;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            //colour = ((uint32_t)depthCMap[i*dW + j]) * 524; // convert approx 2^15 bits of data to 2^24 bits  
            colour = 16777216.0*(((double)depthCMap[i*dW + j])/31999.0); // 2^24 * zval/~2^15(zrange)
            
            depthColouredMap[i*dW*3 + j*3 + 0] = (uint8_t) ((colour << (32 - 8*1)) >> (32 - 8));
            depthColouredMap[i*dW*3 + j*3 + 1] = (uint8_t) ((colour << (32 - 8*2)) >> (32 - 8));
            depthColouredMap[i*dW*3 + j*3 + 2] = (uint8_t) ((colour << (32 - 8*3)) >> (32 - 8));


            //depthColouredMap[i*dW*3 + j*3 + 0] = (uint8_t) (((depthCMap[i*dW + j] << (16 - 5*1)) >> (16 - 5)) << 3);
            //depthColouredMap[i*dW*3 + j*3 + 1] = (uint8_t) (((depthCMap[i*dW + j] << (16 - 5*2)) >> (16 - 5)) << 3);
            //depthColouredMap[i*dW*3 + j*3 + 2] = (uint8_t) (((depthCMap[i*dW + j] << (16 - 5*3)) >> (16 - 5)) << 3);

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

    if (!PyArg_ParseTuple(args, "si", &kern, &repeat))
        return NULL;

    memcpy(dConvolveMap, depthFullMap, dshmsz);
   
    for(int i = 0; i < repeat; i++) {
        applyKernelDepth(kern, dW, dH); 
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
    memcpy(printMap, vertexFullMap, vshmsz*3);
    saveMap(map, file);

    Py_RETURN_NONE;
}

static PyObject *getNormal(PyObject *self, PyObject *args)
{
    memcpy(normalMap, vertexFullMap, vshmsz*3);
    double bias = 0.5;
    char * kern = (char*)"placeholder";
    computeDifferential(kern, bias); // applies sobel kernel to each plane 
    //crossMaps(); // cross product on three planes, store in normalMap result
    buildNormal(); // code will become cross normal soon

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
