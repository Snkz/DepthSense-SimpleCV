/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            initdepthsense.h                                              
 * author:          Abdi Dahir                           
 * modified:        May 9 2014                                               
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:                      
 *                                                                             
 * Imagebuffers defined here along with the depthsense start/stop ops
 * -----------------------------------------------------------------------------
 */

#include <stdint.h>

// map dimensions
static int32_t dW = 320;
static int32_t dH = 240;
static int32_t cW = 640;
static int32_t cH = 480;

static int dshmsz = dW*dH*sizeof(int16_t);
static int cshmsz = cW*cH*sizeof(uint8_t);
static int vshmsz = dW*dH*sizeof(int16_t);
static int ushmsz = dW*dH*sizeof(float);
static int hshmsz = dW*dH*sizeof(uint8_t);

// shared mem depth maps
extern int16_t *depthMap;
extern int16_t *depthFullMap;

// shared mem vertex maps
extern int16_t *vertexMap;
extern int16_t *vertexFullMap;

extern float *vertexFMap;
extern float *vertexFFullMap;

// shared mem colour maps
extern uint8_t *colourMap;
extern uint8_t *colourFullMap;

// shared mem accel maps
extern float *accelMap;
extern float *accelFullMap;

// shared mem uv maps
extern float *uvMap;
extern float *uvFullMap;

// colouring depth map
extern int16_t * depthCMap;
extern uint8_t * depthColouredMap;

// internal maps for edge finding
extern int16_t * dConvolveMap;
extern int16_t * dConvolveResult;

// internal maps for edge finding in colour
extern uint8_t * cConvolveMap;
extern uint8_t * cConvolveResult;

// internal maps for edge finding
extern uint8_t * greyColourMap;
extern uint8_t * greyResult;

// internal maps for normal computation
extern int16_t * normalMap;
extern int16_t * dxMap;
extern int16_t * dyMap;
extern int16_t * diffMap;
extern int16_t * diffResult;
extern int16_t * normalResult;


extern "C" {
    void killds();
    void initds();
}
