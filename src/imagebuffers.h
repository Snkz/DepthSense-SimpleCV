/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            imagebuffers.h                                              
 * author:          Abdi Dahir                           
 * modified:        May 9 2014                                               
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:                      
 *                                                                             
 * Imagebuffers defined here for various image proccessing operations
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

// internal map copies
extern uint8_t colourMapClone[640*480*3];
extern int16_t depthMapClone[320*240];
extern int16_t vertexMapClone[320*240*3];
extern float accelMapClone[3];
extern float uvMapClone[320*240*2];
extern float vertexFMapClone[320*240*3];
extern uint8_t syncMapClone[320*240*3];
extern int16_t nPrintMap[320*240*3];
extern int16_t vPrintMap[320*240*3];

// colouring depth map
extern int16_t depthCMap[320*240];
extern uint8_t depthColouredMap[320*240*3];
extern uint8_t depthColouredMapClone[320*240*3];

// internal maps for edge finding
extern int16_t dConvolveMap[320*240];
extern int16_t dConvolveResult[320*240];
extern int16_t dConvolveResultClone[320*240];

// internal maps for edge finding in colour
extern uint8_t cConvolveMap[640*480];
extern uint8_t cConvolveResult[640*480];
extern uint8_t cConvolveResultClone[640*480];

// internal maps for edge finding
extern uint8_t greyColourMap[640*480*3];
extern uint8_t greyResult[640*480];
extern uint8_t greyResultClone[640*480];

// internal maps for normal computation
extern int16_t normalMap[320*240*3];
extern int16_t dxMap[320*240*3];
extern int16_t dyMap[320*240*3];
extern int16_t diffMap[320*240*3];
extern int16_t diffResult[320*240*3];
extern int16_t normalResult[320*240*3];
extern int16_t normalResultClone[320*240*3];
