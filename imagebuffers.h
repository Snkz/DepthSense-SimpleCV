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

// map dimensions
static int32_t dW = 320;
static int32_t dH = 240;
static int32_t cW = 640;
static int32_t cH = 480;

int dshmsz = dW*dH*sizeof(int16_t);
int cshmsz = cW*cH*sizeof(uint8_t);
int vshmsz = dW*dH*sizeof(int16_t);
int ushmsz = dW*dH*sizeof(float);
int hshmsz = dW*dH*sizeof(uint8_t);

// shared mem depth maps
static int16_t *depthMap;
static int16_t *depthFullMap;

// shared mem vertex maps
static int16_t *vertexMap;
static int16_t *vertexFullMap;

static float *vertexFMap;
static float *vertexFFullMap;


// shared mem colour maps
static uint8_t *colourMap;
static uint8_t *colourFullMap;

// shared mem accel maps
static float *accelMap;
static float *accelFullMap;

// shared mem uv maps
static float *uvMap;
static float *uvFullMap;

// internal map copies
static uint8_t colourMapClone[640*480*3];
static int16_t depthMapClone[320*240];
static int16_t vertexMapClone[320*240*3];
static float accelMapClone[3];
static float uvMapClone[320*240*2];
static float vertexFMapClone[320*240*3];
static uint8_t syncMapClone[320*240*3];
static int16_t printMap[320*240*3];

// colouring depth map
static int16_t depthCMap[320*240];
static uint8_t depthColouredMap[320*240*3];
static uint8_t depthColouredMapClone[320*240*3];

// internal maps for edge finding
static int16_t dConvolveMap[320*240];
static int16_t dConvolveResult[320*240];
static int16_t dConvolveResultClone[320*240];

// internal maps for edge finding in colour
static uint8_t cConvolveMap[640*480];
static uint8_t cConvolveResult[640*480];
static uint8_t cConvolveResultClone[640*480];

// internal maps for edge finding
static uint8_t greyColourMap[640*480*3];
static uint8_t greyResult[640*480];
static uint8_t greyResultClone[640*480];

// internal maps for normal computation
static int16_t normalMap[320*240*3];
static int16_t dxMap[320*240];
static int16_t dyMap[320*240];
static int16_t dzMap[320*240];
static int16_t diffMap[320*240];
static int16_t diffResult[320*240];
static int16_t normalResult[320*240*3];
static int16_t normalResultClone[320*240*3];


