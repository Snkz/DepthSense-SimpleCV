/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            imageproccessing.h
 * author:          Abdi Dahir
 * modified:        May 9 2014
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:
 * 
 * Convoloutions and other assorted proccessing techniques defined here
 * -----------------------------------------------------------------------------
 */

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
      
static int blurKern[9] = { 1,  2,  1,  // needs to be normalized
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
      
static int lapKern[9] = {  1,  -2,   1,  
                          -2,   4,  -2, 
                           1,  -2,   1 };
/* 
 * Set kernel to the name defined by kern. 
 * kern must be in the following list of names
 * [edge, shrp, iden, blur, sobx, soby, scrx, scry, embs, edgh, lapl]
 * Each name in the list corresponds to one of the kernels above.
 */
void pickKern(char* kern, int kernel[9]);

/*
 * 3x3 Convoloutions on DepthMap, Colour Map (greyscaled) and on the vertexmap
 * planes. Each operation writes to a seperate output buffer 
 * (each with their own datatype)
 */
//TODO: Combine these if possible, edge value diffs can be extracted out
int convolve(int i, int j, int kern[9], char *kernel, int W, int H, double bias);
int convolveDepth(int i, int j, int kern[9], char *kernel, int W, int H, double bias); 
int convolveColour(int i, int j, int kern[9], char *kernel, int W, int H, double bias); 

/*
 * Loop through the respective maps and convolve each value
 */
void applyKernelDepth(char *kern, int W, int H, double bias);
void applyKernelColour(char *kern, int W, int H, double bias);
void applyKernel3D(char *kern, int W, int H, double bias);

/*
 * Differentiate the vertex map to compute tangent plane at each point
 * Note this method is used to build the normal map (which is initially
 * set to contain vertex values)
 */
void computeDifferential(char *kern, double bias); 

/*
 * Compute the normal map by finding the normal to the tangent plane
 * specified in dx and dy maps
 */
void crossMaps(void); 

/*
 * Compute the normal map by finding the normal to the tangent plane
 * specified in the vertexMap
 */
void computeNormalMap(void); 

/* 
 * Convert the colour buffer into greyscale values by given weights
 */
void toGreyScale(double rweight, double gweight, double bweight); 

/*
 * Using (assumed to be) up-to-date depth/uv/colour maps build a colour map
 * with the resoloution of the depth map with pixels that exist in both the 
 * depth and colour map exclusively (that info is provided by the uv map)
 */
//void buildSyncMap(void);


