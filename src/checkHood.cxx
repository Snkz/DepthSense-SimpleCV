// Dump for shitty code

// internal maps for blob finding
static int16_t blobMap[320*240];
static int16_t blobResult[320*240];
static int16_t blobResultClone[320*240];
static uint8_t visited[320][240];


static bool checkHood(int p_i, int p_j, int base, double thresh_high, double thresh_low, int * pack)
{

    int depth = blobMap[p_i*dW + p_j];
    bool push_val = false;
    
    if (visited[p_i][p_j] > 3) {
        // natta 
        //blobResult[p_i*dW + p_j] = depth;
    } else if (visited[p_i][p_j] > 2) {
        // include the value since the neighbours all match
        // not sure if i should explore but ...
        if (!((depth < thresh_high + base) &&
            (depth > base - thresh_low))) {
            pack[0] = p_i; pack[1] = p_j; pack[2] = base;
            push_val = true;
            blobResult[p_i*dW + p_j] = depth;
        }
    } else if (visited[p_i][p_j] > 0) {
        if ((depth < thresh_high + base) &&
            (depth > base - thresh_low)) {
            pack[0] = p_i; pack[1] = p_j; pack[2] = depth;
            push_val = true;
            blobResult[p_i*dW + p_j] = depth;
        }

    // init
    } else if (visited[p_i][p_j] == 0) {
        if ((depth < thresh_high + base) &&
            (depth > base - thresh_low)) {
            pack[0] = p_i; pack[1] = p_j; pack[2] = depth;
            push_val = true;
        }
    }

    return push_val;

}

/* 
 * Blobs bitch
 */
static void findBlob(int sy, int sx, double thresh_high, double thresh_low) 
{

    list<int *> queue;
    memset(visited, 0, sizeof(visited));
    memset(blobResult, 32002, sizeof(blobResult));
   
    int *pack = (int *)malloc(sizeof(int) * 3);
    pack[0] = sy; pack[1] = sx; pack[2] = blobMap[sy*dW + sx];

    // assume it passes the threshold/base requirement, can return here possibly
    queue.push_back(pack);
    visited[sy][sx] = 1;
    blobResult[sy*dW + sx] = blobMap[sy*dW + sx];

    while(!queue.empty()){
        int * val = queue.front();
        queue.pop_front();
        int p_i = val[0];
        int p_j = val[1];
        int p_v = val[2];

        // DOWN
        if (p_i + 1 < dH) {
            int *dpack = (int *)malloc(sizeof(int) * 3);
            checkHood(p_i + 1, p_j, p_v, thresh_high, thresh_low, dpack) ?
                queue.push_back(dpack) : free(dpack);

            visited[p_i + 1][p_j]++;
        }


        // UP
        if (p_i - 1 > 0) {
            int *upack = (int *)malloc(sizeof(int) * 3);
            checkHood(p_i - 1, p_j, p_v, thresh_high, thresh_low, upack) ?
                queue.push_back(upack) : free(upack);

            visited[p_i - 1][p_j]++;
        }

        // LEFT
        if (p_j - 1 > 0) {
            int *lpack = (int *)malloc(sizeof(int) * 3);
            checkHood(p_i, p_j - 1, p_v, thresh_high, thresh_low, lpack) ?
                queue.push_back(lpack) : free(lpack);
            
            visited[p_i][p_j - 1]++;
       }

        // RIGHT
        if (p_j + 1 < dW) {
            int *rpack = (int *)malloc(sizeof(int) * 3);
            checkHood(p_i, p_j + 1, p_v, thresh_high, thresh_low, rpack) ? 
                queue.push_back(rpack) : free(rpack);
            
            visited[p_i][p_j + 1]++;
        }

        free(val);
    }

}


