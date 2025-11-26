///// Uniforms /////
cbuffer cb_t : register(b0)
{
    float4 render_size;
    float4 presentation_size;
    float4 delta;
    float2 jitter_offset;
    float depth_diff_threshold_sr;
    float color_diff_threshold_fg;
    float depth_diff_threshold_fg;
    float depth_scale;
    float depth_bias;
    float render_scale;
};

#define INVALID       uint(0)

///// Packing /////
// Packing constants
static const uint depthBits = 11;
static const uint xBits = 11;
static const uint yBits = 10;

static const uint maxDepth = (1 << depthBits) - 1;
static const int minX = -(1 << (xBits - 1));
static const int maxX = (1 << (xBits - 1)) - 1;
static const int minY = -(1 << (yBits - 1));
static const int maxY = (1 << (yBits - 1)) - 1;

// Pack (depth, relativePos.xy) to 11/11/10 uint
// Depth precision: 0.0004882
// RelativePos.x range: [-1024, 1023]
// RelativePos.y range: [-512, 511]
uint packReprojectionDataToUint(float depth, int2 sourcePos, int2 targetPos)
{
    uint uDepth = uint(float(maxDepth) * depth);
    int2 relativePos = clamp(targetPos - sourcePos, int2(minX, minY), int2(maxX, maxY));
    uint2 uRelativePos = uint2(relativePos - int2(minX, minY));

    uint result = (uDepth << (32 - depthBits)) | (uRelativePos.x << yBits) | (uRelativePos.y);

    return result;
}

float unpackDepthFromUint(uint reprojectionData)
{
    uint uDepth = reprojectionData >> (32 - depthBits);
    return float(uDepth) / float(maxDepth);
}

int2 unpackSourcePosFromUint(uint reprojectionData, int2 targetPos)
{
    uint uRelativeX = (reprojectionData >> yBits) & uint((1 << xBits) - 1);
    uint uRelativeY = reprojectionData & uint((1 << yBits) - 1);
    int2 relativePos = int2(uRelativeX, uRelativeY) + int2(minX, minY);
    int2 sourcePos = targetPos - relativePos;
    return sourcePos;
}

#define SETBIT(x) (1U << x)
#define FILL_DEPTH_DIFF_THRESHOLD 0.0005


Texture2D<uint> r_reprojection : register(t0);
RWTexture2D<uint> rw_filled_reprojection : register(u0);

// Select a neighboring pixel with valid value. And the depth of this pixel should be smaller than center pixel.
void selectValidNeighbor(int2 centerPos, float centerDepth, uint idx, inout float nearestDepth, inout uint selectedData, inout uint mask)
{
    const int2 offsets[9] =
    {
        int2(-1, -1),
        int2(-1, 0),
        int2(-1, 1),
        int2(0, -1),
        int2(0, 0),
        int2(0, 1),
        int2(1, -1),
        int2(1, 0),
        int2(1, 1)
    };
    
    int2 neighborPos = centerPos + offsets[idx];
    if (any(neighborPos < int2(0, 0)) || any(neighborPos >= int2(render_size.xy))) return;
    
    uint neighborData = r_reprojection.Load(int3(neighborPos, 0)).x;
    bool neighborValid = bool(neighborData != INVALID);
    float neighborDepth = unpackDepthFromUint(neighborData);
    
    float diff = neighborDepth - centerDepth;
    if (neighborValid && diff > FILL_DEPTH_DIFF_THRESHOLD)
    {
        if (neighborDepth > nearestDepth)
        {
            nearestDepth = neighborDepth;
            selectedData = neighborData;
        }
    }
    else
    {
        mask |= SETBIT(idx);
    }
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int2 pos = int2(DTid.xy);

    uint centerData = r_reprojection.Load(int3(pos, 0)).x;
    float centerDepth = unpackDepthFromUint(centerData);
    float nearestDepth = 0;
    uint selectedData = INVALID;
    uint mask = SETBIT(4);

    selectValidNeighbor(pos, centerDepth, 0, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 1, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 2, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 3, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 5, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 6, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 7, nearestDepth, selectedData, mask);
    selectValidNeighbor(pos, centerDepth, 8, nearestDepth, selectedData, mask);

    /*  
        idx:
        0 1 2
        3 4 5
        6 7 8
    */
    const uint rejectionMasks[4] =
    {
        SETBIT(0) | SETBIT(1) | SETBIT(3) | SETBIT(4), // Upper left
        SETBIT(1) | SETBIT(2) | SETBIT(4) | SETBIT(5), // Upper right
        SETBIT(3) | SETBIT(4) | SETBIT(6) | SETBIT(7), // Lower left
        SETBIT(4) | SETBIT(5) | SETBIT(7) | SETBIT(8), // Lower right
    };

    bool reject =
        ((mask & rejectionMasks[0]) == rejectionMasks[0]) ||
        ((mask & rejectionMasks[1]) == rejectionMasks[1]) ||
        ((mask & rejectionMasks[2]) == rejectionMasks[2]) ||
        ((mask & rejectionMasks[3]) == rejectionMasks[3]);
    
    uint result = 0;
    if (reject)
    {
        if (centerData != INVALID)
        {
            result = centerData;
        }
        else
        {
            result = INVALID;
        }
    }
    else
    {
        result = selectedData;
    }
    rw_filled_reprojection[pos] = result;
}