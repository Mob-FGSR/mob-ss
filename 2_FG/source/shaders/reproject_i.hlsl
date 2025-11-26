/* 
 *        Interpolation version 
 */

Texture2D<float> r_current_depth : register(t0);
Texture2D<float2> r_current_motion_vector : register(t1);
Texture2D<float2> r_previous_motion_vector : register(t2);
RWTexture2D<uint> rw_reprojection : register(u0);
RWTexture2D<uint> debug_conflict_count : register(u1);

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

[numthreads(8, 8, 1)]
void main(int2 iGlobalId : SV_DispatchThreadID)
{   
    // Screen check
    if (any(iGlobalId >= int2(render_size.xy)))
    {
        return;
    }
    
    float2 uv = (float2(iGlobalId) + 0.5f) * render_size.zw;
    float2 mv_t1 = r_current_motion_vector.Load(int3(iGlobalId, 0)).xy;
    
    // Linear motion estimation
    // float2 uvDelta = uv - mv_t1 * (1 - delta.x);
    
    // Quadratic motion estimation
    int2 pos_t0 = int2((uv - mv_t1) * render_size.xy);
    float2 mv_t0 = r_previous_motion_vector.Load(int3(pos_t0, 0)).xy;
    float2 uvDelta = uv + (-1 + delta.y + delta.w) * mv_t1 + (delta.y - delta.w) * mv_t0;
    
    if (all(uvDelta >= float2(0, 0)) && all(uvDelta <= float2(1, 1)))
    {
        int2 posDelta = int2(uvDelta * render_size.xy);
        float depth = r_current_depth.Load(int3(iGlobalId, 0)).x;
        uint data = packReprojectionDataToUint(depth, iGlobalId, posDelta);
        InterlockedMax(rw_reprojection[posDelta], data);
        InterlockedAdd(debug_conflict_count[posDelta], 1);
    }
}