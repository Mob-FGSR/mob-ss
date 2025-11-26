#define INVALID 0

Texture2D<uint> r_filled_reprojection : register(t0);
Texture2D<float4> r_current_color_input_fg : register(t1);
Texture2D<float4> r_previous_color_input_fg : register(t2);
Texture2D<float> r_current_depth : register(t3);
Texture2D<float> r_previous_depth : register(t4);
Texture2D<float2> r_current_motion_vector : register(t5);
Texture2D<float2> r_previous_motion_vector : register(t6);
Texture2D<float> r_sample_lut : register(t7);
RWTexture2D<float4> rw_frame_generation_result : register(u0);
SamplerState lutSampler : register(s0);

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

///// Sample /////
int2 clampCoord(int2 pos, int2 offset, int2 textureSize)
{
    int2 result = pos + offset;
    result.x = (offset.x < 0) ? max(result.x, 0) : result.x;
    result.x = (offset.x > 0) ? min(result.x, textureSize.x - 1) : result.x;
    result.y = (offset.y < 0) ? max(result.y, 0) : result.y;
    result.y = (offset.y > 0) ? min(result.y, textureSize.y - 1) : result.y;
    return result;
}

float4 sampleWithLut(Texture2D<float4> tex, float2 uv, float2 textureSize, Texture2D<float> lut, SamplerState lutSampler)
{
    float2 fPos = uv * textureSize;
    float2 center = floor(fPos + 0.5);
    float2 d = fPos - center + float2(0.5, 0.5);

    // LUT: (32 * 4) * (32 * 4) = 128 * 128
    // 0.25 = 32 / 128
    float2 lutSampleUV = (31 * d + float2(0.5, 0.5)) / 128.0f;
    float weight00 = lut.SampleLevel(lutSampler, lutSampleUV, 0).x;
    float weight01 = lut.SampleLevel(lutSampler, lutSampleUV + float2(0, 1) * 0.25, 0).x;
    float weight02 = lut.SampleLevel(lutSampler, lutSampleUV + float2(0, 2) * 0.25, 0).x;
    float weight03 = lut.SampleLevel(lutSampler, lutSampleUV + float2(0, 3) * 0.25, 0).x;
    float weight10 = lut.SampleLevel(lutSampler, lutSampleUV + float2(1, 0) * 0.25, 0).x;
    float weight11 = lut.SampleLevel(lutSampler, lutSampleUV + float2(1, 1) * 0.25, 0).x;
    float weight12 = lut.SampleLevel(lutSampler, lutSampleUV + float2(1, 2) * 0.25, 0).x;
    float weight13 = lut.SampleLevel(lutSampler, lutSampleUV + float2(1, 3) * 0.25, 0).x;
    float weight20 = lut.SampleLevel(lutSampler, lutSampleUV + float2(2, 0) * 0.25, 0).x;
    float weight21 = lut.SampleLevel(lutSampler, lutSampleUV + float2(2, 1) * 0.25, 0).x;
    float weight22 = lut.SampleLevel(lutSampler, lutSampleUV + float2(2, 2) * 0.25, 0).x;
    float weight23 = lut.SampleLevel(lutSampler, lutSampleUV + float2(2, 3) * 0.25, 0).x;
    float weight30 = lut.SampleLevel(lutSampler, lutSampleUV + float2(3, 0) * 0.25, 0).x;
    float weight31 = lut.SampleLevel(lutSampler, lutSampleUV + float2(3, 1) * 0.25, 0).x;
    float weight32 = lut.SampleLevel(lutSampler, lutSampleUV + float2(3, 2) * 0.25, 0).x;
    float weight33 = lut.SampleLevel(lutSampler, lutSampleUV + float2(3, 3) * 0.25, 0).x;

    int2 samplePos11 = int2(center - float2(0.5, 0.5));
    int2 iTextureSize = int2(textureSize);
    float4 color00 = tex.Load(int3(clampCoord(samplePos11, int2(-1, -1), iTextureSize), 0));
    float4 color01 = tex.Load(int3(clampCoord(samplePos11, int2(-1, 0), iTextureSize), 0));
    float4 color02 = tex.Load(int3(clampCoord(samplePos11, int2(-1, 1), iTextureSize), 0));
    float4 color03 = tex.Load(int3(clampCoord(samplePos11, int2(-1, 2), iTextureSize), 0));
    float4 color10 = tex.Load(int3(clampCoord(samplePos11, int2(0, -1), iTextureSize), 0));
    float4 color11 = tex.Load(int3(clampCoord(samplePos11, int2(0, 0), iTextureSize), 0));
    float4 color12 = tex.Load(int3(clampCoord(samplePos11, int2(0, 1), iTextureSize), 0));
    float4 color13 = tex.Load(int3(clampCoord(samplePos11, int2(0, 2), iTextureSize), 0));
    float4 color20 = tex.Load(int3(clampCoord(samplePos11, int2(1, -1), iTextureSize), 0));
    float4 color21 = tex.Load(int3(clampCoord(samplePos11, int2(1, 0), iTextureSize), 0));
    float4 color22 = tex.Load(int3(clampCoord(samplePos11, int2(1, 1), iTextureSize), 0));
    float4 color23 = tex.Load(int3(clampCoord(samplePos11, int2(1, 2), iTextureSize), 0));
    float4 color30 = tex.Load(int3(clampCoord(samplePos11, int2(2, -1), iTextureSize), 0));
    float4 color31 = tex.Load(int3(clampCoord(samplePos11, int2(2, 0), iTextureSize), 0));
    float4 color32 = tex.Load(int3(clampCoord(samplePos11, int2(2, 1), iTextureSize), 0));
    float4 color33 = tex.Load(int3(clampCoord(samplePos11, int2(2, 2), iTextureSize), 0));

    float4 result = float4(0, 0, 0, 0);
    result += color00 * weight00;
    result += color01 * weight01;
    result += color02 * weight02;
    result += color03 * weight03;
    result += color10 * weight10;
    result += color11 * weight11;
    result += color12 * weight12;
    result += color13 * weight13;
    result += color20 * weight20;
    result += color21 * weight21;
    result += color22 * weight22;
    result += color23 * weight23;
    result += color30 * weight30;
    result += color31 * weight31;
    result += color32 * weight32;
    result += color33 * weight33;
    result /= (weight00 + weight01 + weight02 + weight03 + weight10 + weight11 + weight12 + weight13 + weight20 + weight21 + weight22 + weight23 + weight30 + weight31 + weight32 + weight33);
    return result;
}


[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int2 pos = int2(DTid.xy);
    float2 uv = (float2(pos) + 0.5f) * presentation_size.zw;
    int2 scaledPos = int2(float2(pos) / render_scale);
    
    uint packedData = r_filled_reprojection.Load(int3(scaledPos, 0)).x;
    float2 mv_t1;
    int2 posBeforeReprojection = int2(-1, -1);
    if (packedData == INVALID)
    {
        mv_t1 = r_previous_motion_vector.Load(int3(scaledPos, 0)).xy;
    }
    else
    {
        posBeforeReprojection = unpackSourcePosFromUint(packedData, scaledPos);
        mv_t1 = r_current_motion_vector.Load(int3(posBeforeReprojection, 0)).xy;
    }
    
    // // Linear motion estimation
    // float2 sampleUV_t1 = uv + mv_t1 * (1.0f - delta.x);
    // float2 sampleUV_t0 = uv - mv_t1 * delta.x;
    
    // Quadratic motion estimation
    float2 sampleUV_t1;
    float2 sampleUV_t0;
    if (all(posBeforeReprojection == int2(-1, -1)))
    {
        // Fallback to linear motion estimation
        sampleUV_t1 = uv + mv_t1 * (1.0f - delta.x);
        sampleUV_t0 = uv - mv_t1 * delta.x;
    }
    else
    {
        float2 uv_t1 = (float2(posBeforeReprojection) + float2(0.5, 0.5)) * render_size.zw;
        float2 uv_t0 = uv_t1 - mv_t1;
        int2 pos_t0 = int2(uv_t0 * render_size.xy);
        float2 mv_t0 = r_previous_motion_vector.Load(int3(pos_t0, 0)).xy;
        sampleUV_t0 = uv + (-delta.y - delta.w) * mv_t1 + (-delta.y + delta.w) * mv_t0;
        sampleUV_t1 = mv_t1 + sampleUV_t0;
    }
    
    int2 samplePos_LR_t1 = int2(sampleUV_t1 * render_size.xy);
    int2 samplePos_LR_t0 = int2(sampleUV_t0 * render_size.xy);
    
    float3 color_t1 = sampleWithLut(r_current_color_input_fg, sampleUV_t1, presentation_size.xy, r_sample_lut, lutSampler).xyz;
    float3 color_t0 = sampleWithLut(r_previous_color_input_fg, sampleUV_t0, presentation_size.xy, r_sample_lut, lutSampler).xyz;
    float depth_t1 = r_current_depth.Load(int3(samplePos_LR_t1, 0)).x;
    float depth_t0 = r_previous_depth.Load(int3(samplePos_LR_t0, 0)).x;
    
    float3 color;
    float depthDiff = abs(depth_t0 - depth_t1);
    if (any(sampleUV_t0 < float2(0, 0)) || any(sampleUV_t0 > float2(1, 1)))
    {
        color = color_t1;    
    }
    else if (any(sampleUV_t1 < float2(0, 0)) || any(sampleUV_t1 > float2(1, 1)))
    {
        color = color_t0;
    }
    else if (depthDiff < depth_diff_threshold_fg)
    {
        float3 colorDiff = abs(color_t1 - color_t0);
        float lumaDiff = colorDiff.r * float(0.5) + (colorDiff.b * float(0.5) + colorDiff.g);
        if (lumaDiff < color_diff_threshold_fg) 
        {
            color = delta.x < 0.5 ? color_t0 : color_t1;
        }
        else 
        {
            color = color_t0;
        }
    }
    else 
    {
        color = depth_t1 <= depth_t0 ? color_t1 : color_t0;
    }
    
    rw_frame_generation_result[pos] = float4(color.xyz, 1.0f);
}