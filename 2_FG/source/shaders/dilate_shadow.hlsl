Texture2D<float> ShadowTexture : register(t0);
RWTexture2D<float> DilatedShadowTexture : register(u0);

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

[numthreads(8, 8, 1)]
void main(int2 iGlobalId : SV_DispatchThreadID)
{   
    int width, height;
    ShadowTexture.GetDimensions(width, height);

    const int2 offsets[8] =
    {
        int2(-1, -1),
        int2(-1, 0),
        int2(-1, 1),
        int2(0, -1),
        int2(0, 1),
        int2(1, -1),
        int2(1, 0),
        int2(1, 1)
    };

    int2 nearestPos = iGlobalId;
    float nearestShadow = ShadowTexture[iGlobalId];

    if (nearestShadow <= 0.1)
    {
        for (int i = 0; i < 8; i++)
        {
            int2 samplePos = clamp(iGlobalId + offsets[i], int2(0, 0), int2(width, height));
            float s = ShadowTexture[samplePos];
            if (s > 0.1)
            {
                nearestShadow = s;
                break;
            }
        }
    }
    
    DilatedShadowTexture[iGlobalId] = nearestShadow;
}