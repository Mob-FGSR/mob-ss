Texture2D<float4> PackedMVTexture : register(t0);
RWTexture2D<float2> MotionVectorTexture : register(u0);

float Unpack16(float2 packedValue)
{
    return dot(packedValue, float2(1.0, 1.0/255.0));
}

[numthreads(8, 8, 1)]
void main(int2 iGlobalId : SV_DispatchThreadID)
{
    int width, height;
    PackedMVTexture.GetDimensions(width, height);
    float4 packedMV = PackedMVTexture[iGlobalId];
    float2 mv;
    mv.x = Unpack16(packedMV.xy);
    mv.y = Unpack16(packedMV.zw);
    mv.x = (mv.x * 2 - 1) * width;
    mv.y = (mv.y * 2 - 1) * height;
    MotionVectorTexture[iGlobalId] = mv;
}