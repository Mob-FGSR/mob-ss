
Texture2D<float> ShadowTexture : register(t0);
RWTexture2D<float> DepthTexture : register(u0);


[numthreads(8, 8, 1)]
void main(int2 iGlobalId: SV_DispatchThreadID)
{
    DepthTexture[iGlobalId] -= 0.01 * ShadowTexture[iGlobalId];
}