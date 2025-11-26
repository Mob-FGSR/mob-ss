RWTexture2D<float2> MotionVectorTexture : register(u0);


[numthreads(8, 8, 1)]
void main(int2 iGlobalId: SV_DispatchThreadID)
{
    int width, height;
    MotionVectorTexture.GetDimensions(width, height);
    float2 MV = MotionVectorTexture[iGlobalId];
    MV.x = MV.x / width;
    MV.y = -MV.y / height;
    MotionVectorTexture[iGlobalId] = MV;
}