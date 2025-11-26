Texture2D<float2> MotionVectorTexture : register(t0);

RWTexture2D<float4> PackedMVTexture: register(u0);

float4 Pack32(float value)
{
    const float4 bitShift = float4(1.0, 255.0, 255.0 * 255.0, 255.0 * 255.0 * 255.0);
    const float4 bitMask = float4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
    float4 encode = frac(value * bitShift);
    encode -= encode.gbaa * bitMask;
    return encode;
}

float2 Pack16(float value)
{
	const float2 bitShift = float2(1.0, 255.0);
	const float2 bitMask = float2(1.0 / 255.0, 0.0);
	float2 encode = frac(value * bitShift);
	encode -= encode.yy * bitMask;
	return encode;
}

[numthreads(8, 8, 1)]
void main(int2 iGlobalId: SV_DispatchThreadID)
{
    int width, height;
    MotionVectorTexture.GetDimensions(width, height);
    float2 MV = MotionVectorTexture[iGlobalId];
    MV.x = (MV.x / width + 1) / 2;
    MV.y = (MV.y / height + 1) / 2;
    PackedMVTexture[iGlobalId] = float4(Pack16(MV.x), Pack16(MV.y));
}