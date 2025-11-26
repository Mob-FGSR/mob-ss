Texture2D<float> DynamicMask : register(t0);
Texture2D<float> Depth : register(t1);
Texture2D<float2> MotionVector : register(t2);
RWTexture2D<float> DilatedDynamicMask : register(u0);
RWTexture2D<float> DilatedDepth : register(u1);
RWTexture2D<float2> DilatedMotionVector : register(u2);

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

void FindClosestDepthOffset(uint2 PixelPos, float DeviceZ, int PixelRadius, out float2 VelocityPixelOffset, out float ClosestDeviceZ)
{
	float4 Depths;
	Depths.x = Depth[int2(PixelPos) + int2(-PixelRadius, -PixelRadius)];
	Depths.y = Depth[int2(PixelPos) + int2( PixelRadius, -PixelRadius)];
	Depths.z = Depth[int2(PixelPos) + int2(-PixelRadius,  PixelRadius)];
	Depths.w = Depth[int2(PixelPos) + int2( PixelRadius,  PixelRadius)];

	float2 DepthOffset = float2(PixelRadius, PixelRadius);
	float DepthOffsetXx = float(PixelRadius);

	if(Depths.x > Depths.y) 
	{
		DepthOffsetXx = -PixelRadius;
	}
	if(Depths.z > Depths.w) 
	{
		DepthOffset.x = -PixelRadius;
	}
	float DepthsXY = max(Depths.x, Depths.y);
	float DepthsZW = max(Depths.z, Depths.w);
	if (DepthsXY > DepthsZW) 
	{
		DepthOffset.y = -PixelRadius;
		DepthOffset.x = DepthOffsetXx; 
	}
	float DepthsXYZW = max(DepthsXY, DepthsZW);
	
	ClosestDeviceZ = DeviceZ;
	VelocityPixelOffset = 0.0;

	if(DepthsXYZW > DeviceZ)
	{
		VelocityPixelOffset = DepthOffset;
		ClosestDeviceZ = DepthsXYZW;
	}
}

[numthreads(8, 8, 1)]
void main(int2 iGlobalId : SV_DispatchThreadID)
{

    float DeviceZ = Depth[iGlobalId];
    float ClosestDeviceZ = DeviceZ;
    float2 PixelOffset = float2(0.0, 0.0);

    FindClosestDepthOffset(iGlobalId, DeviceZ, 1, PixelOffset, ClosestDeviceZ);

    DilatedDepth[iGlobalId] = ClosestDeviceZ;
    DilatedDynamicMask[iGlobalId] = DynamicMask[iGlobalId + PixelOffset];
    DilatedMotionVector[iGlobalId] = MotionVector[iGlobalId + PixelOffset];
}