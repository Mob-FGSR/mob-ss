Texture2D<float> TargetShadowTexture : register(t0);
Texture2D<float> RefShadowTexture : register(t1);
Texture2D<float2> FlowTexture : register(t2);
Texture2D<float2> MotionVectorTexture : register(t3);

RWTexture2D<float> SelectMap : register(u0);

SamplerState LinearSampler : register(s0);

groupshared float MVSads[64];
groupshared float FlowSads[64];

[numthreads(8, 8, 1)]
void main(int2 iGlobalId : SV_DispatchThreadID,
		  int2 iLocalId : SV_GroupThreadID,
		  int2 iGroupId : SV_GroupID)
{
	uint pixelWidth, pixelHeight;
	TargetShadowTexture.GetDimensions(pixelWidth, pixelHeight);

	int2 flow = int2(FlowTexture[iGroupId]);
	float2 mv = MotionVectorTexture[iGlobalId];
	float2 uv = (float2(iGlobalId) + 0.5) / float2(pixelWidth, pixelHeight);
	float2 flowUV = uv + flow / float2(pixelWidth, pixelHeight);
	float2 mvUV = uv - float2(mv.x, -mv.y) / float2(pixelWidth, pixelHeight);

	if (any(mvUV < 0) || any(mvUV > 1))
	{
		FlowSads[iLocalId.y * 8 + iLocalId.x] = 10;
		MVSads[iLocalId.y * 8 + iLocalId.x] = 0;
	}
	else
	{
		float refShadowPixel = RefShadowTexture[iGlobalId];
		float flowTargetPixel = TargetShadowTexture.SampleLevel(LinearSampler, flowUV, 0);
		float MVTargetPixel = TargetShadowTexture.SampleLevel(LinearSampler, mvUV, 0);
		FlowSads[iLocalId.y * 8 + iLocalId.x] = abs(refShadowPixel - flowTargetPixel);
		MVSads[iLocalId.y * 8 + iLocalId.x] = abs(refShadowPixel - MVTargetPixel);
	}

	GroupMemoryBarrierWithGroupSync();

	if (iLocalId.x == 0 && iLocalId.y == 0)
	{
		float MVSad = 0;
		for (int i = 0; i < 64; i++)
			MVSad += MVSads[i];

		float flowSad = 0;
		for (int i = 0; i < 64; i++)
			flowSad += FlowSads[i];

		if (flowSad < MVSad - 3)
		{
			SelectMap[iGroupId] = 1;
		}
		else
		{
			SelectMap[iGroupId] = 0;	
		}
	}
}