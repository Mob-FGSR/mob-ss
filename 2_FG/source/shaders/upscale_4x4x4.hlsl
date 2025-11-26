Texture2D<float> RefTexture : register(t0);
Texture2D<float> TargetTexture : register(t1);
Texture2D<float2> OldFlow : register(t2);
RWTexture2D<float2> Flow : register(u0);

#include "flow_estimation_common.hlsl"

groupshared uint sads[4][4][4];
groupshared int2 nearestVectors[4][4][4];
groupshared uint localRegion[4][4][4];
groupshared uint flow0Sads[4][4][4];

uint LoadRefImagePackedLuma(int2 iPxPos)
{
	int width, height;
	RefTexture.GetDimensions(width, height);

    int2 adjustedPos = int2(
        clamp(iPxPos.x, 0, width - 4),
        clamp(iPxPos.y, 0, height - 1)
    );

    uint luma0 = RefTexture[adjustedPos + int2(0, 0)] * 255;
    uint luma1 = RefTexture[adjustedPos + int2(1, 0)] * 255;
    uint luma2 = RefTexture[adjustedPos + int2(2, 0)] * 255;
    uint luma3 = RefTexture[adjustedPos + int2(3, 0)] * 255;

    return GetPackedLuma(width, iPxPos.x, luma0, luma1, luma2, luma3);
}

uint LoadTargetImagePackedLuma(int2 iPxPos)
{
	int width, height;
	TargetTexture.GetDimensions(width, height);

    int2 adjustedPos = int2(
        clamp(iPxPos.x, 0, width - 4),
        clamp(iPxPos.y, 0, height - 1)
    );

    uint luma0 = TargetTexture[adjustedPos + int2(0, 0)] * 255;
    uint luma1 = TargetTexture[adjustedPos + int2(1, 0)] * 255;
    uint luma2 = TargetTexture[adjustedPos + int2(2, 0)] * 255;
    uint luma3 = TargetTexture[adjustedPos + int2(3, 0)] * 255;

    return GetPackedLuma(width, iPxPos.x, luma0, luma1, luma2, luma3);
}

[numthreads(4, 4, 4)]
void main(int2 globalID : SV_DispatchThreadID, int3 localID : SV_GroupThreadID)
{
	int FlowWidth, FlowHeight;
	Flow.GetDimensions(FlowWidth, FlowHeight);
    
	int OldWidth, OldHeight;
	OldFlow.GetDimensions(OldWidth, OldHeight);

	int xOffset = (localID.z % 2) - 1 + globalID.x % 2;
	int yOffset = (localID.z / 2) - 1 + globalID.y % 2;

	int2 srcOFPos = clamp(int2(
		(globalID.x / 2) + xOffset,
		(globalID.y / 2) + yOffset),
		int2(0, 0),
		int2(FlowWidth - 1, FlowHeight - 1)
	);

	int2 nearestVector = int2(OldFlow[srcOFPos]);
	nearestVectors[localID.z][localID.y][localID.x] = nearestVector * 2;

	int maxY = 4;

	for (int n = localID.z; n < maxY; n += 4)
	{
		{
			int2 lumaPos = int2(globalID.x * 4, globalID.y * maxY + n);
			uint firstPixel = LoadRefImagePackedLuma(lumaPos);
			uint secondPixel = LoadTargetImagePackedLuma(lumaPos);
			localRegion[n][localID.y][localID.x] = firstPixel;
			flow0Sads[n][localID.y][localID.x] = Sad(firstPixel, secondPixel);
		}
	}
	
    GroupMemoryBarrierWithGroupSync();

	uint sad = 0;
	for (int n = 0; n < maxY; n++)
	{
		{
			int2 lumaPos = int2(globalID.x * 4, globalID.y * maxY + n) + nearestVector;
			uint secondPixel = LoadTargetImagePackedLuma(lumaPos);
			sad += Sad(localRegion[n][localID.y][localID.x], secondPixel);
		}
	}
	sad += sqrt(nearestVector.x * nearestVector.x + nearestVector.y * nearestVector.y) * 4;
	sads[localID.z][localID.y][localID.x] = sad;

 	GroupMemoryBarrierWithGroupSync();

    if (localID.z == 0)
    {
		uint bestSad = 0xffffffff;
		uint bestId = 0;
		uint flow0Sad = 0;

		for (int n = 0; n < 4; n++)
		{
			if ((sads[n][localID.y][localID.x]) < bestSad)
			{
				bestSad = sads[n][localID.y][localID.x];
				bestId = n;
			}
			flow0Sad += flow0Sads[n][localID.y][localID.x];
		}

		int2 outputVector = nearestVectors[bestId][localID.y][localID.x];
		if (flow0Sad <= bestSad)
		{
			outputVector = int2(0, 0);
		}

		Flow[globalID] = float2(outputVector);
    }
}

