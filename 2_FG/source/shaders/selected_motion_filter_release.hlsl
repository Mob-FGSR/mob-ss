Texture2D<float> TargetShadowTexture : register(t0);
Texture2D<float> RefShadowTexture : register(t1);
Texture2D<float2> FlowTexture : register(t2);
Texture2D<float2> MotionVectorTexture : register(t3);
Texture2D<float> SelectMap : register(t4);
Texture2D<float> TargetDynamicTexture : register(t5);
Texture2D<float> RefDynamicTexture : register(t6);

RWTexture2D<float2> CorrectedMotionTexture : register(u0);

SamplerState LinearSampler : register(s0);

[numthreads(8, 8, 1)]

void main(int2 iGlobalId : SV_DispatchThreadID,
		  int2 iLocalId : SV_GroupThreadID,
		  int2 iGroupId : SV_GroupID,
		  int iLocalIndex : SV_GroupIndex)
{
	int width, height;
	MotionVectorTexture.GetDimensions(width, height);

	float2 mv = MotionVectorTexture[iGlobalId];
	float ref_shadow = RefShadowTexture[iGlobalId];
	float target_shadow = TargetShadowTexture[iGlobalId];
	float mv_ref_shadow = TargetShadowTexture[int2(iGlobalId.x - mv.x, iGlobalId.y + mv.y)];
	float flow_valid_shadow_area = max(ref_shadow, target_shadow);
	float mv_wrong_area = max(ref_shadow, mv_ref_shadow);

	float ref_dynamic = RefDynamicTexture[iGlobalId];
	float mv_ref_dynamic = TargetDynamicTexture[int2(iGlobalId.x - mv.x, iGlobalId.y + mv.y)];
	float dynamic_area = max(ref_dynamic, mv_ref_dynamic);

	float2 corrected_motion;

	if (SelectMap[iGlobalId / 8] != 0 && dynamic_area == 0 && mv_wrong_area > 0.1) 
	{
		if (flow_valid_shadow_area > 0.1)
		{
			float2 uv = (float2(iGlobalId) + 0.5) / float2(width, height);
			corrected_motion = -FlowTexture.SampleLevel(LinearSampler, uv, 0);
			corrected_motion.y = -corrected_motion.y;
		}
		else
		{
			corrected_motion = float2(0, 0);
		}
	}
	else
	{
		corrected_motion = mv;
	}
	CorrectedMotionTexture[iGlobalId] = float2(corrected_motion);
}

