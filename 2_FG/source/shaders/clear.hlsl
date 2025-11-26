#define INVALID 0

RWTexture2D<uint> rw_reprojection : register(u0);
RWTexture2D<uint> debug_conflict_count : register(u1);

[numthreads(8, 8, 1)]
void main(int2 iGlobalId : SV_DispatchThreadID)
{
    rw_reprojection[iGlobalId] = INVALID;
    debug_conflict_count[iGlobalId] = 0;
}