#ifndef FLOW_ESTIMATION_COMMON_HLSL
#define FLOW_ESTIMATION_COMMON_HLSL


uint GetPackedLuma(int width, int x, uint luma0, uint luma1, uint luma2, uint luma3)
{
    uint packedLuma = luma0 | (luma1 << 8) | (luma2 << 16) | (luma3 << 24);

    if (x < 0)
    {
        uint outOfScreenFiller = packedLuma & 0xffu;
        if (x <= -1)
            packedLuma = (packedLuma << 8) | outOfScreenFiller;
        if (x <= -2)
            packedLuma = (packedLuma << 8) | outOfScreenFiller;
        if (x <= -3)
            packedLuma = (packedLuma << 8) | outOfScreenFiller;
    }
    else if (x > width - 4)
    {
        uint outOfScreenFiller = packedLuma & 0xff000000u;
        if (x >= width - 3)
            packedLuma = (packedLuma >> 8) | outOfScreenFiller;
        if (x >= width - 2)
            packedLuma = (packedLuma >> 8) | outOfScreenFiller;
        if (x >= width - 1)
            packedLuma = (packedLuma >> 8) | outOfScreenFiller;
    }
    return packedLuma;
}


uint Sad(uint a, uint b)
{
    return abs(int((a >> 0) & 0xffu) - int((b >> 0) & 0xffu)) +
        abs(int((a >> 8) & 0xffu) - int((b >> 8) & 0xffu)) +
        abs(int((a >> 16) & 0xffu) - int((b >> 16) & 0xffu)) +
        abs(int((a >> 24) & 0xffu) - int((b >> 24) & 0xffu));
}

uint4 QSad(uint a0, uint a1, uint b)
{
    uint4 sad;
    sad.x = Sad(a0, b);

    a0 = (a0 >> 8) | ((a1 & 0xffu) << 24);
    a1 >>= 8;
    sad.y = Sad(a0, b);

    a0 = (a0 >> 8) | ((a1 & 0xffu) << 24);
    a1 >>= 8;
    sad.z = Sad(a0, b);

    a0 = (a0 >> 8) | ((a1 & 0xffu) << 24);
    sad.w = Sad(a0, b);
    return sad;

}

uint ABfe(uint src, uint off, uint bits) { uint mask = (1u << bits) - 1u; return (src >> off) & mask; }
uint ABfi(uint src, uint ins, uint mask) { return (ins & mask) | (src & (~mask)); }
uint ABfiM(uint src, uint ins, uint bits) { uint mask = (1u << bits) - 1u; return (ins & mask) | (src & (~mask)); }
void MapThreads(in int2 iGroupId, in int iLocalIndex,
                out int2 iSearchId, out int2 iPxPos, out int iLaneToBlockId)
{
    iSearchId = int2(ABfe(iLocalIndex, 0u, 2u), ABfe(iLocalIndex, 2u, 4u));
    iLaneToBlockId = int(ABfe(iLocalIndex, 1u, 1u) | (ABfe(iLocalIndex, 5u, 1u) << 1u));
    iPxPos = (iGroupId << 4u) + iSearchId * int2(4, 1);
}



#endif // FLOW_ESTIMATION_COMMON_HLSL