#define NUM_THREADS 17

Texture2D<float> ref : register(t0);
Texture2D<float> target : register(t1);
RWTexture2D<float2> flow : register(u0);

groupshared float loss_value[NUM_THREADS][NUM_THREADS];

[numthreads(NUM_THREADS, NUM_THREADS, 1)]

void main(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {

    int ref_x = groupID.x * 8;
    int ref_y = groupID.y * 8;
    int offset_x = localID.x - 8;
    int offset_y = localID.y - 8;
    int target_x = ref_x + offset_x;
    int target_y = ref_y + offset_y;
    float loss = 0;

    int width, height;
    ref.GetDimensions(width, height);

    if (target_x >= 0 && target_x <= width - 8 &&
        target_y >= 0 && target_y <= height - 8) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                float ref_pixel = ref.Load(int3(ref_x + j, ref_y + i, 0));
                float target_pixel = target.Load(int3(target_x + j, target_y + i, 0));
                loss += abs(ref_pixel - target_pixel);
            }
        }
        loss += sqrt(offset_x * offset_x + offset_y * offset_y) / 4;
    } else {
        loss = 64;
    }

    loss_value[localID.y][localID.x] = loss;
    GroupMemoryBarrierWithGroupSync();

    if (localID.x == 0 && localID.y == 0) {
        float min_loss = 64;
        int best_dx = 0, best_dy = 0;

        if (loss_value[8][8] < 0.2)
        {
            flow[int2(groupID.x, groupID.y)] = float2(0, 0);
            return;          
        }
        for (int y = 0; y < 17; y++) {
            for (int x = 0; x < 17; x++) {
                if (loss_value[y][x] < min_loss) {
                    min_loss = loss_value[y][x];
                    best_dx = x - 8;
                    best_dy = y - 8;
                }
            }
        }
        flow[int2(groupID.x, groupID.y)] = float2(best_dx, best_dy);
    }
}

