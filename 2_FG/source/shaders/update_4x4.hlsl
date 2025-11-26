#define NUM_THREADS 4

Texture2D<float> ref : register(t0);
Texture2D<float> target : register(t1);
RWTexture2D<float2> flow : register(u0);

groupshared float loss_value[NUM_THREADS][NUM_THREADS][5][2];
groupshared int2 update_flows[NUM_THREADS][NUM_THREADS][5][2];
groupshared int2 best_flow_4[NUM_THREADS][NUM_THREADS];
groupshared int2 best_flow_02[NUM_THREADS][NUM_THREADS], best_flow_42[NUM_THREADS][NUM_THREADS], best_flow_2[NUM_THREADS][NUM_THREADS];
groupshared int2 best_flow_01[NUM_THREADS][NUM_THREADS], best_flow_21[NUM_THREADS][NUM_THREADS], best_flow_1[NUM_THREADS][NUM_THREADS];
groupshared float min_loss_02[NUM_THREADS][NUM_THREADS], min_loss_42[NUM_THREADS][NUM_THREADS];
groupshared float min_loss_01[NUM_THREADS][NUM_THREADS], min_loss_21[NUM_THREADS][NUM_THREADS];

[numthreads(NUM_THREADS, NUM_THREADS, 10)]

void main(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
	int local_Zx = int(localID.z % 5);
	int local_Zy = int(localID.z / 5);
	
	int FlowWidth, FlowHeight;
	flow.GetDimensions(FlowWidth, FlowHeight);
	int width, height;
	ref.GetDimensions(width, height);
	
    int block_x = int(groupID.x * NUM_THREADS + localID.x);
    int block_y = int(groupID.y * NUM_THREADS + localID.y);

	int ref_x = block_x * 8;
	int ref_y = block_y * 8;

	int2 old_blk_flow, update_flow, update_dir;
	float loss;

	if (block_x < FlowWidth && block_y < FlowHeight)
	{
		if (local_Zx == 0)
			update_dir = int2(1, 0);
		else if (local_Zx == 1)
			update_dir = int2(-1, 0);
		else if (local_Zx == 2)
			update_dir = int2(0, 1);
		else if (local_Zx == 3)
			update_dir = int2(0, -1);
		else
			update_dir = int2(0, 0);
	
        old_blk_flow = int2(flow[int2(block_x, block_y)]);

		// -------------------------------- Round 1 -------------------------------
		if (local_Zy == 0) 
			update_flow = old_blk_flow + update_dir * 4;
		else
			update_flow = old_blk_flow + update_dir * 2;

		loss = 0;
		if (local_Zx != 4)
		{
			int target_x = ref_x + update_flow.x;
			int target_y = ref_y + update_flow.y;
			if (target_x >= 0 && target_x <= width - 8 &&
				target_y >= 0 && target_y <= height - 8)
			{
				for (int i = 0; i < 8; i++)
				{
					for (int j = 0; j < 8; j++)
					{
						float ref_pixel = ref.Load(int3(ref_x + j, ref_y + i, 0));
						float target_pixel = target.Load(int3(target_x + j, target_y + i, 0));
						loss += abs(ref_pixel - target_pixel);
					}
				}
			}
			else
			{
				loss = 64;
			}
		}
		else
		{
			loss = 64;
		}

		loss_value[localID.x][localID.y][local_Zx][local_Zy] = loss;
		update_flows[localID.x][localID.y][local_Zx][local_Zy] = update_flow;
	}
	
    GroupMemoryBarrierWithGroupSync();

    if (local_Zx == 0 && block_x < FlowWidth && block_y < FlowHeight)
    {
        float min_loss = 64;
        int2 best_flow;
        for (int i = 0; i < 5; i++)
        {
            if (loss_value[localID.x][localID.y][i][local_Zy] < min_loss)
            {
                min_loss = loss_value[localID.x][localID.y][i][local_Zy];
                best_flow = update_flows[localID.x][localID.y][i][local_Zy];
            }
        }
        if (local_Zy == 0)
        {
            best_flow_4[localID.x][localID.y] = best_flow;
        }
        else
        {
            best_flow_02[localID.x][localID.y] = best_flow;
            min_loss_02[localID.x][localID.y] = min_loss;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // -------------------------------- Round 2 -------------------------------

	if (block_x < FlowWidth && block_y < FlowHeight)
	{
		if (local_Zy == 0)
			update_flow = best_flow_4[localID.x][localID.y] + update_dir * 2;
		else
			update_flow = old_blk_flow + update_dir;
		loss = 0;
		int target_x = ref_x + update_flow.x;
		int target_y = ref_y + update_flow.y;
		if (target_x >= 0 && target_x <= width - 8 &&
			target_y >= 0 && target_y <= height - 8)
		{
			for (int i = 0; i < 8; i++)
			{
				for (int j = 0; j < 8; j++)
				{
					float ref_pixel = ref.Load(int3(ref_x + j, ref_y + i, 0));
					float target_pixel = target.Load(int3(target_x + j, target_y + i, 0));
					loss += abs(ref_pixel - target_pixel);
				}
			}
		}
		else
		{
			loss = 64;
		}

		loss_value[localID.x][localID.y][local_Zx][local_Zy] = loss;
		update_flows[localID.x][localID.y][local_Zx][local_Zy] = update_flow;
	}
	
    GroupMemoryBarrierWithGroupSync();

    if (local_Zx == 0 && block_x < FlowWidth && block_y < FlowHeight)
    {
        if (local_Zy == 0)
            loss_value[localID.x][localID.y][4][0] -= 0.25;
        else
            loss_value[localID.x][localID.y][4][1] -= 0.125;
        float min_loss = 64;
        int2 best_flow;
        for (int i = 0; i < 5; i++)
        {
            if (loss_value[localID.x][localID.y][i][local_Zy] < min_loss)
            {
                min_loss = loss_value[localID.x][localID.y][i][local_Zy];
                best_flow = update_flows[localID.x][localID.y][i][local_Zy];
            }
        }
        if (local_Zy == 0)
        {
            best_flow_42[localID.x][localID.y] = best_flow;
            if (all(best_flow_42[localID.x][localID.y] - best_flow_4[localID.x][localID.y] == int2(0, 0)))
                min_loss += 0.25;
            min_loss_42[localID.x][localID.y] = min_loss;
            if (min_loss_42[localID.x][localID.y] < min_loss_02[localID.x][localID.y])
                best_flow_2[localID.x][localID.y] = best_flow_42[localID.x][localID.y];
            else
                best_flow_2[localID.x][localID.y] = best_flow_02[localID.x][localID.y];
        }
        else
        {
            best_flow_01[localID.x][localID.y] = best_flow;
            if (all(best_flow_01[localID.x][localID.y] - old_blk_flow == int2(0, 0)))
                min_loss += 0.125;
            min_loss_01[localID.x][localID.y] = min_loss;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // -------------------------------- Round 3 -------------------------------

    if (local_Zy == 0 && block_x < FlowWidth && block_y < FlowHeight)
    {
        update_flow = best_flow_2[localID.x][localID.y] + update_dir;
        loss = 0;
        int target_x = ref_x + update_flow.x;
        int target_y = ref_y + update_flow.y;
        if (target_x >= 0 && target_x <= width - 8 &&
            target_y >= 0 && target_y <= height - 8)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    float ref_pixel = ref.Load(int3(ref_x + j, ref_y + i, 0));
                    float target_pixel = target.Load(int3(target_x + j, target_y + i, 0));
                    loss += abs(ref_pixel - target_pixel);
                }
            }
        }
        else
        {
            loss = 64;
        }

        loss_value[localID.x][localID.y][local_Zx][0] = loss;
        update_flows[localID.x][localID.y][local_Zx][0] = update_flow;
    }

    GroupMemoryBarrierWithGroupSync();

    if (localID.z == 0 && block_x < FlowWidth && block_y < FlowHeight)
    {
        loss_value[localID.x][localID.y][4][0] -= 0.125;
        float min_loss = 64;
        int2 best_flow;
        for (int i = 0; i < 5; i++)
        {
            if (loss_value[localID.x][localID.y][i][0] < min_loss)
            {
                min_loss = loss_value[localID.x][localID.y][i][0];
                best_flow = update_flows[localID.x][localID.y][i][0];
            }
        }
        best_flow_21[localID.x][localID.y] = best_flow;
        if (all(best_flow_21[localID.x][localID.y] - best_flow_2[localID.x][localID.y] == int2(0, 0)))
            min_loss += 0.125;
        min_loss_21[localID.x][localID.y] = min_loss;
        int2 offset_21 = best_flow_21[localID.x][localID.y] - old_blk_flow;
        int2 offset_01 = best_flow_01[localID.x][localID.y] - old_blk_flow;
        if (min_loss_21[localID.x][localID.y] + sqrt(offset_21.x * offset_21.x + offset_21.y * offset_21.y) / 4 <
            min_loss_01[localID.x][localID.y] + sqrt(offset_01.x * offset_01.x + offset_01.y * offset_01.y) / 4)
            best_flow_1[localID.x][localID.y] = best_flow_21[localID.x][localID.y];
        else
            best_flow_1[localID.x][localID.y] = best_flow_01[localID.x][localID.y];
    }

    // -------------------------------- Round 4 -------------------------------
    GroupMemoryBarrierWithGroupSync();

    if (local_Zy == 0 && block_x < FlowWidth && block_y < FlowHeight)
    {
        update_flow = best_flow_1[localID.x][localID.y] + update_dir;
        loss = 0;
        int target_x = ref_x + update_flow.x;
        int target_y = ref_y + update_flow.y;
        if (target_x >= 0 && target_x <= width - 8 &&
            target_y >= 0 && target_y <= height - 8)
        {
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    float ref_pixel = ref.Load(int3(ref_x + j, ref_y + i, 0));
                    float target_pixel = target.Load(int3(target_x + j, target_y + i, 0));
                    loss += abs(ref_pixel - target_pixel);
                }
            }
        }
        else
        {
            loss = 64;
        }

        loss_value[localID.x][localID.y][local_Zx][0] = loss;
        update_flows[localID.x][localID.y][local_Zx][0] = update_flow;
    }

    GroupMemoryBarrierWithGroupSync();

    if (localID.z == 0 && block_x < FlowWidth && block_y < FlowHeight)
    {
        loss_value[localID.x][localID.y][4][0] -= 0.125;
        float min_loss = 64;
        int2 best_flow;
        for (int i = 0; i < 5; i++)
        {
            if (loss_value[localID.x][localID.y][i][0] < min_loss)
            {
                min_loss = loss_value[localID.x][localID.y][i][0];
                best_flow = update_flows[localID.x][localID.y][i][0];
            }
        }
        flow[int2(block_x, block_y)] = float2(best_flow);
    }
}

