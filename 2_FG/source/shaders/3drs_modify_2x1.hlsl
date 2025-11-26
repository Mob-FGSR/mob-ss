#define NUM_THREADS 2

Texture2D<float> ref : register(t0);
Texture2D<float> target : register(t1);
RWTexture2D<float2> flow : register(u0);

int now_h;

groupshared float loss_value[NUM_THREADS][5][3];
groupshared int2 update_flows[NUM_THREADS][5][3];
groupshared int2 best_flow_a[NUM_THREADS], best_flow_b[NUM_THREADS], best_flow_g[NUM_THREADS];
groupshared float min_loss_a[NUM_THREADS], min_loss_b[NUM_THREADS], min_loss_g[NUM_THREADS];

[numthreads(NUM_THREADS, 5, 3)]

void main(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
	int FlowWidth, FlowHeight;
	flow.GetDimensions(FlowWidth, FlowHeight);
	
	int block_x = int(groupID.x * NUM_THREADS + localID.x);

	if (block_x < FlowWidth)
	{
		int width, height;
		ref.GetDimensions(width, height);
	
		int ref_x = block_x * 8;
		int ref_y = now_h * 8;
		int2 update_dir;

		if (localID.y == 0)
			update_dir = int2(0, 0);
		else if (localID.y == 1)
			update_dir = int2(1, 0);
		else if (localID.y == 2)
			update_dir = int2(-1, 0);
		else if (localID.y == 3)
			update_dir = int2(0, 1);
		else if (localID.y == 4)
			update_dir = int2(0, -1);
		else
			update_dir = int2(0, 0);

		int2 candi_blk_pos = int2(int(block_x), now_h);
		if (localID.z == 0)
			candi_blk_pos += int2(-1, -1);
		else if (localID.z == 1)
			candi_blk_pos += int2(1, -1);
    
		int2 candi_flow = int2(999, 999);
		float loss = 0;
		if (candi_blk_pos.x >= 0 && candi_blk_pos.x < FlowWidth &&
			candi_blk_pos.y >= 0 && candi_blk_pos.y < FlowHeight)
		{
			candi_flow = int2(flow[int2(candi_blk_pos.x, candi_blk_pos.y)]);
			candi_flow += update_dir;
			int target_x = ref_x + candi_flow.x;
			int target_y = ref_y + candi_flow.y;
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
				if (localID.y != 0)
					loss += 0.125;
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

		loss_value[localID.x][localID.y][localID.z] = loss;
		update_flows[localID.x][localID.y][localID.z] = candi_flow;
	}
	
    GroupMemoryBarrierWithGroupSync();

    if (localID.y == 0 && block_x < FlowWidth)
    {
        float min_loss = 64;
        int2 best_flow;
        for (int i = 0; i < 5; i++)
        {
            if (loss_value[localID.x][i][localID.z] < min_loss)
            {
                min_loss = loss_value[localID.x][i][localID.z];
                best_flow = update_flows[localID.x][i][localID.z];
            }
        }
        if (localID.z == 0)
        {
            best_flow_a[localID.x] = best_flow;
            min_loss_a[localID.x] = min_loss;
        }
        else if (localID.z == 1)
        {
            best_flow_b[localID.x] = best_flow;
            min_loss_b[localID.x] = min_loss;
        }
        else
        {
            best_flow_g[localID.x] = best_flow;
            min_loss_g[localID.x] = min_loss;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (localID.y == 0 && localID.z == 0 && block_x < FlowWidth)
    {
        float min_loss_s;
        int2 best_flow_s;
        if (min_loss_a[localID.x] <= min_loss_b[localID.x])
        {
            min_loss_s = min_loss_a[localID.x];
            best_flow_s = best_flow_a[localID.x];
        }
        else
        {
            min_loss_s = min_loss_b[localID.x];
            best_flow_s = best_flow_b[localID.x];
        }

		if (all(best_flow_g[localID.x] == int2(0, 0)))
        {
            if (min_loss_s < min_loss_g[localID.x]) 
            {
                flow[int2(block_x, now_h)] = float2(best_flow_s);
            }
            else
            {
                flow[int2(block_x, now_h)] = float2(best_flow_g[localID.x]);
            }
        }
        else
        {
            if (min_loss_s <= min_loss_g[localID.x] + 1)
            {
                flow[int2(block_x, now_h)] = float2(best_flow_s);
            }
            else
            {
                flow[int2(block_x, now_h)] = float2(best_flow_g[localID.x]);
            }
        }
    }
}

