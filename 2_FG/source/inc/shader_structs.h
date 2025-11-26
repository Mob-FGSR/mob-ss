#ifndef SHADER_PARAM_STRUCTS_H
#define SHADER_PARAM_STRUCTS_H

struct float2 {
    float x;
    float y;

    float2() 
        : x(0), y(0) {}
    float2(float x, float y) 
        : x(x), y(y) {}
};

struct float4 {
    float x;
    float y;
    float z;
    float w;

    float4()
        : x(0), y(0), z(0), w(0) {}
    float4(float x, float y, float z, float w)
        : x(x), y(y), z(z), w(w) {}
};

struct alignas(16) Modify3drsCBStruct {
    int now_h;

    Modify3drsCBStruct(int now_h = 0)
        :now_h(now_h) {}
};

struct alignas(16) MobFGSRCBStruct {
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

#endif // SHADER_PARAM_STRUCTS_H
