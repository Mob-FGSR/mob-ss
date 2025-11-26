#define NOMINMAX
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <format>
#include <algorithm>
#include "d3d11_utils.h"
#include "shader_structs.h"
#include <DirectXPackedVector.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace fs = std::filesystem;

ComPtr<ID3D11Device> device;
ComPtr<ID3D11DeviceContext> context;

int main(int argc, char* argv[]) {

    if (!initializeD3D11(device, context)) return -1;

    if (argc != 4) {
        std::cerr << "Usage: <exe> <sequence> <start_frame> <end_frame>\n";
        std::cerr << "Example: .\\build\\Debug\\fg.exe BK 6 10 fg\n";
        return -1;
    }

    std::string seq = argv[1];
    int start_frame = std::stoi(argv[2]);
    int end_frame   = std::stoi(argv[3]);;
    if (start_frame >= end_frame) return 0;
    
    /******************** Config *********************/
    std::string shader_dir         = std::format("../../source/shaders");
    std::string resource_dir       = std::format("../../../0_DATA/{}", seq);
    std::string color_dir          = std::format("{}/1080p_vsr", resource_dir);
    std::string depth_dir          = std::format("{}/540p_depth", resource_dir);
    std::string motion_vector_dir  = std::format("{}/540p_motion_vector", resource_dir);
    std::string dynamic_mask_dir   = std::format("{}/540p_dynamic_mask", resource_dir);
    std::string shadow_dir         = std::format("{}/540p_shadow", resource_dir);
    std::string output_dir         = std::format("{}/1080p_vsr_itp", resource_dir);

    int width = 960;
    int height = 540;
    float sr_scale = 2.0f;

    int flow_width = width / 8;
    int flow_height = height / 8;
    int present_width = static_cast<int>(static_cast<float>(width) * sr_scale);
    int present_height = static_cast<int>(static_cast<float>(height) * sr_scale);

    std::filesystem::path source_file = __FILE__;
    std::filesystem::path source_dir  = source_file.parent_path();
    std::filesystem::current_path(source_dir);

    if (!fs::exists(output_dir)) fs::create_directories(output_dir);

    // ---------------------------------------- Prepare Shaders And Textures -------------------------------------
    // compute shaders for data processing
    auto unpack_mv_CS                   = createComputeShader(device, std::format("{}/unpack_mv.hlsl", shader_dir));      
    auto process_mv_CS                  = createComputeShader(device, std::format("{}/process_mv.hlsl", shader_dir));
    auto add_dynamic_depth_CS           = createComputeShader(device, std::format("{}/process_depth.hlsl", shader_dir));
    auto add_shadow_depth_CS            = createComputeShader(device, std::format("{}/add_shadow_depth.hlsl", shader_dir));
    auto sub_shadow_depth_CS            = createComputeShader(device, std::format("{}/sub_shadow_depth.hlsl", shader_dir));
    auto dilate_CS                      = createComputeShader(device, std::format("{}/dilate.hlsl", shader_dir));
    auto dilate_shadow_CS               = createComputeShader(device, std::format("{}/dilate_shadow.hlsl", shader_dir));

    // compute shaders for optical flow estimation
    auto BMA_CS                         = createComputeShader(device, std::format("{}/basic_BMA.hlsl", shader_dir));
    auto upscale_4x4x4_CS               = createComputeShader(device, std::format("{}/upscale_4x4x4.hlsl", shader_dir));
    auto update_4x4_CS                  = createComputeShader(device, std::format("{}/update_4x4.hlsl", shader_dir));
    auto modify_3drs_2x1_CS             = createComputeShader(device, std::format("{}/3drs_modify_2x1.hlsl", shader_dir));

    // compute shaders for motion selection
    auto select_mv_or_flow_CS           = createComputeShader(device, std::format("{}/select_mv_or_flow_release.hlsl", shader_dir));
    auto selected_motion_filter_CS      = createComputeShader(device, std::format("{}/selected_motion_filter_release.hlsl", shader_dir));

    // compute shaders for interpolation
    auto clear_CS                       = createComputeShader(device, std::format("{}/clear.hlsl", shader_dir));
    auto reproject_i_CS                 = createComputeShader(device, std::format("{}/reproject_i.hlsl", shader_dir));
    auto fill_CS                        = createComputeShader(device, std::format("{}/fill.hlsl", shader_dir));
    auto warp_i_CS                      = createComputeShader(device, std::format("{}/warp_i.hlsl", shader_dir));

    // textures
    auto frame0_rgba_texture                 = createTexture(device, context, present_width, present_height, DXGI_FORMAT_R8G8B8A8_UNORM,     false,  nullptr);
    auto frame1_rgba_texture                 = createTexture(device, context, present_width, present_height, DXGI_FORMAT_R8G8B8A8_UNORM,     false,  nullptr);
    
    auto frame0_depth_texture                = createTexture(device, context, width,         height,         DXGI_FORMAT_R32_FLOAT,          false,  nullptr);
    auto frame1_depth_texture                = createTexture(device, context, width,         height,         DXGI_FORMAT_R32_FLOAT,          false,  nullptr);
    auto frame0_dilated_depth_texture        = createTexture(device, context, width,         height,         DXGI_FORMAT_R32_FLOAT,          false,  nullptr);
    auto frame1_dilated_depth_texture        = createTexture(device, context, width,         height,         DXGI_FORMAT_R32_FLOAT,          false,  nullptr);
    
    auto frame0_shadow_texture               = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           true,   nullptr);
    auto frame1_shadow_texture               = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           true,   nullptr);
    auto frame0_dilated_shadow_texture       = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           false,  nullptr);
    auto frame1_dilated_shadow_texture       = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           false,  nullptr);

    auto frame0_dynamic_mask_texture         = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           false,  nullptr);
    auto frame1_dynamic_mask_texture         = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           false,  nullptr);
    auto frame0_dilated_dynamic_mask_texture = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           false,  nullptr);
    auto frame1_dilated_dynamic_mask_texture = createTexture(device, context, width,         height,         DXGI_FORMAT_R8_UNORM,           false,  nullptr);

    auto packed_mv_texture                   = createTexture(device, context, width,         height,         DXGI_FORMAT_R8G8B8A8_UNORM,     false,  nullptr);
    auto frame0_mv_texture                   = createTexture(device, context, width,         height,         DXGI_FORMAT_R16G16_FLOAT,       false,  nullptr);
    auto frame1_mv_texture                   = createTexture(device, context, width,         height,         DXGI_FORMAT_R16G16_FLOAT,       false,  nullptr);  
    auto frame1_dilated_mv_texture           = createTexture(device, context, width,         height,         DXGI_FORMAT_R16G16_FLOAT,       false,  nullptr);

    auto frame1_flow_texture                 = createTexture(device, context, flow_width,    flow_height,    DXGI_FORMAT_R16G16_FLOAT,       true,   nullptr);
    auto select_map                          = createTexture(device, context, flow_width,    flow_height,    DXGI_FORMAT_R8_UNORM,           false,  nullptr);
    auto frame0_final_motion_texture         = createTexture(device, context, width,         height,         DXGI_FORMAT_R16G16_FLOAT,       false,  nullptr);
    auto frame1_final_motion_texture         = createTexture(device, context, width,         height,         DXGI_FORMAT_R16G16_FLOAT,       false,  nullptr);

    auto reprojection_texture                = createTexture(device, context, width,         height,         DXGI_FORMAT_R32_UINT,           false,  nullptr);
    auto filled_reprojection_texture         = createTexture(device, context, width,         height,         DXGI_FORMAT_R32_UINT,           false,  nullptr);
    auto sample_lut_texture                  = createTexture(device, context, 128,           128,            DXGI_FORMAT_R16_FLOAT,          false,  nullptr);
    auto inter_frame_rgba_texture            = createTexture(device, context, present_width, present_height, DXGI_FORMAT_R8G8B8A8_UNORM,     false,  nullptr);
    
    // create constant buffer
    ComPtr<ID3D11Buffer> modify_3drs_constant_buffer = createConstantBuffer(device, sizeof(Modify3drsCBStruct));
    Modify3drsCBStruct modify_3drs_cb_data(0);
    ComPtr<ID3D11Buffer> fg_constant_buffer = createConstantBuffer(device, sizeof(MobFGSRCBStruct));
    MobFGSRCBStruct fg_cb_data;

    fg_cb_data.render_size.x = static_cast<float>(width);
    fg_cb_data.render_size.y = static_cast<float>(height);
    fg_cb_data.render_size.z = 1.0f / static_cast<float>(width);         
    fg_cb_data.render_size.w = 1.0f / static_cast<float>(height);
    fg_cb_data.presentation_size.x = static_cast<float>(present_width);          
    fg_cb_data.presentation_size.y = static_cast<float>(present_height);         
    fg_cb_data.presentation_size.z = 1.0f / static_cast<float>(present_width);   
    fg_cb_data.presentation_size.w = 1.0f / static_cast<float>(present_height);  
    float delta = 0.5f;
    fg_cb_data.delta.x = delta;
    fg_cb_data.delta.y = delta * 0.5f;
    fg_cb_data.delta.z = delta * 1.5f;
    fg_cb_data.delta.w = delta * delta * 0.5f;
    fg_cb_data.jitter_offset.x = 0.f;
    fg_cb_data.jitter_offset.y = 0.f;
    fg_cb_data.depth_diff_threshold_sr = 0.01f;
    fg_cb_data.color_diff_threshold_fg = 0.01f;
    fg_cb_data.depth_diff_threshold_fg = 0.004f;
    fg_cb_data.depth_scale = 1.0f;
    fg_cb_data.depth_bias = 0.0f;
    fg_cb_data.render_scale = sr_scale;

    // create sampler state
    ComPtr<ID3D11SamplerState> linear_sampler = createSamplerState(device, D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP);
    
    ComPtr<ID3D11UnorderedAccessView> null_UAV = nullptr;

    // ---------------------------------------- Load Texture Data of First Frame -------------------------------------

    int output_frame = start_frame * 2;

    if (!updateTextureWithPNG(device, context, frame0_shadow_texture,       std::format("{}/{:04d}.png", shadow_dir,        start_frame), true) ||
        !updateTextureWithPNG(device, context, frame0_rgba_texture,         std::format("{}/{:04d}.png", color_dir,         start_frame), false) ||
        !updateTextureWithPNG(device, context, frame0_dynamic_mask_texture, std::format("{}/{:04d}.png", dynamic_mask_dir,  start_frame), false) ||
        !updateTextureWithPNG(device, context, packed_mv_texture,           std::format("{}/{:04d}.png", motion_vector_dir, start_frame), false))
    {
        std::cout << "Failed to update texture data!" << std::endl;
        return 0;
    }

    {
        std::string frame0_depth_path = std::format("{}/{:04d}.png", depth_dir, start_frame);
        int width, height, channels;
        unsigned char* data = stbi_load(frame0_depth_path.c_str(), &width, &height, &channels, 0);
        float* depth_float_data = new float[width * height];
        for (int i = 0; i < width * height; i++)
        {
            depth_float_data[i] = static_cast<float>(data[i]) / 255.0f / 10.0f;
        }
        context->UpdateSubresource(frame0_depth_texture.Get(), 0, nullptr, static_cast<void*>(depth_float_data), width * sizeof(float), 0);
        delete[] depth_float_data;
        stbi_image_free(data);
    }

    {
        float* sample_lut_float_data = loadEXRChannelData(std::format("{}/lut.exr", shader_dir).c_str(), 3);
        if (!sample_lut_float_data) {std::cerr << "Failed to load lut data!" << std::endl; return 0;}
        uint16_t* sample_lut_data = new uint16_t[128 * 128];
        for (int i = 0; i < 128 * 128; i++)
        {
            sample_lut_data[i] = DirectX::PackedVector::XMConvertFloatToHalf(sample_lut_float_data[i]);
        }
        context->UpdateSubresource(sample_lut_texture.Get(), 0, nullptr, static_cast<void*>(sample_lut_data), 128 * 2, 0);
        delete[] sample_lut_data;
        free(sample_lut_float_data);
    }
    
    context->CSSetShader(unpack_mv_CS.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, createSRV(device, packed_mv_texture, 1, 0).GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame0_mv_texture, 0).GetAddressOf(), nullptr);
    context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
    context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
    
    context->CSSetShader(dilate_shadow_CS.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, createSRV(device, frame0_shadow_texture, 1, 0).GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame0_dilated_shadow_texture, 0).GetAddressOf(), nullptr);
    context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
    context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

    context->CSSetShader(dilate_CS.Get(), nullptr, 0);
    context->CSSetShaderResources(0, 1, createSRV(device, frame0_dynamic_mask_texture, 1, 0).GetAddressOf());
    context->CSSetShaderResources(1, 1, createSRV(device, frame0_depth_texture, 1, 0).GetAddressOf());
    context->CSSetShaderResources(2, 1, createSRV(device, frame0_mv_texture, 1, 0).GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame0_dilated_dynamic_mask_texture, 0).GetAddressOf(), nullptr);
    context->CSSetUnorderedAccessViews(1, 1, createUAV(device, frame0_dilated_depth_texture, 0).GetAddressOf(), nullptr);
    context->CSSetUnorderedAccessViews(2, 1, createUAV(device, frame0_final_motion_texture, 0).GetAddressOf(), nullptr);
    context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
    context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
    context->CSSetUnorderedAccessViews(1, 1, null_UAV.GetAddressOf(), nullptr);
    context->CSSetUnorderedAccessViews(2, 1, null_UAV.GetAddressOf(), nullptr);
    
    context->CSSetShader(process_mv_CS.Get(), nullptr, 0);
    context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame0_final_motion_texture, 0).GetAddressOf(), nullptr);
    context->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
    
    context->CSSetShader(add_dynamic_depth_CS.Get(), nullptr, 0);
    context->CSSetShaderResources(1, 1, createSRV(device, frame0_dilated_dynamic_mask_texture, 1, 0).GetAddressOf());
    context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame0_dilated_depth_texture, 0).GetAddressOf(), nullptr);
    context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
    context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
    
    save_rgba8(device, context, frame0_rgba_texture, 0, std::format("{}/{:04d}.png", output_dir, output_frame).c_str());
    
    // ---------------------------------------- Begin Loops -------------------------------------
    
    context->UpdateSubresource(fg_constant_buffer.Get(), 0, nullptr, &fg_cb_data, 0, 0);
    context->CSSetConstantBuffers(0, 1, fg_constant_buffer.GetAddressOf());

    for (int cur_frame = start_frame + 1; cur_frame <= end_frame; cur_frame++)
    {   
        std::cout << "calculating frame:" << cur_frame << std::endl;

        output_frame = cur_frame * 2;

        // ---------------------------------------- Preparation -------------------------------------
        if (!updateTextureWithPNG(device, context, frame1_shadow_texture,       std::format("{}/{:04d}.png", shadow_dir,        cur_frame), true) ||
            !updateTextureWithPNG(device, context, frame1_rgba_texture,         std::format("{}/{:04d}.png", color_dir,         cur_frame), false) ||
            !updateTextureWithPNG(device, context, frame1_dynamic_mask_texture, std::format("{}/{:04d}.png", dynamic_mask_dir,  cur_frame), false) ||
            !updateTextureWithPNG(device, context, packed_mv_texture,           std::format("{}/{:04d}.png", motion_vector_dir, cur_frame), false))
        {
            std::cout << "Failed to update texture data!" << std::endl;
            return 0;
        }

        {
            std::string frame1_depth_path = std::format("{}/{:04d}.png", depth_dir, cur_frame);
            int width, height, channels;
            unsigned char* data = stbi_load(frame1_depth_path.c_str(), &width, &height, &channels, 0);
            float* depth_float_data = new float[width * height];
            for (int i = 0; i < width * height; i++)
            {
                depth_float_data[i] = static_cast<float>(data[i]) / 255.0f / 10.0f;
            }
            context->UpdateSubresource(frame1_depth_texture.Get(), 0, nullptr, static_cast<void*>(depth_float_data), width * sizeof(float), 0);
            delete[] depth_float_data;
            stbi_image_free(data);
        }
        
        context->CSSetShader(unpack_mv_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, packed_mv_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_mv_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(dilate_shadow_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_dilated_shadow_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(dilate_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_dynamic_mask_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame1_depth_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(2, 1, createSRV(device, frame1_mv_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_dilated_dynamic_mask_texture, 0).GetAddressOf(), nullptr);
        context->CSSetUnorderedAccessViews(1, 1, createUAV(device, frame1_dilated_depth_texture, 0).GetAddressOf(), nullptr);
        context->CSSetUnorderedAccessViews(2, 1, createUAV(device, frame1_dilated_mv_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
        context->CSSetUnorderedAccessViews(1, 1, null_UAV.GetAddressOf(), nullptr);
        context->CSSetUnorderedAccessViews(2, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(add_dynamic_depth_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(1, 1, createSRV(device, frame1_dilated_dynamic_mask_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_dilated_depth_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        save_rgba8(device, context, frame1_rgba_texture, 0, std::format("{}/{:04d}.png", output_dir, output_frame).c_str());

        // ---------------------------------------- Calculate Optical Flow -------------------------------------

        // LEVEL 4
        context->CSSetShader(BMA_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 4).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame0_shadow_texture, 1, 4).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 4).GetAddressOf(), nullptr);
        context->Dispatch(flow_width >> 4, flow_height >> 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // LEVEL 3
        context->CSSetShader(upscale_4x4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(2, 1, createSRV(device, frame1_flow_texture, 1, 4).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 3).GetAddressOf(), nullptr);
        context->Dispatch(((flow_width >> 3) + 3) / 4, ((flow_height >> 3) + 3) / 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(update_4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 3).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame0_shadow_texture, 1, 3).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 3).GetAddressOf(), nullptr);
        context->Dispatch(((flow_width >> 3) + 3) / 4, ((flow_height >> 3) + 3) / 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // LEVEL2
        context->CSSetShader(upscale_4x4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(2, 1, createSRV(device, frame1_flow_texture, 1, 3).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 2).GetAddressOf(), nullptr);
        context->Dispatch(((flow_width >> 2) + 3) / 4, ((flow_height >> 2) + 3) / 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(update_4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 2).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame0_shadow_texture, 1, 2).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 2).GetAddressOf(), nullptr);
        context->Dispatch(((flow_width >> 2) + 3) / 4, ((flow_height >> 2) + 3) / 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // LEVEL 1
        context->CSSetShader(upscale_4x4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(2, 1, createSRV(device, frame1_flow_texture, 1, 2).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 1).GetAddressOf(), nullptr);
        context->Dispatch(((flow_width >> 1) + 3) / 4, ((flow_height >> 1) + 3) / 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(modify_3drs_2x1_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 1).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame0_shadow_texture, 1, 1).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 1).GetAddressOf(), nullptr);
        for (int i = 0; i < (flow_height >> 1); i++) {
            modify_3drs_cb_data.now_h = i;
            context->UpdateSubresource(modify_3drs_constant_buffer.Get(), 0, nullptr, &modify_3drs_cb_data, 0, 0);
            context->CSSetConstantBuffers(0, 1, modify_3drs_constant_buffer.GetAddressOf());
            context->Dispatch(((flow_width >> 1) + 1) / 2, 1, 1);
        }
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // LEVEL 0
        context->CSSetShader(upscale_4x4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(2, 1, createSRV(device, frame1_flow_texture, 1, 1).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((flow_width + 3) / 4, (flow_height + 3) / 4, 1);

        context->CSSetShader(update_4x4_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame0_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_flow_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch(((flow_width >> 0) + 3) / 4, ((flow_height >> 0) + 3) / 4, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // ---------------------------------------- Select Optical Flow or MV-------------------------------------

        context->CSSetShader(select_mv_or_flow_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame0_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame1_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(2, 1, createSRV(device, frame1_flow_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(3, 1, createSRV(device, frame1_dilated_mv_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, select_map, 0).GetAddressOf(), nullptr);
        context->CSSetSamplers(0, 1, linear_sampler.GetAddressOf());
        context->Dispatch(flow_width, flow_height, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
    
        context->CSSetShader(selected_motion_filter_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame0_dilated_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame1_dilated_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(4, 1, createSRV(device, select_map, 1, 0).GetAddressOf());
        context->CSSetShaderResources(5, 1, createSRV(device, frame0_dilated_dynamic_mask_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(6, 1, createSRV(device, frame1_dilated_dynamic_mask_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_final_motion_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // ---------------------------------------- Frame Interpolate -------------------------------------

        // prepare data
        context->CSSetShader(process_mv_CS.Get(), nullptr, 0);
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_final_motion_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // clear
        context->CSSetShader(clear_CS.Get(), nullptr, 0);
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, reprojection_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // reprojection i-frame
        context->CSSetShader(add_shadow_depth_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_dilated_depth_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        context->CSSetShader(reproject_i_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_dilated_depth_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame1_final_motion_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(2, 1, createSRV(device, frame0_final_motion_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, reprojection_texture, 0).GetAddressOf(), nullptr);
        context->CSSetConstantBuffers(0, 1, fg_constant_buffer.GetAddressOf());
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);
    
        context->CSSetShader(sub_shadow_depth_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, frame1_shadow_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, frame1_dilated_depth_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // fill
        context->CSSetShader(fill_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, reprojection_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, filled_reprojection_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((width + 7) / 8,  (height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        // warp i-frame
        context->CSSetShader(warp_i_CS.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, createSRV(device, filled_reprojection_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(1, 1, createSRV(device, frame1_rgba_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(2, 1, createSRV(device, frame0_rgba_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(3, 1, createSRV(device, frame1_dilated_depth_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(4, 1, createSRV(device, frame0_dilated_depth_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(5, 1, createSRV(device, frame1_final_motion_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(6, 1, createSRV(device, frame0_final_motion_texture, 1, 0).GetAddressOf());
        context->CSSetShaderResources(7, 1, createSRV(device, sample_lut_texture, 1, 0).GetAddressOf());
        context->CSSetUnorderedAccessViews(0, 1, createUAV(device, inter_frame_rgba_texture, 0).GetAddressOf(), nullptr);
        context->Dispatch((present_width + 7) / 8,  (present_height + 7) / 8, 1);
        context->CSSetUnorderedAccessViews(0, 1, null_UAV.GetAddressOf(), nullptr);

        save_rgba8(device, context, inter_frame_rgba_texture, 0, std::format("{}/{:04d}.png", output_dir, output_frame - 1).c_str());

        // --------------------------------------------------------------------------------------------
        
        context->CopyResource(frame0_shadow_texture.Get(), frame1_shadow_texture.Get());
        context->CopyResource(frame0_rgba_texture.Get(), frame1_rgba_texture.Get());
        context->CopyResource(frame0_final_motion_texture.Get(), frame1_final_motion_texture.Get());
        context->CopyResource(frame0_dilated_dynamic_mask_texture.Get(), frame1_dilated_dynamic_mask_texture.Get());
        context->CopyResource(frame0_dilated_shadow_texture.Get(), frame1_dilated_shadow_texture.Get());
        context->CopyResource(frame0_dilated_depth_texture.Get(), frame1_dilated_depth_texture.Get());
    }

    return 0;
}
