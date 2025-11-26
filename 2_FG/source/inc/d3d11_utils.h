#ifndef D3D11_UTILS_H
#define D3D11_UTILS_H

#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl.h>
#include <iostream>
#include <fstream>

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Microsoft::WRL; 


// Initialize D3D11
bool initializeD3D11(ComPtr<ID3D11Device>& device, ComPtr<ID3D11DeviceContext>& context) {
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_DEBUG,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &device,
        nullptr,
        &context
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device!" << std::endl;
        return false;
    }

    return true;
}


ComPtr<ID3D11ComputeShader> createComputeShader(ComPtr<ID3D11Device> device, std::string filename) {
    ComPtr<ID3DBlob> shaderBlob;
    ComPtr<ID3DBlob> errorBlob;
    HRESULT hr = D3DCompileFromFile(std::filesystem::path(filename).c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0", 0, 0, &shaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Failed to compile compute shader: " << static_cast<const char*>(errorBlob->GetBufferPointer()) << std::endl;
        } else {
            std::cerr << "Failed to compile compute shader: HRESULT: 0x" << std::hex << hr << std::dec << std::endl;
        }
        return nullptr;
    }

    ComPtr<ID3D11ComputeShader> shader;
    hr = device->CreateComputeShader(shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr, &shader);
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader!" << std::endl;
        return nullptr;
    }
    return shader;
}


// Create shader resource view
ComPtr<ID3D11ShaderResourceView> createSRV(ComPtr<ID3D11Device> device, ComPtr<ID3D11Texture2D> texture, UINT mipLevels = 1, UINT mostDetailedMip = 0) {
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    D3D11_TEXTURE2D_DESC textureDesc;
    texture->GetDesc(&textureDesc);

    srvDesc.Format = textureDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = mipLevels;
    srvDesc.Texture2D.MostDetailedMip = mostDetailedMip;

    ComPtr<ID3D11ShaderResourceView> srv;
    HRESULT hr = device->CreateShaderResourceView(texture.Get(), &srvDesc, &srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create srv!" << std::endl;
        return nullptr;
    }
    return srv;
}


// Create unordered access view
ComPtr<ID3D11UnorderedAccessView> createUAV(ComPtr<ID3D11Device> device, ComPtr<ID3D11Texture2D> texture, UINT mipLevel = 0) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    D3D11_TEXTURE2D_DESC textureDesc;
    texture->GetDesc(&textureDesc);

    uavDesc.Format = textureDesc.Format;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Texture2D.MipSlice = mipLevel;

    ComPtr<ID3D11UnorderedAccessView> uav;
    HRESULT hr = device->CreateUnorderedAccessView(texture.Get(), &uavDesc, &uav);
    if (FAILED(hr)) {
        std::cerr << "Failed to create uav!" << std::endl;
        return nullptr;
    }
    return uav;
}


// Create constant buffer
ComPtr<ID3D11Buffer> createConstantBuffer(ComPtr<ID3D11Device> device, UINT byteWidth) {
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = (byteWidth + 15) & ~15;
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = 0;

    ComPtr<ID3D11Buffer> cb;
    HRESULT hr = device->CreateBuffer(&cbDesc, nullptr, &cb);
    if (FAILED(hr)) {
        std::cerr << "Failed to create cb!" << std::endl;
        return nullptr;
    }
    return cb;
}


// Create texture
ComPtr<ID3D11Texture2D> createTexture(ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> context, 
                                    int width, int height, DXGI_FORMAT format, bool autogen, void* data = nullptr) {

    UINT fmtSupport = 0;
    HRESULT hr = device->CheckFormatSupport(format, &fmtSupport);
    if (FAILED(hr))
    {
        std::cerr << "format not supported!" << std::endl;
        exit(0);
    }
    if (autogen)
    {
        if (!(fmtSupport & D3D11_FORMAT_SUPPORT_MIP_AUTOGEN))
        {
            std::cerr << "not support mip auto-gen" << std::endl;
            exit(0);
        }
    }

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = autogen ? 0 : 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS | (autogen ? D3D11_BIND_RENDER_TARGET : 0);
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = autogen ? D3D11_RESOURCE_MISC_GENERATE_MIPS : 0;

    D3D11_SUBRESOURCE_DATA initData = {};
    if (data) {
        initData.pSysMem = data;
        UINT bytesPerPixel = 0;
        switch (format) {
            case DXGI_FORMAT_R8_UNORM:            bytesPerPixel = 1; break;
            case DXGI_FORMAT_R16_FLOAT:           bytesPerPixel = 2; break;
            case DXGI_FORMAT_R8G8B8A8_UNORM:      bytesPerPixel = 4; break;
            case DXGI_FORMAT_R32_FLOAT:           bytesPerPixel = 4; break;
            case DXGI_FORMAT_R16G16_FLOAT:        bytesPerPixel = 4; break;
            case DXGI_FORMAT_R16G16B16A16_FLOAT:  bytesPerPixel = 8; break;
            case DXGI_FORMAT_R32G32_FLOAT:        bytesPerPixel = 8; break;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:  bytesPerPixel = 16; break;
            default:
                std::cerr << "Unsupported format!" << std::endl;
                return nullptr;
        }
        initData.SysMemPitch = width * bytesPerPixel;
    }

    ComPtr<ID3D11Texture2D> texture;
    hr = device->CreateTexture2D(&desc, (data && !autogen)? &initData : nullptr, &texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create texture! HRESULT: 0x" << std::hex << hr << std::dec << std::endl;
        return nullptr;
    }

    if (autogen) {
        if (data)
            context->UpdateSubresource(texture.Get(), 0, nullptr, data, initData.SysMemPitch, 0);
        context->GenerateMips(createSRV(device, texture, unsigned(-1), 0).Get());
    }

    return texture;
}


// Get texture size according to level
void getTextureSize(ComPtr<ID3D11Texture2D> texture, int level, int& width, int& height) {
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    width = desc.Width >> level;
    height = desc.Height >> level;
}


void save_rgba8(ComPtr<ID3D11Device> device, ComPtr<ID3D11DeviceContext> context, 
                ComPtr<ID3D11Texture2D> texture, int level, const char* savePath) {
    int width, height;
    getTextureSize(texture, level, width, height);

    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> stagingTexture;
    device->CreateTexture2D(&desc, nullptr, &stagingTexture);
    context->CopySubresourceRegion(stagingTexture.Get(), 0, 0, 0, 0, texture.Get(), level, nullptr);

    D3D11_MAPPED_SUBRESOURCE mapped;
    context->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    int memWidth = mapped.RowPitch / 4;
    unsigned char* uint8_rgba_read = new unsigned char[memWidth * height * 4];
    memcpy(uint8_rgba_read, mapped.pData, memWidth * height * 4);
    context->Unmap(stagingTexture.Get(), 0);

    unsigned char* uint8_rgba_buffer = new unsigned char[width * height * 4];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
        int texPixelIndex = i * width + j;
        int memPixelIndex = i * memWidth + j;
            uint8_rgba_buffer[texPixelIndex * 4    ] = uint8_rgba_read[memPixelIndex * 4    ];
            uint8_rgba_buffer[texPixelIndex * 4 + 1] = uint8_rgba_read[memPixelIndex * 4 + 1];
            uint8_rgba_buffer[texPixelIndex * 4 + 2] = uint8_rgba_read[memPixelIndex * 4 + 2];
            uint8_rgba_buffer[texPixelIndex * 4 + 3] = uint8_rgba_read[memPixelIndex * 4 + 3];
        }
    }

    if (!stbi_write_png(savePath, width, height, 4, uint8_rgba_buffer, width * 4))
        std::cout << "rgba8 saved fail!" << std::endl;

    delete[] uint8_rgba_read;
    delete[] uint8_rgba_buffer;
}


ComPtr<ID3D11SamplerState> createSamplerState(ComPtr<ID3D11Device> device, D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE addressMode) {
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = filter;
    samplerDesc.AddressU = addressMode;
    samplerDesc.AddressV = addressMode;
    samplerDesc.AddressW = addressMode;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    ComPtr<ID3D11SamplerState> samplerState;
    HRESULT hr = device->CreateSamplerState(&samplerDesc, &samplerState);
    if (FAILED(hr)) {
        std::cerr << "Failed to create sampler state!" << std::endl;
        return nullptr;
    }
    return samplerState;
}


bool updateTextureWithPNG(
    ComPtr<ID3D11Device> device, 
    ComPtr<ID3D11DeviceContext> context,
    ComPtr<ID3D11Texture2D> texture, 
    std::string picturePath,
    bool genMips)
{
    int width, height, channels;
    unsigned char* data = stbi_load(picturePath.c_str(), &width, &height, &channels, 0);
    if (!data) 
    {
        std::cerr << "Failed to load data from " << picturePath << ": " << stbi_failure_reason() << std::endl; 
        return false;
    }
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);

    if (desc.Format != DXGI_FORMAT_R8_UNORM &&
        desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        std::cerr << "Texture format is not supported for this function!\n";
        stbi_image_free(data);
        return false;
    }

    if (width != desc.Width || height != desc.Height)
    {
        std::cerr << "Failed to match size: " << picturePath << std::endl; 
        return false;    
    }

    if (channels == 1 && desc.Format == DXGI_FORMAT_R8_UNORM ||
        channels == 4 && desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        context->UpdateSubresource(texture.Get(), 0, nullptr, data, width * channels, 0);
        stbi_image_free(data);
    }
    else if (channels == 3 && desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        unsigned char* rgbaData = new unsigned char[width * height * 4];
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int rgbIndex = (i * width + j) * 3;
                int rgbaIndex = (i * width + j) * 4;
                rgbaData[rgbaIndex + 0] = data[rgbIndex + 0];
                rgbaData[rgbaIndex + 1] = data[rgbIndex + 1];
                rgbaData[rgbaIndex + 2] = data[rgbIndex + 2];
                rgbaData[rgbaIndex + 3] = 255;
            }
        }
        context->UpdateSubresource(texture.Get(), 0, nullptr, rgbaData, width * 4, 0);
        delete[] rgbaData;
        stbi_image_free(data);
    }
    else
    {
        std::cerr << "Channel count does not match texture format: " << picturePath << std::endl; 
        stbi_image_free(data);
        return false;
    }

    if (genMips)
        context->GenerateMips(createSRV(device, texture, unsigned(-1), 0).Get());

    return true;

}


float* loadEXRChannelData(const char* filename, int channel)
{
	int width, height;
	float* data;
	const char* err;
	int ret = LoadEXR(&data, &width, &height, filename, &err);
	if (ret < 0)
	{
		std::cerr << err << "\n";
		return nullptr;
	}

	float* resizedData = new float[width * height];
	for (int i = 0; i < width * height; i++)
	{
		resizedData[i] = data[i * 4 + channel];
	}
	free(data);

	return resizedData;
}

#endif