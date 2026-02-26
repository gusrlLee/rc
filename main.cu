// OpenGL
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// cuda c++
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <algorithm>

#include "math.cuh"
#include "structs.cuh"
#include "kernel.cuh"

__global__ void cascadeTestKernel(cudaSurfaceObject_t surface, int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // 테스트용: 화면 좌표를 기반으로 색상 생성
        float u = (float)x / (float)width;
        float v = (float)y / (float)height;
        float4 color = make_float4(u, v, 0.5f, 1.0f); // R, G, B, A

        // surf2Dwrite는 x 좌표를 바이트 단위로 계산해야 합니다.
        surf2Dwrite(color, surface, x * sizeof(float4), y);
    }
}

void launchComputeCascade0(float4* cascadeBuffer, CascadeLevel* metaData, int gridW, int gridH) 
{
    dim3 blockSize(16, 16);
    dim3 gridSize((gridW + blockSize.x - 1) / blockSize.x,
                  (gridH + blockSize.y - 1) / blockSize.y);
                  
    computeCascade0Kernel<<<gridSize, blockSize>>>(cascadeBuffer, metaData);
    cudaDeviceSynchronize();
}

void launchVisualizeCascade0(cudaSurfaceObject_t surface, float4* cascadeBuffer, CascadeLevel* metaData, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
                  
    visualizeCascade0Kernel<<<gridSize, blockSize>>>(surface, cascadeBuffer, metaData, width, height);
    cudaDeviceSynchronize();
}


void launchComputeCascade(float4* buffer, CascadeLevel* meta, int levelIdx, int w, int h) {
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    computeCascadeKernel<<<gridSize, blockSize>>>(buffer, meta, levelIdx);
}

void launchMergeCascade(float4* buffer, CascadeLevel* meta, int lowerLevelIdx, int w, int h) {
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    mergeCascadeKernel<<<gridSize, blockSize>>>(buffer, meta, lowerLevelIdx);
}

int main()
{
    // 1. GLFW 초기화
    if (!glfwInit())
        return -1;
    GLFWwindow *window = glfwCreateWindow(800, 600, "Radiance Cascades 2D", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 2. GLAD 초기화 (이게 없으면 glGenTextures에서 터집니다)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "GLAD 초기화 실패" << std::endl;
        return -1;
    }

    int width = 800, height = 600;

    // 3. 텍스처 생성 (기존 코드와 동일)
    GLuint glTexture;
    glGenTextures(1, &glTexture);
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // 4. CUDA 등록
    cudaGraphicsResource_t cudaResource;
    // cudaGraphicsRegisterFlagsWriteDiscard 대신 기본 플래그로 테스트해보세요
    cudaGraphicsGLRegisterImage(&cudaResource, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    GLuint readFbo;
    glGenFramebuffers(1, &readFbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, glTexture, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    int numCascades = 6;
    std::vector<CascadeLevel> cascades = buildCascadeHierarchy(width, height, numCascades);
    CascadeLevel& lastLevel = cascades.back();
    
    int totalProbesData = lastLevel.dataOffset + (lastLevel.gridWidth * lastLevel.gridHeight * lastLevel.numDirections);

    float4* d_cascadeBuffer = nullptr;
    cudaMalloc(&d_cascadeBuffer, totalProbesData * sizeof(float4));
    cudaMemset(d_cascadeBuffer, 0, totalProbesData * sizeof(float4));

    // Cascade 메타데이터도 GPU로 복사 (커널에서 간격, 오프셋 등을 읽기 위함)
    CascadeLevel* d_cascadesMeta = nullptr;
    cudaMalloc(&d_cascadesMeta, cascades.size() * sizeof(CascadeLevel));
    cudaMemcpy(d_cascadesMeta, cascades.data(), cascades.size() * sizeof(CascadeLevel), cudaMemcpyHostToDevice);

    // 5. 렌더링 루프 내부
    while (!glfwWindowShouldClose(window))
    {
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaArray_t cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        cudaSurfaceObject_t surface;
        cudaCreateSurfaceObject(&surface, &resDesc);

        // ----------------------------------------------------
        // [1] 모든 Cascade 레벨에 대해 독립적으로 Raymarching 실행
        // ----------------------------------------------------
        for (int i = 0; i < numCascades; ++i) {
            launchComputeCascade(d_cascadeBuffer, d_cascadesMeta, i, cascades[i].gridWidth, cascades[i].gridHeight);
        }

        cudaDeviceSynchronize(); // 동기화 보장

        // ----------------------------------------------------
        // [2] Top-Down 방식으로 Cascade 병합 (Merge)
        // 최상위 레벨에서 시작하여 레벨 0까지 내려옵니다.
        // ----------------------------------------------------
        for (int i = numCascades - 2; i >= 0; --i) {
            launchMergeCascade(d_cascadeBuffer, d_cascadesMeta, i, cascades[i].gridWidth, cascades[i].gridHeight);
            cudaDeviceSynchronize();
        }

        // ----------------------------------------------------
        // [3] 최종적으로 모든 빛이 모인 Cascade 0을 화면 텍스처에 시각화
        // ----------------------------------------------------
        launchVisualizeCascade0(surface, d_cascadeBuffer, d_cascadesMeta, width, height);

        cudaDestroySurfaceObject(surface);
        cudaGraphicsUnmapResources(1, &cudaResource, 0);

        // 프레임버퍼 Blit (화면에 그리기)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); 
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 정리
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteTextures(1, &glTexture);
    glfwTerminate();
    return 0;
}