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

void launchCascadeKernel(cudaSurfaceObject_t surface, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cascadeTestKernel<<<gridSize, blockSize>>>(surface, width, height);

    // 에러 체크 및 동기화 (디버깅에 유용합니다)
    cudaDeviceSynchronize();
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

    int width = 1080, height = 720;

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

        // CUDA 커널 실행 (주석 해제!)
        launchCascadeKernel(surface, width, height);

        cudaDestroySurfaceObject(surface);
        cudaGraphicsUnmapResources(1, &cudaResource, 0);

        // 6. 결과 확인을 위한 Blit (화면에 그리기)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // 0은 기본 윈도우 프레임버퍼
        
        // FBO에 연결된 텍스처를 화면 크기에 맞춰 복사
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 정리
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteTextures(1, &glTexture);
    glfwTerminate();
    return 0;
}