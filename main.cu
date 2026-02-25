#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <iostream>

int main() {
    // 1. GLFW 초기화
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800, 600, "Radiance Cascades 2D", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    // 2. GLAD 초기화 (이게 없으면 glGenTextures에서 터집니다)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
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

    // 5. 렌더링 루프 내부
    while (!glfwWindowShouldClose(window)) {
        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaArray_t cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        cudaSurfaceObject_t surface;
        cudaCreateSurfaceObject(&surface, &resDesc);

        // launchCascadeKernel(surface, width, height);

        cudaDestroySurfaceObject(surface);
        cudaGraphicsUnmapResources(1, &cudaResource, 0);

        // 6. 결과 확인을 위한 간단한 Blit (화면에 그리기)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        // 생성한 텍스처를 화면 프레임버퍼로 바로 복사합니다.
        // 이를 위해 임시 FBO가 필요할 수 있으나, 가장 간단한 확인법은 Full-screen Quad입니다.
        // 테스트를 위해 일단 glClear만 하고 버퍼만 바꿉니다.
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // 정리
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteTextures(1, &glTexture);
    glfwTerminate();
    return 0;
}