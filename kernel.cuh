#ifndef KERNER_CUDA_HEADER
#define KERNER_CUDA_HEADER

#include "math.cuh"
#include "structs.cuh"

__device__ float sceneSDF(float2 p, float4 &outEmission)
{
    float2 center = make_float2(400.0f, 300.0f);
    float radius = 50.0f;
    float dist = length(make_float2(p.x - center.x, p.y - center.y)) - radius;

    // 구체 내부에 있으면 빛을 방출 (Cyan 색상)
    if (dist <= 0.0f)
    {
        outEmission = make_float4(0.0f, 1.0f, 1.0f, 1.0f); // R, G, B, Intensity
    }
    else
    {
        outEmission = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return dist;
}

// --- 2. Cascade 0 레이마칭 커널 ---
__global__ void computeCascade0Kernel(float4 *cascadeBuffer, CascadeLevel *metaData)
{
    CascadeLevel level = metaData[0]; // Cascade 0 정보 가져오기

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= level.gridWidth || py >= level.gridHeight)
        return;

    // 현재 프로브의 실제 화면 픽셀 좌표
    float2 probePos = make_float2(px * level.probeSpacing, py * level.probeSpacing);

    // 이 프로브의 데이터가 저장될 메모리 시작 위치
    int probeBaseIndex = level.dataOffset + (py * level.gridWidth + px) * level.numDirections;

    // 정해진 방향(numDirections, Cascade 0은 4방향)으로 Ray를 발사
    for (int d = 0; d < level.numDirections; ++d)
    {
        // 각도 계산 (0 ~ 2PI 분할)
        float angle = (2.0f * 3.14159265f * d) / level.numDirections;
        float2 dir = make_float2(cosf(angle), sinf(angle));

        float4 accumulatedRadiance = make_float4(0, 0, 0, 0);
        float t = level.rayRangeMin; // 탐색 시작 거리 (Cascade 0은 보통 0)
        float stepSize = 1.0f;       // 1픽셀 단위로 전진 (나중에는 SDF를 활용해 최적화 가능)

        // 정해진 최대 거리(rayRangeMax)까지만 빛을 탐색
        while (t < level.rayRangeMax)
        {
            float2 samplePos = make_float2(probePos.x + dir.x * t, probePos.y + dir.y * t);

            float4 emission;
            float dist = sceneSDF(samplePos, emission);

            // 빛을 발견하면 누적하고 루프 종료 (가장 먼저 부딪힌 빛만 계산)
            if (dist <= 0.0f)
            {
                accumulatedRadiance = emission;
                break;
            }
            t += stepSize;
        }

        // 결과 버퍼에 저장
        cascadeBuffer[probeBaseIndex + d] = accumulatedRadiance;
    }
}



__global__ void visualizeCascade0Kernel(cudaSurfaceObject_t surface, float4* cascadeBuffer, CascadeLevel* metaData, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    CascadeLevel level = metaData[0]; // Cascade 0 정보

    // 현재 픽셀이 속한 프로브의 그리드 인덱스 계산
    int gridX = px / level.probeSpacing;
    int gridY = py / level.probeSpacing;

    // 해당 프로브의 데이터 시작 위치
    int probeBaseIndex = level.dataOffset + (gridY * level.gridWidth + gridX) * level.numDirections;

    // 프로브가 수집한 4방향의 빛을 모두 더해서 픽셀 색상으로 만듭니다.
    float4 finalColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    for (int d = 0; d < level.numDirections; ++d) {
        float4 rad = cascadeBuffer[probeBaseIndex + d];
        finalColor.x += rad.x;
        finalColor.y += rad.y;
        finalColor.z += rad.z;
    }

    // 간단한 톤매핑 (값이 1.0을 넘지 않도록 제한)
    finalColor.x = fminf(finalColor.x, 1.0f);
    finalColor.y = fminf(finalColor.y, 1.0f);
    finalColor.z = fminf(finalColor.z, 1.0f);

    // surface에 결과 색상 쓰기 (x 좌표는 반드시 바이트 단위여야 함)
    surf2Dwrite(finalColor, surface, px * sizeof(float4), py);
}

#endif
