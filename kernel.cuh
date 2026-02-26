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

    // 1. 프로브 그리드 상의 연속적인(실수) 좌표 계산
    // 픽셀의 중앙 위치를 기준으로 보간하기 위해 0.5f를 더해 조정합니다.
    float probeX = ((float)px + 0.5f) / level.probeSpacing - 0.5f;
    float probeY = ((float)py + 0.5f) / level.probeSpacing - 0.5f;

    // 2. 인접한 4개의 프로브 인덱스 계산 (화면 경계를 넘어가지 않도록 클램핑)
    int px0 = max(0, min((int)floorf(probeX), level.gridWidth - 1));
    int py0 = max(0, min((int)floorf(probeY), level.gridHeight - 1));
    int px1 = min(px0 + 1, level.gridWidth - 1);
    int py1 = min(py0 + 1, level.gridHeight - 1);

    // 보간 가중치 (소수점 부분)
    float tx = probeX - floorf(probeX);
    float ty = probeY - floorf(probeY);

    // 특정 프로브의 Irradiance(모든 방향의 빛의 평균)를 구하는 헬퍼 람다
    auto getIrradiance = [&](int x, int y) {
        int baseIdx = level.dataOffset + (y * level.gridWidth + x) * level.numDirections;
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int d = 0; d < level.numDirections; ++d) {
            float4 rad = cascadeBuffer[baseIdx + d];
            sum.x += rad.x; sum.y += rad.y; sum.z += rad.z;
        }
        // 방향의 개수로 나누어 평균을 냅니다.
        sum.x /= level.numDirections;
        sum.y /= level.numDirections;
        sum.z /= level.numDirections;
        return sum;
    };

    // 3. 4개의 인접 프로브에서 빛 정보 가져오기
    float4 c00 = getIrradiance(px0, py0);
    float4 c10 = getIrradiance(px1, py0);
    float4 c01 = getIrradiance(px0, py1);
    float4 c11 = getIrradiance(px1, py1);

    // 4. 이중 선형 보간 (Bilinear Interpolation)
    // X축 보간
    float3 cx0 = make_float3(
        c00.x * (1 - tx) + c10.x * tx,
        c00.y * (1 - tx) + c10.y * tx,
        c00.z * (1 - tx) + c10.z * tx
    );
    float3 cx1 = make_float3(
        c01.x * (1 - tx) + c11.x * tx,
        c01.y * (1 - tx) + c11.y * tx,
        c01.z * (1 - tx) + c11.z * tx
    );

    // Y축 보간 (최종 색상)
    float4 finalColor = make_float4(
        cx0.x * (1 - ty) + cx1.x * ty,
        cx0.y * (1 - ty) + cx1.y * ty,
        cx0.z * (1 - ty) + cx1.z * ty,
        1.0f
    );

    // 톤매핑 (기존과 동일)
    finalColor.x = fminf(finalColor.x * 1.5f, 1.0f);
    finalColor.y = fminf(finalColor.y * 1.5f, 1.0f);
    finalColor.z = fminf(finalColor.z * 1.5f, 1.0f);

    surf2Dwrite(finalColor, surface, px * sizeof(float4), py);
}

#if 0
__global__ void computeCascadeKernel(float4* cascadeBuffer, CascadeLevel* metaData, int levelIdx) {
    CascadeLevel level = metaData[levelIdx];

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= level.gridWidth || py >= level.gridHeight) return;

    float2 probePos = make_float2(px * level.probeSpacing, py * level.probeSpacing);
    int probeBaseIndex = level.dataOffset + (py * level.gridWidth + px) * level.numDirections;

    for (int d = 0; d < level.numDirections; ++d) {
        float angle = (2.0f * 3.14159265f * d) / level.numDirections;
        float2 dir = make_float2(cosf(angle), sinf(angle));

        // w 채널은 Transparency(Beta) 값입니다. 기본값 1.0f (투명)
        float4 accumulatedRadiance = make_float4(0, 0, 0, 1.0f);
        float t = level.rayRangeMin;
        float stepSize = 1.0f; 

        while (t < level.rayRangeMax) {
            float2 samplePos = make_float2(probePos.x + dir.x * t, probePos.y + dir.y * t);
            float4 emission;
            float dist = sceneSDF(samplePos, emission);

            if (dist <= 0.0f) {
                // 부딪히면 빛 색상을 저장하고, 투명도를 0.0f(불투명)로 설정
                accumulatedRadiance = make_float4(emission.x, emission.y, emission.z, 0.0f);
                break;
            }
            t += stepSize;
        }

        cascadeBuffer[probeBaseIndex + d] = accumulatedRadiance;
    }
}
#else
__global__ void computeCascadeKernel(float4* cascadeBuffer, CascadeLevel* metaData, int levelIdx) {
    CascadeLevel level = metaData[levelIdx];

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= level.gridWidth || py >= level.gridHeight) return;

    // 수정됨: 프로브의 중심점(0.5f)을 기준으로 정확한 좌표 계산
    float2 probePos = make_float2((px + 0.5f) * level.probeSpacing, (py + 0.5f) * level.probeSpacing);
    int probeBaseIndex = level.dataOffset + (py * level.gridWidth + px) * level.numDirections;

    // 현재 Cascade의 한 방향이 차지하는 각도 (원뿔의 너비)
    float coneAngle = (2.0f * 3.14159265f) / level.numDirections;

    for (int d = 0; d < level.numDirections; ++d) {
        float angle = coneAngle * d;
        float2 dir = make_float2(cosf(angle), sinf(angle));

        float4 accumulatedRadiance = make_float4(0, 0, 0, 1.0f);
        float t = level.rayRangeMin;

        // 투명도가 남아있고, 최대 거리에 도달하지 않은 동안 루프
        while (t < level.rayRangeMax && accumulatedRadiance.w > 0.01f) {
            float2 samplePos = make_float2(probePos.x + dir.x * t, probePos.y + dir.y * t);
            float4 emission;
            float dist = sceneSDF(samplePos, emission);

            // 수정됨: 현재 거리 t에서 원뿔(Cone)의 반지름 계산
            float coneRadius = fmaxf(t * coneAngle * 0.5f, 1.0f); // 최소 1픽셀 두께 보장

            // 광선이 물체와 교차했는지 확인 (부드러운 충돌 - Soft Intersection)
            if (dist < coneRadius) {
                // 원뿔 내에서 물체가 차지하는 비율 (Coverage: 0.0 ~ 1.0)
                float coverage = (coneRadius - dist) / (2.0f * coneRadius);
                coverage = fminf(fmaxf(coverage, 0.0f), 1.0f);

                // 빛 누적 (Emission * 투명도 * Coverage)
                accumulatedRadiance.x += emission.x * accumulatedRadiance.w * coverage;
                accumulatedRadiance.y += emission.y * accumulatedRadiance.w * coverage;
                accumulatedRadiance.z += emission.z * accumulatedRadiance.w * coverage;

                // 차단된 만큼 투명도(Beta) 감소
                accumulatedRadiance.w *= (1.0f - coverage);
            }

            // 구체 추적(Sphere Tracing) 최적화: 물체에서 먼 만큼 안전하게 점프!
            t += fmaxf(dist - coneRadius, 1.0f); 
        }

        cascadeBuffer[probeBaseIndex + d] = accumulatedRadiance;
    }
}
#endif

__global__ void mergeCascadeKernel(float4* cascadeBuffer, CascadeLevel* metaData, int lowerLevelIdx) {
#if 0
    CascadeLevel lower = metaData[lowerLevelIdx];
    CascadeLevel upper = metaData[lowerLevelIdx + 1];

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= lower.gridWidth || py >= lower.gridHeight) return;

    // 하위 레벨의 현재 프로브 화면 좌표
    float2 probePos = make_float2(px * lower.probeSpacing, py * lower.probeSpacing);
    int lowerBaseIndex = lower.dataOffset + (py * lower.gridWidth + px) * lower.numDirections;

    // 상위 레벨 그리드에서의 실수 좌표 (보간용)
    float upperX = probePos.x / upper.probeSpacing;
    float upperY = probePos.y / upper.probeSpacing;

    // 상위 레벨의 4개 인접 프로브 인덱스 계산 (Bilinear Interpolation)
    int ux0 = max(0, min((int)floorf(upperX), upper.gridWidth - 1));
    int uy0 = max(0, min((int)floorf(upperY), upper.gridHeight - 1));
    int ux1 = min(ux0 + 1, upper.gridWidth - 1);
    int uy1 = min(uy0 + 1, upper.gridHeight - 1);

    float tx = upperX - floorf(upperX);
    float ty = upperY - floorf(upperY);
#else
    CascadeLevel lower = metaData[lowerLevelIdx];
    CascadeLevel upper = metaData[lowerLevelIdx + 1];

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= lower.gridWidth || py >= lower.gridHeight) return;

    // 1. 하위 프로브의 정확한 화면 좌표 (중심점 기준)
    float2 probePos = make_float2((px + 0.5f) * lower.probeSpacing, (py + 0.5f) * lower.probeSpacing);
    int lowerBaseIndex = lower.dataOffset + (py * lower.gridWidth + px) * lower.numDirections;

    // 2. 상위 격자에서의 실수 좌표 계산 (상위 프로브 중심점 기준 보정)
    float upperX = (probePos.x / upper.probeSpacing) - 0.5f;
    float upperY = (probePos.y / upper.probeSpacing) - 0.5f;

    // 3. 인접한 상위 4개 프로브의 인덱스 계산 (화면 밖으로 나가지 않게 클램핑)
    int ux0 = max(0, min((int)floorf(upperX), upper.gridWidth - 1));
    int uy0 = max(0, min((int)floorf(upperY), upper.gridHeight - 1));
    int ux1 = min(ux0 + 1, upper.gridWidth - 1);
    int uy1 = min(uy0 + 1, upper.gridHeight - 1);

    float tx = upperX - floorf(upperX);
    float ty = upperY - floorf(upperY);
#endif

    for (int d = 0; d < lower.numDirections; ++d) {
        
        // 1. 각도 필터링을 포함하여 상위 레벨의 Radiance를 가져오는 헬퍼 람다
        auto getFilteredUpper = [&](int x, int y) {
            int baseIdx = upper.dataOffset + (y * upper.gridWidth + x) * upper.numDirections;
            
            // 하위 방향 d에 대응하는 상위 방향은 중심이 2d 입니다.
            // 주변 각도를 0.25 : 0.5 : 0.25 비율로 섞어 부드럽게 가져옵니다.
            int u_prev = (d * 2 - 1 + upper.numDirections) % upper.numDirections;
            int u_mid  = (d * 2) % upper.numDirections;
            int u_next = (d * 2 + 1) % upper.numDirections;

            float4 c_prev = cascadeBuffer[baseIdx + u_prev];
            float4 c_mid  = cascadeBuffer[baseIdx + u_mid];
            float4 c_next = cascadeBuffer[baseIdx + u_next];

            return make_float4(
                c_prev.x * 0.25f + c_mid.x * 0.5f + c_next.x * 0.25f,
                c_prev.y * 0.25f + c_mid.y * 0.5f + c_next.y * 0.25f,
                c_prev.z * 0.25f + c_mid.z * 0.5f + c_next.z * 0.25f,
                c_prev.w * 0.25f + c_mid.w * 0.5f + c_next.w * 0.25f
            );
        };

        // 각 모서리에서 필터링된 값을 가져옵니다.
        float4 c00 = getFilteredUpper(ux0, uy0);
        float4 c10 = getFilteredUpper(ux1, uy0);
        float4 c01 = getFilteredUpper(ux0, uy1);
        float4 c11 = getFilteredUpper(ux1, uy1);

        // 2. 공간적 보간 (Bilinear - 기존과 같지만 w채널(beta)도 함께 보간)
        float4 cx0 = make_float4(
            c00.x * (1 - tx) + c10.x * tx, c00.y * (1 - tx) + c10.y * tx, c00.z * (1 - tx) + c10.z * tx, c00.w * (1 - tx) + c10.w * tx);
        float4 cx1 = make_float4(
            c01.x * (1 - tx) + c11.x * tx, c01.y * (1 - tx) + c11.y * tx, c01.z * (1 - tx) + c11.z * tx, c01.w * (1 - tx) + c11.w * tx);
        
        float4 upperRadiance = make_float4(
            cx0.x * (1 - ty) + cx1.x * ty, cx0.y * (1 - ty) + cx1.y * ty, cx0.z * (1 - ty) + cx1.z * ty, cx0.w * (1 - ty) + cx1.w * ty);

        // 3. 병합 공식 적용: L_lower = L_lower + Beta_lower * L_upper
        float4& lowerRadiance = cascadeBuffer[lowerBaseIndex + d];
        
        lowerRadiance.x += lowerRadiance.w * upperRadiance.x;
        lowerRadiance.y += lowerRadiance.w * upperRadiance.y;
        lowerRadiance.z += lowerRadiance.w * upperRadiance.z;
        lowerRadiance.w *= upperRadiance.w; // 새로운 투명도 계산 (Beta_ac = Beta_ab * Beta_bc)
    }
}

#endif
