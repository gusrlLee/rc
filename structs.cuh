#ifndef STRUCTS_CUDA_HEADER
#define STRUCTS_CUDA_HEADER

#include "math.cuh"

#include <iostream>
#include <vector>
#include <algorithm>

struct CascadeLevel {
    int index;          // Cascade 인덱스 (0, 1, 2...)
    int probeSpacing;   // 프로브 간격 (픽셀 단위, 예: 2, 4, 8...)
    int numDirections;  // 방향 개수 (예: 4, 8, 16...)
    int gridWidth;      // 이 Cascade의 프로브 그리드 가로 크기
    int gridHeight;     // 이 Cascade의 프로브 그리드 세로 크기
    float rayRangeMin;  // 탐색 시작 거리
    float rayRangeMax;  // 탐색 종료 거리
    int dataOffset;     // 거대한 1D/2D 메모리 풀 내에서 이 Cascade 데이터가 시작하는 오프셋
};

inline std::vector<CascadeLevel> buildCascadeHierarchy(int screenW, int screenH, int numCascades) 
{
    std::vector<CascadeLevel> cascades;
    int currentOffset = 0;
    
    // 초기 설정
    int currentSpacing = 2; // Cascade 0은 2x2 픽셀마다 1개
    int currentDirs = 4;    // Cascade 0은 4방향
    float currentRange = 4.0f; // 초기 Ray 탐색 거리
    float startRange = 0.0f;

    for (int i = 0; i < numCascades; ++i) {
        CascadeLevel level;
        level.index = i;
        level.probeSpacing = currentSpacing;
        level.numDirections = currentDirs;
        level.gridWidth = (screenW + currentSpacing - 1) / currentSpacing;
        level.gridHeight = (screenH + currentSpacing - 1) / currentSpacing;
        level.rayRangeMin = startRange;
        level.rayRangeMax = startRange + currentRange;
        level.dataOffset = currentOffset;

        // 다음 Cascade를 위한 데이터 갱신
        int numProbes = level.gridWidth * level.gridHeight;
        currentOffset += numProbes * currentDirs; // float4로 저장한다고 가정할 때의 총 개수
        
        startRange = level.rayRangeMax;
        currentRange *= 2.0f; // 탐색 범위 2배
        currentSpacing *= 2;  // 간격 2배 (해상도 1/2)
        currentDirs *= 2;     // 방향 2배
        
        cascades.push_back(level);
    }
    return cascades;
}

#endif