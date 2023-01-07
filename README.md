## 프로젝트 목적

이미지 인식을 위한 CNN(Convolution Neural Networks)을 진행하는 과정 중 **Convolution Layer에서 상당히 많은 연산 작업을 수행**합니다. 우리는 이를  **하나의 프로세서를 사용하여 연산 과정을 수행**하므로 Convolution Layer를 수행하기 위해 상당히 많은 시간 소비합니다.  우리는 이 시간 비용을 절감하기 위해 **GPU를 이용해 연산을 분산** 처리합니다. GPU를 이용하여 성능을 향상시키는 과정에서 **"싱글 코어 프로그램(CPU만 사용)을 멀티 코어 프로그램(CPU와 GPU를 이용)으로 변환하는 과정이 필요**합니다. 이 과정을 수행하기 위해 **분산 처리를 위한 동기화**, **메모리 사용에 대한 효율성 등 다양한 시각**에서 프로그래밍을 진행합니다. 우리는 이 과정을 통해 기존의 프로그래밍 방식에서 벗어나 새로운 차원의 프로그래밍을 진행할 수 있습니다. 이를 통해 프로그래밍에 대한 시각을 넓힐 수 있는 기회를 얻을 수 있고, 병렬 프로그래밍이 미래에서 중요한 기술인 이유를 직접 체감할 수 있다.

## 최적화 아이디어 발전 과정

### 1. 메모리 오버헤드 감소 및 버퍼 재사용

- 기존 Sequential Code를 사용하여 메모리 오버헤드를 감소를 위해 버퍼에 사용할 data를 미리 할당
- 각 Layer의 결과가 담긴 버퍼를 다음 Layer의 입력으로 사용하여 메모리 오버헤드 감소

### 2. Pooling Layer와 FC layer 병렬화

- 다수의 work-item에 할당하여 **동시다발적으로 연산**할 수 있도록 병렬화

### 3. Unrolling 기법 사용

- 연산을 진행하는 모든 **for문에 분기 횟수를 줄이기 위해** 적용

### 4. Work Group당 하나의 픽셀에 대한 입력 채널(inDim)에 대한 가중치 연산 진행 후 출력 채널(outDim)개의 행렬의 N(GROUP_ID)번째 원소에 결과 저장

- 상당한 성능 향상(수행 시간 23초 소요)을 이루었으나 이후에 Tiling이라는 더 좋은 기법을 찾아 변경

### 5. Tiling 기법 사용

- 컨볼루션 성능을 비약적으로 향상 시켜준 기법
- 최종적으로 수행 시간 5초 소요

## 최적화 기법 설명

### for All Layer

- Loop unrolling 기법 사용
    - 바이너리 코드의 크기는 증가하지만, 하드웨어 가속을 추구하는 기법으로 다음 루프로 이동하는 동안 일어나는 동기화, 인덱스 증가, 비교문과 같은 불필요한 계산 시간 비용 절감
        
        ![loop_unrolling](https://user-images.githubusercontent.com/86178336/211130865-57cd399d-1521-44db-827e-b086d9d2f564.png)

        

### Architecture

- Constant Variable 사용
    - 자주 사용 또는 의미가 존재하는 숫자들을 미리 선언하여 오류 방지 및 가독성 향상
        
        ![constant_variable](https://user-images.githubusercontent.com/86178336/211130889-ed5284f6-b397-4ce1-b90c-a519f61cc624.png)
        

- Layer 연산 결과에 대한 버퍼 재사용
    - 연산 결과를 쓸데없이 Read/Write 하지 않음으로써 시간 비용 절감
        
        ![buffer_recycle](https://user-images.githubusercontent.com/86178336/211130918-34d0403e-c274-4dd3-b0e5-3e54b4ae2616.png)
        
- Layer 연산에 사용할 data들을 미리 버퍼에 할당
    - 오프셋을 이용하여 사용함으로써 메모리 접근에 사용하는 시간 비용 절감
        
        ![allocation_buffer](https://user-images.githubusercontent.com/86178336/211130967-c70f0468-eb36-4edb-82ed-74ce164ff2d4.png)
          
        ![use_offset](https://user-images.githubusercontent.com/86178336/211130970-44546ea2-8691-4ac9-b7f4-32d33aadfe9e.png)
        

### Convolution

컨볼루션 레이어는 2단계로 나누어 수행

- **Step1 : Convolution_1  - 데이터 직렬화**
    - 컨볼루션 연산을 위한 데이터 구조화 진행
        
        ![data_serialization](https://user-images.githubusercontent.com/86178336/211131012-bde6c7da-dd3f-4249-a1e4-41c6df200f05.png)
        
        
- **Step2 : Convolution_2 - 타일링 기법을 사용하여 행렬 곱셈 연산**
    - 직렬화 된 데이터들을 로컬 메모리에 저장
        
        ![save_data](https://user-images.githubusercontent.com/86178336/211131032-a6327ee5-4209-4d7f-adea-5907f050e3bd.png)
        
        
    - 로컬 메모리를 이용하여 가중치 연산
        
        ![weight_calc](https://user-images.githubusercontent.com/86178336/211131034-de04ef12-e5f6-485c-a11a-250fb786da61.png)
        
        
    
- 기존 좌측 방법의 행렬 곱셈에서 우측의 행렬 곱셈으로 변환
    
    ![change_calc_method](https://user-images.githubusercontent.com/86178336/211131072-b046b9a9-7f62-4664-8ff3-f54936468b03.png)
    

### **Max Pooling**

- 병렬화 기법
    - 한 Work Group에서 N * N * 4개의 입력값들에 대한 N * N개의 결과값 생성
    - 아래 사진을 참고
        
        ![max_pooling](https://user-images.githubusercontent.com/86178336/211131094-8485b09e-5ee6-464a-800a-a4977052b447.png)
        

### Fully-connected

- 병렬화 기법
    - 하나의 Work Group에서 TS개의 결과값 생성
    - 아래 사진 참고
    
    ![fully_connected](https://user-images.githubusercontent.com/86178336/211131098-43139ab4-17c5-400f-9717-325691f017dd.png)
    

## 결과

결과적으로 기존 850 ~ 900초의 성능을 가지는 순차 코드를 병렬화와 기법 적용을 통해 [5.7초의 성능](https://github.com/ParkRootSeok/CNN/blob/master/console_result.png)을 획득
