---
layout: post
title: "Making Every Event Count: Balancing Data Efficiency and Accuracy in Event Camera Subsampling"
date: 2025-06-09 21:00:00 +0900
categories: [Paper Review]
tags:
  [
    Paper review, Event camera, Event subsampling, Video classification
  ]
math: true
image:
  path: /assets/img/2025-06-09-EventSubsample/Untitled-0.png
#   alt: 대표 이미지 캡션---
---
> **CVPRW, 2025**<br/>
> [**[Paper](https://arxiv.org/abs/2505.21187)**]

## Contributions
- CNN을 활용해 다양한 benchmark dataset에서 6가지 **hardware-friendly subsampling** 방법을 평가
- **Causal density-based subsampling method** 제안 → 고밀도 event가 더 많은 정보를 담고 있음을 평가
- Subsampling method의 성능 저하에 기여하는 요인 심층 분석 → hyperparameter에 대한 민감도가 높음을 보임

> Event representation : EST → voxel grid format

## Related work
### Event rate reduction
- Downsampling : temporal 또는 spatial 해상도를 줄여 event 수 감소
  - SNN 활용, random event subsampling augmentation, event 개수 정규화 등
- Filtering : 특정 threshold에 따라 event를 걸러냄
  - Event 밀도를 활용해 Background activity(BA, event camera의 random noise) 제거

### Event processing (representation 방식)
- Event data를 deep learning 모델(특히 CNN)에 넣기 위해서는 데이터 전처리가 필요
  - [Timesurface](https://ieeexplore.ieee.org/abstract/document/7508476) : 각 픽셀에 가장 최근의 event timestamp 값 할당 → 시간 정보 유지
  - Counting events : 픽셀 마다 event 개수 합치기. fine-grained temporal detail은 놓칠 수 있음
  - [Voxel grid](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.html) : t를 일정 구간으로 나누고 특정 연산을 적용
  - [Event Spike Tensor](https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html) (EST) : voxel grid를 확장한 방식. 전체 시간 t를 일정 구간으로 나누고 learnable parameters를 적용 

## Method

### Event representation and training procedure
- EST 활용(MLP), 각 polarity에 대해 2d grid frame으로 누적, $V\in\mathbb{R}^{2B\times H\times W}$가 CNN의 input이 됨
- MLP와 CNN은 jointly updated
- MLP : bins $B=9$
- CNN : ResNet34(ImageNet-1k_v1), input layer는 $2B=19$ channels

### Subsampling types
Subsampling의 전제조건 : Causal할 것. 즉, 미래의 입력에 영향받지 않을 것 (latency, power consumption 문제)

<span id="fig2"></span>
![fig2](/assets/img/2025-06-09-EventSubsample/Untitled-1.png){: width="60%" height="60%"}
- Spatial subsampling
  - <a href="#fig2">fig2(a)</a>처럼 특정 값 만큼 건너뛰면서 event를 보존
  $$e=(x,y,t,p)\ is\ kept\ if\ (x-r_{x,0})\ mod\ r_x=0\ and\ (y-r_{y,0})\ mod\ r_y=0$$

- Temporal subsampling
  - temporal window size $w_t$를 $r_t$만큼 나누어 $\Delta t$를 구함
  $$kw_t+\Delta t_0 \leq t < kw_t +\Delta t + \Delta t_0$$
  - <a href="#fig2">fig2(b)</a>에 보이는 오렌지색 구간만 남게 됨

- Random subsampling
  - 독립적인 event 사이에서 random하게 추출

- **Causal density-based subsampling**
  - 논문에서 제안하는 subsampling 방법, causal, memory-efficient, computational inexpensive
  1. $p_i$에 따른 density value $f_i^{(p_i)}$를 계산
  
    $$f_i^{(p_i)} = \sum_{j=1 | p_j = p_i}^{i} s(x_i - x_j, y_i - y_j) \exp\left(\frac{t_i - t_j}{\tau}\right)$$

     $s(\cdot,\cdot)$ : $w_d\times w_d$ 크기의 spatial kernel → spatial filtering <br/>
     $\tau$ : decay parameter → temporal filtering
  2. density value와 threshold $f^{(thresh)}$를 가지고 남길 event를 선택
     - $f^{(thresh)}$ 값이 크면 더 많은 event를 필터링
  
  <span id="fig3"></span>
  ![fig3](/assets/img/2025-06-09-EventSubsample/Untitled-2.png){: width="60%" height="60%"}
  - 밀집된 영역에서 너무 많은 event가 선택되는 것을 방지하기 위해 random thresholding 적용 (<a href="#fig3">fig3(b)</a>)
    
    $$f_i^{(p_i)} \geq u f^{\text{(thresh)}}$$

    $u$ : random coefficient, $u\sim\mathcal{U}(0,1)$

- [Event count subsampling](https://hal.science/hal-03814075/)
  ![eventcount](/assets/img/2025-06-09-EventSubsample/Untitled-3.png)
  1. Spatial한 event 이미지를 겹치지 않게 $(r_x\times r_y)$ 크기의 window로 나눔
  2. Window 내의 event polarity를 평균 내어 normalized event count로 정함
  3. Count 값이 임계값 $p^{(thresh)}_{EC}$을 넘으면 해당 window에 대응하는 event가 발생
  4. noramlized event count의 변화량으로 대응 event의 polarity 결정

- [Corner-based subsampling](https://ieeexplore.ieee.org/abstract/document/9652120)
  - Image corner를 활용해 subsampling하는 방식

  1. Input event를 기준으로 threshold-ordinal surface(TOS) 기반 event representation 생성 혹은 업데이트
  2. Input event 위치를 기준으로 $w_c\times w_c$ 패치 생성 후 Harris score ($h_c$) 계산
  3. $h_c > H_{(thres)c}$인 경우 corner로 유지

## Results
- Dataset : N-Caltech101, DVS-Gesture, N-Cars
![tab1](/assets/img/2025-06-09-EventSubsample/Untitled-7.png){: width="80%" height="80%"}

- 모든 subsampling에 대한 결과를 위해 metric $\text{nAUC}$ 정의

  $$AUC_{\text{acc-#events}} = \int \text{acc}(\text{#events}) d(\log_{10} \text{#events})$$

  $$\text{nAUC}_{\text{acc-#events}} = \frac{\text{AUC}_{\text{acc-#events}}}{\text{AUC}_{\text{acc-#events}}^{({oracle})}}$$

![tab2](/assets/img/2025-06-09-EventSubsample/Untitled-8.png){: width="80%" height="80%"}
- Subsampling 별 complexity<br/>

![fig4and5](/assets/img/2025-06-09-EventSubsample/Untitled-4.png)
- Density-based와 corner-based subsampling이 좋은 결과를 보였음
- Random subsampling이 temporal 혹은 spatial subsampling보다 좋은 성능을 보임
- 특히 Density-based subsampling은 sparse한 event 상황에서도 좋은 성능을 유지함

  ![fig6](/assets/img/2025-06-09-EventSubsample/Untitled-5.png){: width="60%" height="60%"}
- DVS-Gesture 에서 보다 큰 편차를 보임 → 위의 바이올린 플롯으로 spatial subsampling이 offset 값에 민감하게 반응함을 알 수 있음

![fig8and9](/assets/img/2025-06-09-EventSubsample/Untitled-6.png)
- N-Cars 데이터셋은 이벤트 수 자체가 다른 데이터셋보다 적어 정확도 낮고 편차 큼(불안정한 결과)
- 데이터셋 자체가 video마다 event 수 편차가 큼 → event가 적은 video에서는 denseityp-based subsampling 결과로 다수의 event가 제거될 수 있음
- 문제 해결을 위해 causal한 특성을 버리고, density value를 정규화 하는 방법 존재

Workshop paper라 그런가, 결과 분석 내용이 인상 깊었음
