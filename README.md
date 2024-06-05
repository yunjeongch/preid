# preid

online으로 pose 뽑아서 inferability 계산하므로 mmpose 패키지 설치해서 사용해야 함. [패키지 설치 관련]([url](https://mmpose.readthedocs.io/en/latest/installation.html))

### Config 

MODEL:
  MMPOSE_CONFIG: '../../mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'  -> mmpose extractor config file [download]([url](https://mmpose.readthedocs.io/en/latest/installation.html)) Verify installation step 1 참고
  MMPOSE_CKPT: '../../mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

INFERABILITY:
  TRIPLET: True  -> inferability를 적용할 것인지 여부
  ALPHA: 0.5  -> continuous ver. inferabiltiy (중간 보고서 버전)의 hyperparam
  POS: False  -> positive에만 적용할 것인지 (True) 둘다 적용할 것인지 (false)
  DISCRETE: False  -> discrete ver. inferability인지 여부. discrete ver. : 앞 1 / 옆 0 / 뒤 -1
