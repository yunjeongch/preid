# preid

online으로 pose 뽑아서 inferability 계산하므로 mmpose 패키지 설치해서 사용해야 함. [패키지 설치 관련](https://mmpose.readthedocs.io/en/latest/installation.html)

### 파일 구조
```
data
 ┣ cuhk03
 ┃ ┣ cuhk03_release
 ┃ ┃ ┣ README.md
 ┃ ┃ ┗ cuhk-03.mat
 ┃ ┣ images_detected
 ┃ ┃ ┣ 1_001_1_01.png
 ┃ ┃ ┗ ...
 ┃ ┣ images_labeled
 ┃ ┃ ┣ 1_001_1_01.png
 ┃ ┃ ┗ ...
 ┃ ┣ cuhk03_new_protocol_config_detected.mat
 ┃ ┣ cuhk03_new_protocol_config_labeled.mat
 ┃ ┣ splits_classic_detected.json
 ┃ ┣ splits_classic_labeled.json
 ┃ ┣ splits_new_detected.json
 ┃ ┗ splits_new_labeled.json
 ┣ market1501
 ┃ ┣ bounding_box_test
 ┃ ┃ ┣ -1_c1s1_000401_03.jpg
 ┃ ┃ ┗ ...
 ┃ ┣ bounding_box_train
 ┃ ┃ ┣ 0002_c1s1_000451_03.jpg
 ┃ ┃ ┗ ...
 ┃ ┣ gt_bbox
 ┃ ┃ ┣ 0001_c1s1_001051_00.jpg
 ┃ ┃ ┗ ...
 ┃ ┣ gt_query
 ┃ ┃ ┣ 0001_c1s1_001051_00_good.mat
 ┃ ┃ ┗ ...
 ┃ ┣ query
 ┃ ┃ ┣ 0001_c1s1_001051_00.jpg
 ┃ ┃ ┗ ...
 ┃ ┣ posture.txt
 ┃ ┗ readme.txt
 ┣ MSMT17_V2
 ┃ ┣ mask_test_v2
 ┃ ┃ ┣ 0000
 ┃ ┃ ┃ ┣ 0000_000_01_0303morning_0015_0.jpg
 ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┗ ...
 ┃ ┣ mask_train_v2
 ┃ ┃ ┣ 0000
 ┃ ┃ ┃ ┣ 0000_000_01_0303morning_0008_0.jpg
 ┃ ┃ ┃ ┗ ...
 ┃ ┃ ┗ ...
 ┃ ┣ list_gallery.txt
 ┃ ┣ list_query.txt
 ┃ ┣ list_train.txt
 ┃ ┣ list_val.txt
 ┗ ┗ posture.txt
PAT
 ┣ PAT
 ┃ ┣ ckpt
 ┃ ┃ ┗ jx_vit_base_p16_224-80ecf9dd.pth
 ┃ ┣ config
 ┃ ┃ ┣ PAT.yml
 ┃ ┃ ┣ PAT_org.yml
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ defaults.py
 ┃ ┣ data
 ┃ ┃ ┣ datasets
 ┃ ┃ ┃ ┣ AirportALERT.py
 ┃ ┃ ┃ ┣ DG_cuhk02.py
 ┃ ┃ ┃ ┣ DG_cuhk03_detected.py
 ┃ ┃ ┃ ┣ DG_cuhk03_labeled.py
 ┃ ┃ ┃ ┣ DG_cuhk_sysu.py
 ┃ ┃ ┃ ┣ DG_dukemtmcreid.py
 ┃ ┃ ┃ ┣ DG_grid.py
 ┃ ┃ ┃ ┣ DG_iLIDS.py
 ┃ ┃ ┃ ┣ DG_market1501.py
 ┃ ┃ ┃ ┣ DG_prid.py
 ┃ ┃ ┃ ┣ DG_viper.py
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ bases.py
 ┃ ┃ ┃ ┣ caviara.py
 ┃ ┃ ┃ ┣ cuhk03.py
 ┃ ┃ ┃ ┣ dukemtmcreid.py
 ┃ ┃ ┃ ┣ grid.py
 ┃ ┃ ┃ ┣ iLIDS.py
 ┃ ┃ ┃ ┣ lpw.py
 ┃ ┃ ┃ ┣ market1501.py
 ┃ ┃ ┃ ┣ msmt17.py
 ┃ ┃ ┃ ┣ pes3d.py
 ┃ ┃ ┃ ┣ pku.py
 ┃ ┃ ┃ ┣ prai.py
 ┃ ┃ ┃ ┣ prid.py
 ┃ ┃ ┃ ┣ randperson.py
 ┃ ┃ ┃ ┣ sensereid.py
 ┃ ┃ ┃ ┣ shinpuhkan.py
 ┃ ┃ ┃ ┣ sysu_mm.py
 ┃ ┃ ┃ ┣ thermalworld.py
 ┃ ┃ ┃ ┣ vehicleid.py
 ┃ ┃ ┃ ┣ veri.py
 ┃ ┃ ┃ ┣ veri_keypoint.py
 ┃ ┃ ┃ ┣ veriwild.py
 ┃ ┃ ┃ ┗ viper.py
 ┃ ┃ ┣ samplers
 ┃ ┃ ┃ ┣ __pycache__
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ data_sampler.py
 ┃ ┃ ┃ ┗ triplet_sampler.py
 ┃ ┃ ┣ transform
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ autoaugment.py
 ┃ ┃ ┃ ┣ build.py
 ┃ ┃ ┃ ┣ functional.py
 ┃ ┃ ┃ ┗ transforms.py
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ build_DG_dataloader.py
 ┃ ┃ ┣ common.py
 ┃ ┃ ┗ data_utils.py
 ┃ ┣ log
 ┃ ┣ loss
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ arcface.py
 ┃ ┃ ┣ build_loss.py
 ┃ ┃ ┣ ce_labelSmooth.py
 ┃ ┃ ┣ center_loss.py
 ┃ ┃ ┣ inferability.py
 ┃ ┃ ┣ make_loss.py
 ┃ ┃ ┣ metric_learning.py
 ┃ ┃ ┣ myloss.py
 ┃ ┃ ┣ smooth.py
 ┃ ┃ ┣ softmax_loss.py
 ┃ ┃ ┗ triplet_loss.py
 ┃ ┣ model
 ┃ ┃ ┣ backbones
 ┃ ┃ ┃ ┣ IBN.py
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ resnet.py
 ┃ ┃ ┃ ┣ resnet_ibn.py
 ┃ ┃ ┃ ┗ vit_pytorch.py
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ make_model.py
 ┃ ┣ processor
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ ori_vit_processor_with_amp.py
 ┃ ┃ ┗ part_attention_vit_processor.py
 ┃ ┣ solver
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ cosine_lr.py
 ┃ ┃ ┣ lr_scheduler.py
 ┃ ┃ ┣ make_optimizer.py
 ┃ ┃ ┣ scheduler.py
 ┃ ┃ ┗ scheduler_factory.py
 ┃ ┣ tb_log
 ┃ ┣ utils
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ comm.py
 ┃ ┃ ┣ file_io.py
 ┃ ┃ ┣ iotools.py
 ┃ ┃ ┣ logger.py
 ┃ ┃ ┣ meter.py
 ┃ ┃ ┣ metrics.py
 ┃ ┃ ┣ registry.py
 ┃ ┃ ┗ reranking.py
 ┃ ┣ .gitignore
 ┃ ┣ 3.jsonl
 ┃ ┣ README.md
 ┃ ┣ run.sh
 ┃ ┣ test.py
 ┃ ┣ test.sh
 ┃ ┣ train.py
 ┃ ┣ visualize.py
 ┃ ┗ visualize.sh
 ┗ README.md
```

### Config 

MODEL:

  MMPOSE_CONFIG: '../../mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'  -> mmpose extractor config file [download](https://mmpose.readthedocs.io/en/latest/installation.html) Verify installation step 1 참고
  
  MMPOSE_CKPT: '../../mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'


INFERABILITY:

  TRIPLET: True  -> inferability를 적용할 것인지 여부
  
  ALPHA: 0.5  -> continuous ver. inferabiltiy (중간 보고서 버전)의 hyperparam
  
  POS: False  -> positive에만 적용할 것인지 (True) 둘다 적용할 것인지 (false)
  
  DISCRETE: False  -> discrete ver. inferability인지 여부. discrete ver. : 앞 1 / 옆 0 / 뒤 -1
