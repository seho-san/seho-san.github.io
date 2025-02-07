# gpu 활성화
gpu를 사용해서 ML이 빠르게 돌아가기 때문에 반드시 gpu로 동작시켜야 한다.

cpu대비 약 10배 이상 빠름

##### is available이 true면 gpu가 활성화 된 것이다.
처음에는 cpu가 로드되지만, epoch가 시작되면 작업관리자에서 gpu에 부하가 걸리기 시작한다.


```python
import torch
print(torch.cuda.is_available())  # True가 출력되면 GPU 사용 가능
print(torch.cuda.get_device_name(0))  # GPU 이름 출력
print(f"get device name : {torch.cuda.get_device_name(0)}")
print(f"is available : {torch.cuda.is_available()}")
print(f"torch version : {torch.__version__}")
```

    True
    NVIDIA GeForce RTX 3060 Laptop GPU
    get device name : NVIDIA GeForce RTX 3060 Laptop GPU
    is available : True
    torch version : 2.5.1
    

# roboflow 데이터 테스트

### 디렉토리 마운팅
이 과정을 거쳐야 드라이브와 colab 환경을 연동할 수 있다


```python
import os
from IPython.display import Image, clear_output
import torch
import yaml

os.chdir(r"C:\Users\sunny\졸업작품\yolov5")
```

# 디렉토리 마운팅 후에 해당 코드 다시 실행해야 함
전체 코드 작성하는 과정에서 한 번만 실행하면 됨
현재 이미 동작했기 때문에 주석처리 해 놓았음

전체 동작 과정에서 최초 1회만 동작하면 됨
만약 전체를 다시 돌릴 일이 있으면 해당 코드는 주석처리 하고 돌릴 것


```python
#!git clone https://github.com/ultralytics/yolov5
# YOLOv5 리포지토리 clone
```

#### 위 코드 잘못 작성했을 때 yolov5가 하위폴더로 생성되었을 때 하위에 잘못 생긴 폴더 삭제하는 코드
평상시엔 동작할 필요 없어서 주석처리 해놨음


```python
# 삭제할 디렉토리 경로
#folder_path = '/content/yolov5/yolov5'

# 디렉토리 삭제
#!rm -rf {folder_path}
```

# 필요한 패키지 다운로드 및 임포트
필요시 주석은 따로 markdown 작성할 것


```python
%cd C:\Users\sunny\졸업작품\yolov5
```

    C:\Users\sunny\졸업작품\yolov5
    

##### dependencies 설치


```python
%pip install -qr requirements.txt
%pip install -q roboflow

import torch
import yaml
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Setup complete. Using torch 2.5.1 (NVIDIA GeForce RTX 3060 Laptop GPU)
    

# 사용할 데이터셋 경로 및 데이터셋의 yaml 파일 경로 지정
제대로 연동 안 됐으면 왼쪽 폴더 모양에서 드라이브 마운트 다시 진행
경로는 아래 작성한 코드를 ctrl+click 했을 때, 왼쪽에 해당하는 폴더가 뜨면 드라이브 마운트 정상적으로 완료된 것임


```python
# 데이터 디렉토리와 YAML 파일 경로 설정
data_dir = r"C:\Users\sunny\졸업작품\BLACKICE-segmentation-4"
data_yaml = os.path.join(data_dir, "data.yaml")
```

반드시 위의 코드도 함께 실행할 것


```python
import yaml

# YAML 파일 로드
with open(data_yaml, 'r') as f:
    film = yaml.safe_load(f)

# train 및 val 경로 수정
film['train'] = 'C:/Users/sunny/졸업작품/BLACKICE-segmentation-4/train/images'
film['val'] = 'C:/Users/sunny/졸업작품/BLACKICE-segmentation-4/test/images'

# 수정된 내용을 다시 YAML 파일에 저장
with open(data_yaml, 'w') as f:
    yaml.dump(film, f)

print('변경된 yaml 파일:')
print(film)

```

    변경된 yaml 파일:
    {'names': ['clear-road', 'iced-road'], 'nc': 2, 'roboflow': {'license': 'MIT', 'project': 'blackice-segmentation', 'url': 'https://universe.roboflow.com/swacademy5/blackice-segmentation/dataset/4', 'version': 4, 'workspace': 'swacademy5'}, 'test': '../test/images', 'train': 'C:/Users/sunny/졸업작품/BLACKICE-segmentation-4/train/images', 'val': 'C:/Users/sunny/졸업작품/BLACKICE-segmentation-4/test/images'}
    


```python
!python train.py --img 320 --batch 8 --epochs 30 --data {data_yaml} --weights yolov5s.pt --device 0
```

    [34m[1mtrain: [0mweights=yolov5s.pt, cfg=, data=C:\Users\sunny\\BLACKICE-segmentation-4\data.yaml, hyp=data\hyps\hyp.scratch-low.yaml, epochs=30, batch_size=8, imgsz=320, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data\hyps, resume_evolve=None, bucket=, cache=None, image_weights=False, device=0, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs\train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
    [34m[1mgithub: [0mup to date with https://github.com/ultralytics/yolov5 
    YOLOv5  v7.0-397-gde62f93c Python-3.12.7 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mComet: [0mrun 'pip install comet_ml' to automatically track and visualize YOLOv5  runs in Comet
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs\train', view at http://localhost:6006/
    Overriding model.yaml nc=80 with nc=2
    
                     from  n    params  module                                  arguments                     
      0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
      1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
      2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
      3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
      4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
      5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
      6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
      7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
      8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
      9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
     10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
     14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
     18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
     21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     24      [17, 20, 23]  1     18879  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    Model summary: 214 layers, 7025023 parameters, 7025023 gradients, 16.0 GFLOPs
    
    Transferred 343/349 items from yolov5s.pt
    C:\Users\sunny\졸업작품\yolov5\models\common.py:894: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with amp.autocast(autocast):
    C:\Users\sunny\졸업작품\yolov5\models\common.py:894: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with amp.autocast(autocast):
    [34m[1mAMP: [0mchecks passed 
    [34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
    
    [34m[1mtrain: [0mScanning C:\Users\sunny\졸업작품\BLACKICE-segmentation-4\train\labels.cache... 108 images, 0 backgrounds, 0 corrupt: 100%|##########| 108/108 [00:00<?, ?it/s]
    [34m[1mtrain: [0mScanning C:\Users\sunny\졸업작품\BLACKICE-segmentation-4\train\labels.cache... 108 images, 0 backgrounds, 0 corrupt: 100%|##########| 108/108 [00:00<?, ?it/s]
    
    [34m[1mval: [0mScanning C:\Users\sunny\졸업작품\BLACKICE-segmentation-4\test\labels.cache... 35 images, 0 backgrounds, 0 corrupt: 100%|##########| 35/35 [00:00<?, ?it/s]
    [34m[1mval: [0mScanning C:\Users\sunny\졸업작품\BLACKICE-segmentation-4\test\labels.cache... 35 images, 0 backgrounds, 0 corrupt: 100%|##########| 35/35 [00:00<?, ?it/s]
    
    [34m[1mAutoAnchor: [0m4.30 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset 
    Plotting labels to runs\train\exp2\labels.jpg... 
    C:\Users\sunny\졸업작품\yolov5\train.py:355: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      scaler = torch.cuda.amp.GradScaler(enabled=amp)
    Image sizes 320 train, 320 val
    Using 8 dataloader workers
    Logging results to [1mruns\train\exp2[0m
    Starting training for 30 epochs...
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.482G     0.1231    0.01755    0.03128         28        320:   0%|          | 0/14 [00:00<?, ?it/s]
           0/29     0.482G     0.1231    0.01755    0.03128         28        320:   7%|7         | 1/14 [00:00<00:12,  1.02it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1229    0.01468    0.03015         16        320:   7%|7         | 1/14 [00:01<00:12,  1.02it/s]
           0/29     0.497G     0.1229    0.01468    0.03015         16        320:  14%|#4        | 2/14 [00:01<00:06,  1.92it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G      0.122    0.01425    0.02998         17        320:  14%|#4        | 2/14 [00:01<00:06,  1.92it/s]
           0/29     0.497G      0.122    0.01425    0.02998         17        320:  21%|##1       | 3/14 [00:01<00:03,  2.82it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1222    0.01449    0.03023         22        320:  21%|##1       | 3/14 [00:01<00:03,  2.82it/s]
           0/29     0.497G     0.1222    0.01449    0.03023         22        320:  29%|##8       | 4/14 [00:01<00:02,  3.56it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1223    0.01454    0.03042         21        320:  29%|##8       | 4/14 [00:01<00:02,  3.56it/s]
           0/29     0.497G     0.1223    0.01454    0.03042         21        320:  36%|###5      | 5/14 [00:01<00:02,  4.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1225    0.01495    0.03077         26        320:  36%|###5      | 5/14 [00:01<00:02,  4.38it/s]
           0/29     0.497G     0.1225    0.01495    0.03077         26        320:  43%|####2     | 6/14 [00:01<00:01,  5.06it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1216    0.01497     0.0307         19        320:  43%|####2     | 6/14 [00:01<00:01,  5.06it/s]
           0/29     0.497G     0.1216    0.01497     0.0307         19        320:  50%|#####     | 7/14 [00:01<00:01,  5.58it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1214     0.0146    0.03069         15        320:  50%|#####     | 7/14 [00:02<00:01,  5.58it/s]
           0/29     0.497G     0.1214     0.0146    0.03069         15        320:  57%|#####7    | 8/14 [00:02<00:00,  6.05it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.497G     0.1212    0.01478    0.03083         23        320:  57%|#####7    | 8/14 [00:02<00:00,  6.05it/s]
           0/29     0.497G     0.1212    0.01478    0.03083         23        320:  64%|######4   | 9/14 [00:02<00:00,  6.85it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.512G     0.1211    0.01481    0.03068         25        320:  64%|######4   | 9/14 [00:02<00:00,  6.85it/s]
           0/29     0.512G     0.1211    0.01481    0.03068         25        320:  71%|#######1  | 10/14 [00:02<00:00,  6.82it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.512G     0.1209    0.01479    0.03061         20        320:  71%|#######1  | 10/14 [00:02<00:00,  6.82it/s]
           0/29     0.512G     0.1209    0.01479    0.03061         20        320:  79%|#######8  | 11/14 [00:02<00:00,  7.51it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.512G     0.1207     0.0148    0.03057         22        320:  79%|#######8  | 11/14 [00:02<00:00,  7.51it/s]
           0/29     0.512G     0.1207     0.0148    0.03057         22        320:  86%|########5 | 12/14 [00:02<00:00,  7.46it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.512G     0.1202    0.01536    0.03055         29        320:  86%|########5 | 12/14 [00:02<00:00,  7.46it/s]
           0/29     0.512G     0.1202    0.01536    0.03055         29        320:  93%|#########2| 13/14 [00:02<00:00,  8.03it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           0/29     0.516G     0.1199    0.01564    0.03056         13        320:  93%|#########2| 13/14 [00:02<00:00,  8.03it/s]
           0/29     0.516G     0.1199    0.01564    0.03056         13        320: 100%|##########| 14/14 [00:02<00:00,  5.87it/s]
           0/29     0.516G     0.1199    0.01564    0.03056         13        320: 100%|##########| 14/14 [00:02<00:00,  4.79it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.04it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.35it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  4.98it/s]
                       all         35         38    0.00296      0.816     0.0086     0.0028
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1137     0.0149     0.0308         19        320:   0%|          | 0/14 [00:00<?, ?it/s]
           1/29     0.631G     0.1137     0.0149     0.0308         19        320:   7%|7         | 1/14 [00:00<00:01,  9.74it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1141    0.01691    0.02974         26        320:   7%|7         | 1/14 [00:00<00:01,  9.74it/s]
           1/29     0.631G     0.1141    0.01691    0.02974         26        320:  14%|#4        | 2/14 [00:00<00:01,  8.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1144    0.01697    0.02942         23        320:  14%|#4        | 2/14 [00:00<00:01,  8.11it/s]
           1/29     0.631G     0.1144    0.01697    0.02942         23        320:  21%|##1       | 3/14 [00:00<00:01,  8.76it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1131    0.01746    0.02947         22        320:  21%|##1       | 3/14 [00:00<00:01,  8.76it/s]
           1/29     0.631G     0.1131    0.01746    0.02947         22        320:  29%|##8       | 4/14 [00:00<00:01,  8.05it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1126    0.01768    0.02939         24        320:  29%|##8       | 4/14 [00:00<00:01,  8.05it/s]
           1/29     0.631G     0.1126    0.01768    0.02939         24        320:  36%|###5      | 5/14 [00:00<00:01,  8.61it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1123    0.01811    0.02943         24        320:  36%|###5      | 5/14 [00:00<00:01,  8.61it/s]
           1/29     0.631G     0.1123    0.01811    0.02943         24        320:  43%|####2     | 6/14 [00:00<00:00,  8.04it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1117    0.01836     0.0294         23        320:  43%|####2     | 6/14 [00:00<00:00,  8.04it/s]
           1/29     0.631G     0.1117    0.01836     0.0294         23        320:  50%|#####     | 7/14 [00:00<00:00,  8.54it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1111    0.01841    0.02925         24        320:  50%|#####     | 7/14 [00:00<00:00,  8.54it/s]
           1/29     0.631G     0.1111    0.01841    0.02925         24        320:  57%|#####7    | 8/14 [00:00<00:00,  8.06it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1107    0.01823    0.02943         17        320:  57%|#####7    | 8/14 [00:01<00:00,  8.06it/s]
           1/29     0.631G     0.1107    0.01823    0.02943         17        320:  64%|######4   | 9/14 [00:01<00:00,  8.48it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G       0.11    0.01859    0.02942         25        320:  64%|######4   | 9/14 [00:01<00:00,  8.48it/s]
           1/29     0.631G       0.11    0.01859    0.02942         25        320:  71%|#######1  | 10/14 [00:01<00:00,  8.73it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1093    0.01845    0.02936         16        320:  71%|#######1  | 10/14 [00:01<00:00,  8.73it/s]
           1/29     0.631G     0.1093    0.01845    0.02936         16        320:  79%|#######8  | 11/14 [00:01<00:00,  8.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G     0.1088    0.01911    0.02922         29        320:  79%|#######8  | 11/14 [00:01<00:00,  8.11it/s]
           1/29     0.631G     0.1088    0.01911    0.02922         29        320:  86%|########5 | 12/14 [00:01<00:00,  8.58it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G      0.108    0.01882    0.02899         14        320:  86%|########5 | 12/14 [00:01<00:00,  8.58it/s]
           1/29     0.631G      0.108    0.01882    0.02899         14        320:  93%|#########2| 13/14 [00:01<00:00,  8.78it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           1/29     0.631G      0.108    0.01852    0.02907          8        320:  93%|#########2| 13/14 [00:01<00:00,  8.78it/s]
           1/29     0.631G      0.108    0.01852    0.02907          8        320: 100%|##########| 14/14 [00:01<00:00,  8.23it/s]
           1/29     0.631G      0.108    0.01852    0.02907          8        320: 100%|##########| 14/14 [00:01<00:00,  8.39it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.52it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.52it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.84it/s]
                       all         35         38    0.00684      0.921      0.013    0.00351
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09873    0.02755     0.0298         29        320:   0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G     0.0984    0.02234    0.02823         20        320:   0%|          | 0/14 [00:00<?, ?it/s]
           2/29     0.631G     0.0984    0.02234    0.02823         20        320:  14%|#4        | 2/14 [00:00<00:01, 11.43it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09982    0.02181    0.02817         22        320:  14%|#4        | 2/14 [00:00<00:01, 11.43it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09768    0.02214    0.02762         22        320:  14%|#4        | 2/14 [00:00<00:01, 11.43it/s]
           2/29     0.631G    0.09768    0.02214    0.02762         22        320:  29%|##8       | 4/14 [00:00<00:01,  8.86it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09729     0.0209    0.02749         15        320:  29%|##8       | 4/14 [00:00<00:01,  8.86it/s]
           2/29     0.631G    0.09729     0.0209    0.02749         15        320:  36%|###5      | 5/14 [00:00<00:01,  8.90it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09663    0.02031    0.02777         16        320:  36%|###5      | 5/14 [00:00<00:01,  8.90it/s]
           2/29     0.631G    0.09663    0.02031    0.02777         16        320:  43%|####2     | 6/14 [00:00<00:00,  8.24it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09652    0.02044    0.02781         23        320:  43%|####2     | 6/14 [00:00<00:00,  8.24it/s]
           2/29     0.631G    0.09652    0.02044    0.02781         23        320:  50%|#####     | 7/14 [00:00<00:00,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G     0.0967    0.02116    0.02793         28        320:  50%|#####     | 7/14 [00:00<00:00,  8.47it/s]
           2/29     0.631G     0.0967    0.02116    0.02793         28        320:  57%|#####7    | 8/14 [00:00<00:00,  8.49it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09618    0.02133    0.02784         24        320:  57%|#####7    | 8/14 [00:01<00:00,  8.49it/s]
           2/29     0.631G    0.09618    0.02133    0.02784         24        320:  64%|######4   | 9/14 [00:01<00:00,  8.50it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09637    0.02136    0.02787         22        320:  64%|######4   | 9/14 [00:01<00:00,  8.50it/s]
           2/29     0.631G    0.09637    0.02136    0.02787         22        320:  71%|#######1  | 10/14 [00:01<00:00,  7.92it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09584    0.02171    0.02772         25        320:  71%|#######1  | 10/14 [00:01<00:00,  7.92it/s]
           2/29     0.631G    0.09584    0.02171    0.02772         25        320:  79%|#######8  | 11/14 [00:01<00:00,  8.14it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09508    0.02204    0.02763         22        320:  79%|#######8  | 11/14 [00:01<00:00,  8.14it/s]
           2/29     0.631G    0.09508    0.02204    0.02763         22        320:  86%|########5 | 12/14 [00:01<00:00,  8.28it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09456    0.02187    0.02754         19        320:  86%|########5 | 12/14 [00:01<00:00,  8.28it/s]
           2/29     0.631G    0.09456    0.02187    0.02754         19        320:  93%|#########2| 13/14 [00:01<00:00,  8.33it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           2/29     0.631G    0.09419    0.02217    0.02758         13        320:  93%|#########2| 13/14 [00:01<00:00,  8.33it/s]
           2/29     0.631G    0.09419    0.02217    0.02758         13        320: 100%|##########| 14/14 [00:01<00:00,  7.45it/s]
           2/29     0.631G    0.09419    0.02217    0.02758         13        320: 100%|##########| 14/14 [00:01<00:00,  8.25it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  3.48it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  3.80it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  4.81it/s]
                       all         35         38    0.00362          1     0.0447     0.0151
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08085    0.02123    0.02843         16        320:   0%|          | 0/14 [00:00<?, ?it/s]
           3/29     0.631G    0.08085    0.02123    0.02843         16        320:   7%|7         | 1/14 [00:00<00:01,  9.69it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08262    0.02224    0.02806         23        320:   7%|7         | 1/14 [00:00<00:01,  9.69it/s]
           3/29     0.631G    0.08262    0.02224    0.02806         23        320:  14%|#4        | 2/14 [00:00<00:01,  9.50it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08412    0.02135    0.02759         17        320:  14%|#4        | 2/14 [00:00<00:01,  9.50it/s]
           3/29     0.631G    0.08412    0.02135    0.02759         17        320:  21%|##1       | 3/14 [00:00<00:01,  9.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08553    0.02396    0.02707         30        320:  21%|##1       | 3/14 [00:00<00:01,  9.47it/s]
           3/29     0.631G    0.08553    0.02396    0.02707         30        320:  29%|##8       | 4/14 [00:00<00:01,  8.32it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08665    0.02489    0.02693         27        320:  29%|##8       | 4/14 [00:00<00:01,  8.32it/s]
           3/29     0.631G    0.08665    0.02489    0.02693         27        320:  36%|###5      | 5/14 [00:00<00:01,  8.81it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G     0.0875    0.02528    0.02671         26        320:  36%|###5      | 5/14 [00:00<00:01,  8.81it/s]
           3/29     0.631G     0.0875    0.02528    0.02671         26        320:  43%|####2     | 6/14 [00:00<00:00,  8.93it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08799    0.02549    0.02671         28        320:  43%|####2     | 6/14 [00:00<00:00,  8.93it/s]
           3/29     0.631G    0.08799    0.02549    0.02671         28        320:  50%|#####     | 7/14 [00:00<00:00,  8.95it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08711    0.02508    0.02666         20        320:  50%|#####     | 7/14 [00:00<00:00,  8.95it/s]
           3/29     0.631G    0.08711    0.02508    0.02666         20        320:  57%|#####7    | 8/14 [00:00<00:00,  7.98it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08658    0.02434    0.02626         17        320:  57%|#####7    | 8/14 [00:01<00:00,  7.98it/s]
           3/29     0.631G    0.08658    0.02434    0.02626         17        320:  64%|######4   | 9/14 [00:01<00:00,  8.34it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G     0.0845    0.02405    0.02602         17        320:  64%|######4   | 9/14 [00:01<00:00,  8.34it/s]
           3/29     0.631G     0.0845    0.02405    0.02602         17        320:  71%|#######1  | 10/14 [00:01<00:00,  8.59it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08413    0.02483    0.02593         27        320:  71%|#######1  | 10/14 [00:01<00:00,  8.59it/s]
           3/29     0.631G    0.08413    0.02483    0.02593         27        320:  79%|#######8  | 11/14 [00:01<00:00,  8.87it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08369    0.02507    0.02571         26        320:  79%|#######8  | 11/14 [00:01<00:00,  8.87it/s]
           3/29     0.631G    0.08369    0.02507    0.02571         26        320:  86%|########5 | 12/14 [00:01<00:00,  9.16it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08324    0.02511    0.02555         22        320:  86%|########5 | 12/14 [00:01<00:00,  9.16it/s]
           3/29     0.631G    0.08324    0.02511    0.02555         22        320:  93%|#########2| 13/14 [00:01<00:00,  8.46it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           3/29     0.631G    0.08331    0.02461    0.02545          8        320:  93%|#########2| 13/14 [00:01<00:00,  8.46it/s]
           3/29     0.631G    0.08331    0.02461    0.02545          8        320: 100%|##########| 14/14 [00:01<00:00,  8.82it/s]
           3/29     0.631G    0.08331    0.02461    0.02545          8        320: 100%|##########| 14/14 [00:01<00:00,  8.76it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.51it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.57it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.88it/s]
                       all         35         38     0.0608      0.237      0.126      0.037
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07689     0.0262    0.02535         23        320:   0%|          | 0/14 [00:00<?, ?it/s]
           4/29     0.631G    0.07689     0.0262    0.02535         23        320:   7%|7         | 1/14 [00:00<00:01,  9.49it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.08097    0.03017    0.02505         30        320:   7%|7         | 1/14 [00:00<00:01,  9.49it/s]
           4/29     0.631G    0.08097    0.03017    0.02505         30        320:  14%|#4        | 2/14 [00:00<00:01,  9.35it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.08037    0.02671    0.02426         18        320:  14%|#4        | 2/14 [00:00<00:01,  9.35it/s]
           4/29     0.631G    0.08037    0.02671    0.02426         18        320:  21%|##1       | 3/14 [00:00<00:01,  9.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07975    0.02542    0.02437         18        320:  21%|##1       | 3/14 [00:00<00:01,  9.18it/s]
           4/29     0.631G    0.07975    0.02542    0.02437         18        320:  29%|##8       | 4/14 [00:00<00:01,  8.30it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07887    0.02872    0.02445         34        320:  29%|##8       | 4/14 [00:00<00:01,  8.30it/s]
           4/29     0.631G    0.07887    0.02872    0.02445         34        320:  36%|###5      | 5/14 [00:00<00:01,  8.73it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07888     0.0287    0.02443         26        320:  36%|###5      | 5/14 [00:00<00:01,  8.73it/s]
           4/29     0.631G    0.07888     0.0287    0.02443         26        320:  43%|####2     | 6/14 [00:00<00:00,  8.93it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07928     0.0283     0.0244         24        320:  43%|####2     | 6/14 [00:00<00:00,  8.93it/s]
           4/29     0.631G    0.07928     0.0283     0.0244         24        320:  50%|#####     | 7/14 [00:00<00:00,  9.04it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07851    0.02761    0.02429         18        320:  50%|#####     | 7/14 [00:00<00:00,  9.04it/s]
           4/29     0.631G    0.07851    0.02761    0.02429         18        320:  57%|#####7    | 8/14 [00:00<00:00,  9.17it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07752    0.02836    0.02423         26        320:  57%|#####7    | 8/14 [00:01<00:00,  9.17it/s]
           4/29     0.631G    0.07752    0.02836    0.02423         26        320:  64%|######4   | 9/14 [00:01<00:00,  8.43it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07746    0.02763    0.02391         19        320:  64%|######4   | 9/14 [00:01<00:00,  8.43it/s]
           4/29     0.631G    0.07746    0.02763    0.02391         19        320:  71%|#######1  | 10/14 [00:01<00:00,  8.71it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07645    0.02729    0.02349         18        320:  71%|#######1  | 10/14 [00:01<00:00,  8.71it/s]
           4/29     0.631G    0.07645    0.02729    0.02349         18        320:  79%|#######8  | 11/14 [00:01<00:00,  8.89it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07614    0.02787    0.02326         31        320:  79%|#######8  | 11/14 [00:01<00:00,  8.89it/s]
           4/29     0.631G    0.07614    0.02787    0.02326         31        320:  86%|########5 | 12/14 [00:01<00:00,  9.08it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07598    0.02736    0.02312         18        320:  86%|########5 | 12/14 [00:01<00:00,  9.08it/s]
           4/29     0.631G    0.07598    0.02736    0.02312         18        320:  93%|#########2| 13/14 [00:01<00:00,  9.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           4/29     0.631G    0.07653    0.02708    0.02308         11        320:  93%|#########2| 13/14 [00:01<00:00,  9.11it/s]
           4/29     0.631G    0.07653    0.02708    0.02308         11        320: 100%|##########| 14/14 [00:01<00:00,  9.09it/s]
           4/29     0.631G    0.07653    0.02708    0.02308         11        320: 100%|##########| 14/14 [00:01<00:00,  8.95it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.49it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.32it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.53it/s]
                       all         35         38      0.084       0.32      0.131     0.0385
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G     0.0693    0.02257    0.02204         21        320:   0%|          | 0/14 [00:00<?, ?it/s]
           5/29     0.631G     0.0693    0.02257    0.02204         21        320:   7%|7         | 1/14 [00:00<00:01,  7.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07492    0.02709    0.02148         28        320:   7%|7         | 1/14 [00:00<00:01,  7.11it/s]
           5/29     0.631G    0.07492    0.02709    0.02148         28        320:  14%|#4        | 2/14 [00:00<00:01,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07741    0.02537    0.02261         25        320:  14%|#4        | 2/14 [00:00<00:01,  8.47it/s]
           5/29     0.631G    0.07741    0.02537    0.02261         25        320:  21%|##1       | 3/14 [00:00<00:01,  8.89it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07788    0.02395    0.02153         18        320:  21%|##1       | 3/14 [00:00<00:01,  8.89it/s]
           5/29     0.631G    0.07788    0.02395    0.02153         18        320:  29%|##8       | 4/14 [00:00<00:01,  9.03it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07683    0.02457    0.02198         24        320:  29%|##8       | 4/14 [00:00<00:01,  9.03it/s]
           5/29     0.631G    0.07683    0.02457    0.02198         24        320:  36%|###5      | 5/14 [00:00<00:00,  9.03it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07638     0.0243    0.02187         17        320:  36%|###5      | 5/14 [00:00<00:00,  9.03it/s]
           5/29     0.631G    0.07638     0.0243    0.02187         17        320:  43%|####2     | 6/14 [00:00<00:00,  9.13it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07606    0.02329    0.02163         15        320:  43%|####2     | 6/14 [00:00<00:00,  9.13it/s]
           5/29     0.631G    0.07606    0.02329    0.02163         15        320:  50%|#####     | 7/14 [00:00<00:00,  8.41it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07719    0.02318    0.02185         27        320:  50%|#####     | 7/14 [00:00<00:00,  8.41it/s]
           5/29     0.631G    0.07719    0.02318    0.02185         27        320:  57%|#####7    | 8/14 [00:00<00:00,  8.77it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07628    0.02385    0.02185         24        320:  57%|#####7    | 8/14 [00:00<00:00,  8.77it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07506    0.02332    0.02149         16        320:  57%|#####7    | 8/14 [00:01<00:00,  8.77it/s]
           5/29     0.631G    0.07506    0.02332    0.02149         16        320:  71%|#######1  | 10/14 [00:01<00:00,  9.66it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07514    0.02365    0.02176         24        320:  71%|#######1  | 10/14 [00:01<00:00,  9.66it/s]
           5/29     0.631G    0.07514    0.02365    0.02176         24        320:  79%|#######8  | 11/14 [00:01<00:00,  9.53it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07458    0.02374    0.02186         21        320:  79%|#######8  | 11/14 [00:01<00:00,  9.53it/s]
           5/29     0.631G    0.07458    0.02374    0.02186         21        320:  86%|########5 | 12/14 [00:01<00:00,  9.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07395    0.02395    0.02168         20        320:  86%|########5 | 12/14 [00:01<00:00,  9.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           5/29     0.631G    0.07453    0.02378    0.02169          9        320:  86%|########5 | 12/14 [00:01<00:00,  9.47it/s]
           5/29     0.631G    0.07453    0.02378    0.02169          9        320: 100%|##########| 14/14 [00:01<00:00,  9.01it/s]
           5/29     0.631G    0.07453    0.02378    0.02169          9        320: 100%|##########| 14/14 [00:01<00:00,  9.01it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  3.72it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.29it/s]
                       all         35         38      0.581      0.132     0.0937     0.0343
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.08129    0.02111    0.02079         20        320:   0%|          | 0/14 [00:00<?, ?it/s]
           6/29     0.631G    0.08129    0.02111    0.02079         20        320:   7%|7         | 1/14 [00:00<00:01,  9.45it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07832    0.02573    0.02066         25        320:   7%|7         | 1/14 [00:00<00:01,  9.45it/s]
           6/29     0.631G    0.07832    0.02573    0.02066         25        320:  14%|#4        | 2/14 [00:00<00:01,  9.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07881    0.02595    0.02035         26        320:  14%|#4        | 2/14 [00:00<00:01,  9.37it/s]
           6/29     0.631G    0.07881    0.02595    0.02035         26        320:  21%|##1       | 3/14 [00:00<00:01,  9.24it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07752    0.02585    0.02001         24        320:  21%|##1       | 3/14 [00:00<00:01,  9.24it/s]
           6/29     0.631G    0.07752    0.02585    0.02001         24        320:  29%|##8       | 4/14 [00:00<00:01,  9.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07802    0.02567     0.0196         24        320:  29%|##8       | 4/14 [00:00<00:01,  9.27it/s]
           6/29     0.631G    0.07802    0.02567     0.0196         24        320:  36%|###5      | 5/14 [00:00<00:00,  9.24it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07688     0.0271       0.02         28        320:  36%|###5      | 5/14 [00:00<00:00,  9.24it/s]
           6/29     0.631G    0.07688     0.0271       0.02         28        320:  43%|####2     | 6/14 [00:00<00:00,  9.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07621     0.0263    0.02049         18        320:  43%|####2     | 6/14 [00:00<00:00,  9.27it/s]
           6/29     0.631G    0.07621     0.0263    0.02049         18        320:  50%|#####     | 7/14 [00:00<00:00,  8.42it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07699     0.0253    0.01994         18        320:  50%|#####     | 7/14 [00:00<00:00,  8.42it/s]
           6/29     0.631G    0.07699     0.0253    0.01994         18        320:  57%|#####7    | 8/14 [00:00<00:00,  8.40it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07765    0.02505    0.01955         22        320:  57%|#####7    | 8/14 [00:01<00:00,  8.40it/s]
           6/29     0.631G    0.07765    0.02505    0.01955         22        320:  64%|######4   | 9/14 [00:01<00:00,  8.33it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07651    0.02434    0.01904         15        320:  64%|######4   | 9/14 [00:01<00:00,  8.33it/s]
           6/29     0.631G    0.07651    0.02434    0.01904         15        320:  71%|#######1  | 10/14 [00:01<00:00,  7.99it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07572    0.02431    0.01937         21        320:  71%|#######1  | 10/14 [00:01<00:00,  7.99it/s]
           6/29     0.631G    0.07572    0.02431    0.01937         21        320:  79%|#######8  | 11/14 [00:01<00:00,  7.54it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07563    0.02412    0.01954         20        320:  79%|#######8  | 11/14 [00:01<00:00,  7.54it/s]
           6/29     0.631G    0.07563    0.02412    0.01954         20        320:  86%|########5 | 12/14 [00:01<00:00,  7.85it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07576    0.02378    0.01916         18        320:  86%|########5 | 12/14 [00:01<00:00,  7.85it/s]
           6/29     0.631G    0.07576    0.02378    0.01916         18        320:  93%|#########2| 13/14 [00:01<00:00,  7.94it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           6/29     0.631G    0.07591    0.02455    0.01913         18        320:  93%|#########2| 13/14 [00:01<00:00,  7.94it/s]
           6/29     0.631G    0.07591    0.02455    0.01913         18        320: 100%|##########| 14/14 [00:01<00:00,  8.06it/s]
           6/29     0.631G    0.07591    0.02455    0.01913         18        320: 100%|##########| 14/14 [00:01<00:00,  8.36it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.79it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.99it/s]
                       all         35         38      0.573      0.158       0.12     0.0472
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07748    0.01764    0.01619         18        320:   0%|          | 0/14 [00:00<?, ?it/s]
           7/29     0.631G    0.07748    0.01764    0.01619         18        320:   7%|7         | 1/14 [00:00<00:02,  5.74it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07437    0.02122    0.01825         22        320:   7%|7         | 1/14 [00:00<00:02,  5.74it/s]
           7/29     0.631G    0.07437    0.02122    0.01825         22        320:  14%|#4        | 2/14 [00:00<00:01,  7.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07062    0.02184    0.01743         20        320:  14%|#4        | 2/14 [00:00<00:01,  7.18it/s]
           7/29     0.631G    0.07062    0.02184    0.01743         20        320:  21%|##1       | 3/14 [00:00<00:01,  7.89it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07089    0.02214    0.01739         20        320:  21%|##1       | 3/14 [00:00<00:01,  7.89it/s]
           7/29     0.631G    0.07089    0.02214    0.01739         20        320:  29%|##8       | 4/14 [00:00<00:01,  8.44it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07042    0.02402    0.01735         26        320:  29%|##8       | 4/14 [00:00<00:01,  8.44it/s]
           7/29     0.631G    0.07042    0.02402    0.01735         26        320:  36%|###5      | 5/14 [00:00<00:01,  8.53it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07208    0.02301     0.0174         17        320:  36%|###5      | 5/14 [00:00<00:01,  8.53it/s]
           7/29     0.631G    0.07208    0.02301     0.0174         17        320:  43%|####2     | 6/14 [00:00<00:00,  8.56it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07244    0.02376    0.01715         25        320:  43%|####2     | 6/14 [00:00<00:00,  8.56it/s]
           7/29     0.631G    0.07244    0.02376    0.01715         25        320:  50%|#####     | 7/14 [00:00<00:00,  8.79it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07143    0.02403    0.01706         22        320:  50%|#####     | 7/14 [00:00<00:00,  8.79it/s]
           7/29     0.631G    0.07143    0.02403    0.01706         22        320:  57%|#####7    | 8/14 [00:00<00:00,  8.81it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07236    0.02295    0.01773         14        320:  57%|#####7    | 8/14 [00:01<00:00,  8.81it/s]
           7/29     0.631G    0.07236    0.02295    0.01773         14        320:  64%|######4   | 9/14 [00:01<00:00,  8.17it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07285    0.02249    0.01772         17        320:  64%|######4   | 9/14 [00:01<00:00,  8.17it/s]
           7/29     0.631G    0.07285    0.02249    0.01772         17        320:  71%|#######1  | 10/14 [00:01<00:00,  8.62it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07308    0.02193    0.01775         15        320:  71%|#######1  | 10/14 [00:01<00:00,  8.62it/s]
           7/29     0.631G    0.07308    0.02193    0.01775         15        320:  79%|#######8  | 11/14 [00:01<00:00,  8.89it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07299    0.02211    0.01786         23        320:  79%|#######8  | 11/14 [00:01<00:00,  8.89it/s]
           7/29     0.631G    0.07299    0.02211    0.01786         23        320:  86%|########5 | 12/14 [00:01<00:00,  9.13it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07307    0.02234    0.01778         23        320:  86%|########5 | 12/14 [00:01<00:00,  9.13it/s]
           7/29     0.631G    0.07307    0.02234    0.01778         23        320:  93%|#########2| 13/14 [00:01<00:00,  9.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           7/29     0.631G    0.07365    0.02241    0.01769         11        320:  93%|#########2| 13/14 [00:01<00:00,  9.18it/s]
           7/29     0.631G    0.07365    0.02241    0.01769         11        320: 100%|##########| 14/14 [00:01<00:00,  9.21it/s]
           7/29     0.631G    0.07365    0.02241    0.01769         11        320: 100%|##########| 14/14 [00:01<00:00,  8.62it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.83it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.62it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.79it/s]
                       all         35         38       0.58      0.211      0.134     0.0329
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.08088    0.01824    0.01549         19        320:   0%|          | 0/14 [00:00<?, ?it/s]
           8/29     0.631G    0.08088    0.01824    0.01549         19        320:   7%|7         | 1/14 [00:00<00:01,  9.49it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07509    0.01788    0.01263         17        320:   7%|7         | 1/14 [00:00<00:01,  9.49it/s]
           8/29     0.631G    0.07509    0.01788    0.01263         17        320:  14%|#4        | 2/14 [00:00<00:01,  9.48it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07705    0.02016    0.01515         25        320:  14%|#4        | 2/14 [00:00<00:01,  9.48it/s]
           8/29     0.631G    0.07705    0.02016    0.01515         25        320:  21%|##1       | 3/14 [00:00<00:01,  8.23it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07245    0.02147    0.01487         24        320:  21%|##1       | 3/14 [00:00<00:01,  8.23it/s]
           8/29     0.631G    0.07245    0.02147    0.01487         24        320:  29%|##8       | 4/14 [00:00<00:01,  8.83it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G     0.0721    0.02211    0.01539         21        320:  29%|##8       | 4/14 [00:00<00:01,  8.83it/s]
           8/29     0.631G     0.0721    0.02211    0.01539         21        320:  36%|###5      | 5/14 [00:00<00:01,  8.66it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07286    0.02199    0.01594         22        320:  36%|###5      | 5/14 [00:00<00:01,  8.66it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07405      0.024    0.01613         33        320:  36%|###5      | 5/14 [00:00<00:01,  8.66it/s]
           8/29     0.631G    0.07405      0.024    0.01613         33        320:  50%|#####     | 7/14 [00:00<00:00,  9.05it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07304    0.02294    0.01634         14        320:  50%|#####     | 7/14 [00:00<00:00,  9.05it/s]
           8/29     0.631G    0.07304    0.02294    0.01634         14        320:  57%|#####7    | 8/14 [00:00<00:00,  8.84it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07333    0.02337    0.01675         24        320:  57%|#####7    | 8/14 [00:01<00:00,  8.84it/s]
           8/29     0.631G    0.07333    0.02337    0.01675         24        320:  64%|######4   | 9/14 [00:01<00:00,  8.61it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07294    0.02297    0.01669         17        320:  64%|######4   | 9/14 [00:01<00:00,  8.61it/s]
           8/29     0.631G    0.07294    0.02297    0.01669         17        320:  71%|#######1  | 10/14 [00:01<00:00,  8.65it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07268    0.02302    0.01682         20        320:  71%|#######1  | 10/14 [00:01<00:00,  8.65it/s]
           8/29     0.631G    0.07268    0.02302    0.01682         20        320:  79%|#######8  | 11/14 [00:01<00:00,  8.03it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07253    0.02305    0.01711         23        320:  79%|#######8  | 11/14 [00:01<00:00,  8.03it/s]
           8/29     0.631G    0.07253    0.02305    0.01711         23        320:  86%|########5 | 12/14 [00:01<00:00,  8.21it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07293    0.02333    0.01689         27        320:  86%|########5 | 12/14 [00:01<00:00,  8.21it/s]
           8/29     0.631G    0.07293    0.02333    0.01689         27        320:  93%|#########2| 13/14 [00:01<00:00,  8.21it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           8/29     0.631G    0.07392    0.02429    0.01684         18        320:  93%|#########2| 13/14 [00:01<00:00,  8.21it/s]
           8/29     0.631G    0.07392    0.02429    0.01684         18        320: 100%|##########| 14/14 [00:01<00:00,  7.94it/s]
           8/29     0.631G    0.07392    0.02429    0.01684         18        320: 100%|##########| 14/14 [00:01<00:00,  8.42it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.58it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.69it/s]
                       all         35         38      0.572      0.128      0.112     0.0391
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.07353    0.01886    0.01732         20        320:   0%|          | 0/14 [00:00<?, ?it/s]
           9/29     0.631G    0.07353    0.01886    0.01732         20        320:   7%|7         | 1/14 [00:00<00:01,  9.29it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.07433    0.02339    0.01679         26        320:   7%|7         | 1/14 [00:00<00:01,  9.29it/s]
           9/29     0.631G    0.07433    0.02339    0.01679         26        320:  14%|#4        | 2/14 [00:00<00:01,  9.39it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.07081    0.02368    0.01557         23        320:  14%|#4        | 2/14 [00:00<00:01,  9.39it/s]
           9/29     0.631G    0.07081    0.02368    0.01557         23        320:  21%|##1       | 3/14 [00:00<00:01,  8.80it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.07268    0.02698    0.01596         35        320:  21%|##1       | 3/14 [00:00<00:01,  8.80it/s]
           9/29     0.631G    0.07268    0.02698    0.01596         35        320:  29%|##8       | 4/14 [00:00<00:01,  8.31it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.07081    0.02556    0.01563         16        320:  29%|##8       | 4/14 [00:00<00:01,  8.31it/s]
           9/29     0.631G    0.07081    0.02556    0.01563         16        320:  36%|###5      | 5/14 [00:00<00:01,  7.72it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.07016    0.02611    0.01573         26        320:  36%|###5      | 5/14 [00:00<00:01,  7.72it/s]
           9/29     0.631G    0.07016    0.02611    0.01573         26        320:  43%|####2     | 6/14 [00:00<00:01,  7.73it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06973     0.0258    0.01577         22        320:  43%|####2     | 6/14 [00:00<00:01,  7.73it/s]
           9/29     0.631G    0.06973     0.0258    0.01577         22        320:  50%|#####     | 7/14 [00:00<00:00,  7.88it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06974    0.02525    0.01533         19        320:  50%|#####     | 7/14 [00:00<00:00,  7.88it/s]
           9/29     0.631G    0.06974    0.02525    0.01533         19        320:  57%|#####7    | 8/14 [00:00<00:00,  7.74it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06903    0.02459    0.01507         15        320:  57%|#####7    | 8/14 [00:01<00:00,  7.74it/s]
           9/29     0.631G    0.06903    0.02459    0.01507         15        320:  64%|######4   | 9/14 [00:01<00:00,  7.41it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06944    0.02442    0.01546         20        320:  64%|######4   | 9/14 [00:01<00:00,  7.41it/s]
           9/29     0.631G    0.06944    0.02442    0.01546         20        320:  71%|#######1  | 10/14 [00:01<00:00,  7.75it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06912    0.02478    0.01522         24        320:  71%|#######1  | 10/14 [00:01<00:00,  7.75it/s]
           9/29     0.631G    0.06912    0.02478    0.01522         24        320:  79%|#######8  | 11/14 [00:01<00:00,  7.80it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06957      0.025    0.01547         26        320:  79%|#######8  | 11/14 [00:01<00:00,  7.80it/s]
           9/29     0.631G    0.06957      0.025    0.01547         26        320:  86%|########5 | 12/14 [00:01<00:00,  8.09it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06963    0.02477    0.01585         21        320:  86%|########5 | 12/14 [00:01<00:00,  8.09it/s]
           9/29     0.631G    0.06963    0.02477    0.01585         21        320:  93%|#########2| 13/14 [00:01<00:00,  7.51it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
           9/29     0.631G    0.06949    0.02564    0.01622         16        320:  93%|#########2| 13/14 [00:01<00:00,  7.51it/s]
           9/29     0.631G    0.06949    0.02564    0.01622         16        320: 100%|##########| 14/14 [00:01<00:00,  7.91it/s]
           9/29     0.631G    0.06949    0.02564    0.01622         16        320: 100%|##########| 14/14 [00:01<00:00,  7.93it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.82it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.65it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.85it/s]
                       all         35         38      0.336      0.316      0.167     0.0539
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06345    0.02941    0.01398         25        320:   0%|          | 0/14 [00:00<?, ?it/s]
          10/29     0.631G    0.06345    0.02941    0.01398         25        320:   7%|7         | 1/14 [00:00<00:01,  8.62it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G     0.0618    0.02684    0.01235         20        320:   7%|7         | 1/14 [00:00<00:01,  8.62it/s]
          10/29     0.631G     0.0618    0.02684    0.01235         20        320:  14%|#4        | 2/14 [00:00<00:01,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G     0.0625    0.02487    0.01375         21        320:  14%|#4        | 2/14 [00:00<00:01,  8.47it/s]
          10/29     0.631G     0.0625    0.02487    0.01375         21        320:  21%|##1       | 3/14 [00:00<00:01,  8.77it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06392    0.02726      0.014         33        320:  21%|##1       | 3/14 [00:00<00:01,  8.77it/s]
          10/29     0.631G    0.06392    0.02726      0.014         33        320:  29%|##8       | 4/14 [00:00<00:01,  8.97it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G     0.0651    0.02675    0.01437         25        320:  29%|##8       | 4/14 [00:00<00:01,  8.97it/s]
          10/29     0.631G     0.0651    0.02675    0.01437         25        320:  36%|###5      | 5/14 [00:00<00:00,  9.17it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06484    0.02761    0.01433         27        320:  36%|###5      | 5/14 [00:00<00:00,  9.17it/s]
          10/29     0.631G    0.06484    0.02761    0.01433         27        320:  43%|####2     | 6/14 [00:00<00:00,  8.88it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06572     0.0279    0.01421         28        320:  43%|####2     | 6/14 [00:00<00:00,  8.88it/s]
          10/29     0.631G    0.06572     0.0279    0.01421         28        320:  50%|#####     | 7/14 [00:00<00:00,  7.65it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06508    0.02678    0.01361         16        320:  50%|#####     | 7/14 [00:00<00:00,  7.65it/s]
          10/29     0.631G    0.06508    0.02678    0.01361         16        320:  57%|#####7    | 8/14 [00:00<00:00,  8.07it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06417     0.0264    0.01353         21        320:  57%|#####7    | 8/14 [00:01<00:00,  8.07it/s]
          10/29     0.631G    0.06417     0.0264    0.01353         21        320:  64%|######4   | 9/14 [00:01<00:00,  8.29it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06328    0.02614    0.01388         20        320:  64%|######4   | 9/14 [00:01<00:00,  8.29it/s]
          10/29     0.631G    0.06328    0.02614    0.01388         20        320:  71%|#######1  | 10/14 [00:01<00:00,  8.45it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06347    0.02654    0.01383         31        320:  71%|#######1  | 10/14 [00:01<00:00,  8.45it/s]
          10/29     0.631G    0.06347    0.02654    0.01383         31        320:  79%|#######8  | 11/14 [00:01<00:00,  8.22it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06307    0.02567    0.01347         16        320:  79%|#######8  | 11/14 [00:01<00:00,  8.22it/s]
          10/29     0.631G    0.06307    0.02567    0.01347         16        320:  86%|########5 | 12/14 [00:01<00:00,  8.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06308    0.02592    0.01395         30        320:  86%|########5 | 12/14 [00:01<00:00,  8.37it/s]
          10/29     0.631G    0.06308    0.02592    0.01395         30        320:  93%|#########2| 13/14 [00:01<00:00,  8.44it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          10/29     0.631G    0.06338    0.02532    0.01381          8        320:  93%|#########2| 13/14 [00:01<00:00,  8.44it/s]
          10/29     0.631G    0.06338    0.02532    0.01381          8        320: 100%|##########| 14/14 [00:01<00:00,  8.13it/s]
          10/29     0.631G    0.06338    0.02532    0.01381          8        320: 100%|##########| 14/14 [00:01<00:00,  8.36it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  3.93it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  3.39it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  4.55it/s]
                       all         35         38      0.176      0.342      0.156     0.0626
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06781    0.02241    0.01554         22        320:   0%|          | 0/14 [00:00<?, ?it/s]
          11/29     0.631G    0.06781    0.02241    0.01554         22        320:   7%|7         | 1/14 [00:00<00:01,  6.95it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06495    0.02602    0.01157         30        320:   7%|7         | 1/14 [00:00<00:01,  6.95it/s]
          11/29     0.631G    0.06495    0.02602    0.01157         30        320:  14%|#4        | 2/14 [00:00<00:01,  8.24it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06525    0.02369     0.0125         17        320:  14%|#4        | 2/14 [00:00<00:01,  8.24it/s]
          11/29     0.631G    0.06525    0.02369     0.0125         17        320:  21%|##1       | 3/14 [00:00<00:01,  8.75it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G     0.0665     0.0249    0.01272         29        320:  21%|##1       | 3/14 [00:00<00:01,  8.75it/s]
          11/29     0.631G     0.0665     0.0249    0.01272         29        320:  29%|##8       | 4/14 [00:00<00:01,  8.93it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06707    0.02559    0.01311         24        320:  29%|##8       | 4/14 [00:00<00:01,  8.93it/s]
          11/29     0.631G    0.06707    0.02559    0.01311         24        320:  36%|###5      | 5/14 [00:00<00:01,  8.99it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06659     0.0244    0.01406         19        320:  36%|###5      | 5/14 [00:00<00:01,  8.99it/s]
          11/29     0.631G    0.06659     0.0244    0.01406         19        320:  43%|####2     | 6/14 [00:00<00:00,  9.12it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06612    0.02423    0.01412         23        320:  43%|####2     | 6/14 [00:00<00:00,  9.12it/s]
          11/29     0.631G    0.06612    0.02423    0.01412         23        320:  50%|#####     | 7/14 [00:00<00:00,  9.05it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06547     0.0236    0.01451         19        320:  50%|#####     | 7/14 [00:00<00:00,  9.05it/s]
          11/29     0.631G    0.06547     0.0236    0.01451         19        320:  57%|#####7    | 8/14 [00:00<00:00,  9.08it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06606    0.02347    0.01428         23        320:  57%|#####7    | 8/14 [00:01<00:00,  9.08it/s]
          11/29     0.631G    0.06606    0.02347    0.01428         23        320:  64%|######4   | 9/14 [00:01<00:00,  8.07it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06607    0.02288    0.01413         16        320:  64%|######4   | 9/14 [00:01<00:00,  8.07it/s]
          11/29     0.631G    0.06607    0.02288    0.01413         16        320:  71%|#######1  | 10/14 [00:01<00:00,  8.35it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06474    0.02279     0.0137         21        320:  71%|#######1  | 10/14 [00:01<00:00,  8.35it/s]
          11/29     0.631G    0.06474    0.02279     0.0137         21        320:  79%|#######8  | 11/14 [00:01<00:00,  8.48it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06427    0.02274    0.01357         19        320:  79%|#######8  | 11/14 [00:01<00:00,  8.48it/s]
          11/29     0.631G    0.06427    0.02274    0.01357         19        320:  86%|########5 | 12/14 [00:01<00:00,  8.79it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06407    0.02313    0.01365         22        320:  86%|########5 | 12/14 [00:01<00:00,  8.79it/s]
          11/29     0.631G    0.06407    0.02313    0.01365         22        320:  93%|#########2| 13/14 [00:01<00:00,  8.95it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          11/29     0.631G    0.06439    0.02322    0.01406         13        320:  93%|#########2| 13/14 [00:01<00:00,  8.95it/s]
          11/29     0.631G    0.06439    0.02322    0.01406         13        320: 100%|##########| 14/14 [00:01<00:00,  9.11it/s]
          11/29     0.631G    0.06439    0.02322    0.01406         13        320: 100%|##########| 14/14 [00:01<00:00,  8.77it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.98it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.74it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.09it/s]
                       all         35         38      0.244      0.237      0.155     0.0549
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05858      0.033    0.01803         29        320:   0%|          | 0/14 [00:00<?, ?it/s]
          12/29     0.631G    0.05858      0.033    0.01803         29        320:   7%|7         | 1/14 [00:00<00:01,  9.70it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05952    0.02768    0.01608         22        320:   7%|7         | 1/14 [00:00<00:01,  9.70it/s]
          12/29     0.631G    0.05952    0.02768    0.01608         22        320:  14%|#4        | 2/14 [00:00<00:01,  8.87it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G     0.0597    0.02789    0.01457         26        320:  14%|#4        | 2/14 [00:00<00:01,  8.87it/s]
          12/29     0.631G     0.0597    0.02789    0.01457         26        320:  21%|##1       | 3/14 [00:00<00:01,  7.88it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05963    0.02662    0.01284         20        320:  21%|##1       | 3/14 [00:00<00:01,  7.88it/s]
          12/29     0.631G    0.05963    0.02662    0.01284         20        320:  29%|##8       | 4/14 [00:00<00:01,  8.21it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05883    0.02535    0.01312         17        320:  29%|##8       | 4/14 [00:00<00:01,  8.21it/s]
          12/29     0.631G    0.05883    0.02535    0.01312         17        320:  36%|###5      | 5/14 [00:00<00:01,  8.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05751    0.02443    0.01326         17        320:  36%|###5      | 5/14 [00:00<00:01,  8.27it/s]
          12/29     0.631G    0.05751    0.02443    0.01326         17        320:  43%|####2     | 6/14 [00:00<00:00,  8.31it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05827    0.02539    0.01351         27        320:  43%|####2     | 6/14 [00:00<00:00,  8.31it/s]
          12/29     0.631G    0.05827    0.02539    0.01351         27        320:  50%|#####     | 7/14 [00:00<00:00,  8.08it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05795    0.02465    0.01276         17        320:  50%|#####     | 7/14 [00:00<00:00,  8.08it/s]
          12/29     0.631G    0.05795    0.02465    0.01276         17        320:  57%|#####7    | 8/14 [00:00<00:00,  8.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05843    0.02506    0.01233         26        320:  57%|#####7    | 8/14 [00:01<00:00,  8.37it/s]
          12/29     0.631G    0.05843    0.02506    0.01233         26        320:  64%|######4   | 9/14 [00:01<00:00,  8.71it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05838    0.02434    0.01188         18        320:  64%|######4   | 9/14 [00:01<00:00,  8.71it/s]
          12/29     0.631G    0.05838    0.02434    0.01188         18        320:  71%|#######1  | 10/14 [00:01<00:00,  8.91it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05894     0.0239     0.0114         18        320:  71%|#######1  | 10/14 [00:01<00:00,  8.91it/s]
          12/29     0.631G    0.05894     0.0239     0.0114         18        320:  79%|#######8  | 11/14 [00:01<00:00,  8.19it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05891    0.02335    0.01104         16        320:  79%|#######8  | 11/14 [00:01<00:00,  8.19it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.05994    0.02312    0.01141         19        320:  79%|#######8  | 11/14 [00:01<00:00,  8.19it/s]
          12/29     0.631G    0.05994    0.02312    0.01141         19        320:  93%|#########2| 13/14 [00:01<00:00,  8.82it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          12/29     0.631G    0.06088    0.02316    0.01138         12        320:  93%|#########2| 13/14 [00:01<00:00,  8.82it/s]
          12/29     0.631G    0.06088    0.02316    0.01138         12        320: 100%|##########| 14/14 [00:01<00:00,  8.90it/s]
          12/29     0.631G    0.06088    0.02316    0.01138         12        320: 100%|##########| 14/14 [00:01<00:00,  8.58it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  3.40it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  3.85it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  4.96it/s]
                       all         35         38      0.142      0.237      0.161     0.0676
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06691    0.02381   0.007741         26        320:   0%|          | 0/14 [00:00<?, ?it/s]
          13/29     0.631G    0.06691    0.02381   0.007741         26        320:   7%|7         | 1/14 [00:00<00:01,  9.33it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06678    0.02367    0.01183         21        320:   7%|7         | 1/14 [00:00<00:01,  9.33it/s]
          13/29     0.631G    0.06678    0.02367    0.01183         21        320:  14%|#4        | 2/14 [00:00<00:01,  9.33it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G     0.0679    0.02432    0.01196         26        320:  14%|#4        | 2/14 [00:00<00:01,  9.33it/s]
          13/29     0.631G     0.0679    0.02432    0.01196         26        320:  21%|##1       | 3/14 [00:00<00:01,  9.25it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06871    0.02562    0.01292         24        320:  21%|##1       | 3/14 [00:00<00:01,  9.25it/s]
          13/29     0.631G    0.06871    0.02562    0.01292         24        320:  29%|##8       | 4/14 [00:00<00:01,  9.31it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06564    0.02483     0.0124         18        320:  29%|##8       | 4/14 [00:00<00:01,  9.31it/s]
          13/29     0.631G    0.06564    0.02483     0.0124         18        320:  36%|###5      | 5/14 [00:00<00:01,  8.45it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06509    0.02455    0.01153         21        320:  36%|###5      | 5/14 [00:00<00:01,  8.45it/s]
          13/29     0.631G    0.06509    0.02455    0.01153         21        320:  43%|####2     | 6/14 [00:00<00:00,  8.90it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06323    0.02343    0.01127         15        320:  43%|####2     | 6/14 [00:00<00:00,  8.90it/s]
          13/29     0.631G    0.06323    0.02343    0.01127         15        320:  50%|#####     | 7/14 [00:00<00:00,  9.06it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06266    0.02405    0.01121         26        320:  50%|#####     | 7/14 [00:00<00:00,  9.06it/s]
          13/29     0.631G    0.06266    0.02405    0.01121         26        320:  57%|#####7    | 8/14 [00:00<00:00,  9.20it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G     0.0623    0.02377    0.01096         21        320:  57%|#####7    | 8/14 [00:00<00:00,  9.20it/s]
          13/29     0.631G     0.0623    0.02377    0.01096         21        320:  64%|######4   | 9/14 [00:00<00:00,  9.19it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06193    0.02388    0.01112         25        320:  64%|######4   | 9/14 [00:01<00:00,  9.19it/s]
          13/29     0.631G    0.06193    0.02388    0.01112         25        320:  71%|#######1  | 10/14 [00:01<00:00,  9.23it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06196    0.02432    0.01086         28        320:  71%|#######1  | 10/14 [00:01<00:00,  9.23it/s]
          13/29     0.631G    0.06196    0.02432    0.01086         28        320:  79%|#######8  | 11/14 [00:01<00:00,  9.26it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06211    0.02463    0.01081         30        320:  79%|#######8  | 11/14 [00:01<00:00,  9.26it/s]
          13/29     0.631G    0.06211    0.02463    0.01081         30        320:  86%|########5 | 12/14 [00:01<00:00,  9.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06186    0.02478     0.0111         25        320:  86%|########5 | 12/14 [00:01<00:00,  9.38it/s]
          13/29     0.631G    0.06186    0.02478     0.0111         25        320:  93%|#########2| 13/14 [00:01<00:00,  8.56it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          13/29     0.631G    0.06145    0.02564    0.01107         17        320:  93%|#########2| 13/14 [00:01<00:00,  8.56it/s]
          13/29     0.631G    0.06145    0.02564    0.01107         17        320: 100%|##########| 14/14 [00:01<00:00,  8.89it/s]
          13/29     0.631G    0.06145    0.02564    0.01107         17        320: 100%|##########| 14/14 [00:01<00:00,  9.03it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.87it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.47it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.80it/s]
                       all         35         38      0.185      0.263      0.202     0.0928
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.06294    0.02488   0.007993         23        320:   0%|          | 0/14 [00:00<?, ?it/s]
          14/29     0.631G    0.06294    0.02488   0.007993         23        320:   7%|7         | 1/14 [00:00<00:01,  8.84it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.06174    0.02575    0.01151         27        320:   7%|7         | 1/14 [00:00<00:01,  8.84it/s]
          14/29     0.631G    0.06174    0.02575    0.01151         27        320:  14%|#4        | 2/14 [00:00<00:01,  9.43it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05922    0.02535    0.01061         24        320:  14%|#4        | 2/14 [00:00<00:01,  9.43it/s]
          14/29     0.631G    0.05922    0.02535    0.01061         24        320:  21%|##1       | 3/14 [00:00<00:01,  9.12it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05896    0.02414     0.0094         20        320:  21%|##1       | 3/14 [00:00<00:01,  9.12it/s]
          14/29     0.631G    0.05896    0.02414     0.0094         20        320:  29%|##8       | 4/14 [00:00<00:01,  8.83it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05947    0.02502   0.008816         28        320:  29%|##8       | 4/14 [00:00<00:01,  8.83it/s]
          14/29     0.631G    0.05947    0.02502   0.008816         28        320:  36%|###5      | 5/14 [00:00<00:01,  8.57it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05977    0.02432    0.00973         19        320:  36%|###5      | 5/14 [00:00<00:01,  8.57it/s]
          14/29     0.631G    0.05977    0.02432    0.00973         19        320:  43%|####2     | 6/14 [00:00<00:00,  8.60it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05917    0.02345   0.009763         18        320:  43%|####2     | 6/14 [00:00<00:00,  8.60it/s]
          14/29     0.631G    0.05917    0.02345   0.009763         18        320:  50%|#####     | 7/14 [00:00<00:00,  7.68it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05872     0.0242    0.01012         28        320:  50%|#####     | 7/14 [00:00<00:00,  7.68it/s]
          14/29     0.631G    0.05872     0.0242    0.01012         28        320:  57%|#####7    | 8/14 [00:00<00:00,  7.92it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05836    0.02421    0.01144         21        320:  57%|#####7    | 8/14 [00:01<00:00,  7.92it/s]
          14/29     0.631G    0.05836    0.02421    0.01144         21        320:  64%|######4   | 9/14 [00:01<00:00,  8.08it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05809      0.024    0.01194         23        320:  64%|######4   | 9/14 [00:01<00:00,  8.08it/s]
          14/29     0.631G    0.05809      0.024    0.01194         23        320:  71%|#######1  | 10/14 [00:01<00:00,  8.17it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05763    0.02366    0.01157         19        320:  71%|#######1  | 10/14 [00:01<00:00,  8.17it/s]
          14/29     0.631G    0.05763    0.02366    0.01157         19        320:  79%|#######8  | 11/14 [00:01<00:00,  8.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05819     0.0238    0.01204         23        320:  79%|#######8  | 11/14 [00:01<00:00,  8.27it/s]
          14/29     0.631G    0.05819     0.0238    0.01204         23        320:  86%|########5 | 12/14 [00:01<00:00,  8.31it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05828    0.02315    0.01178         16        320:  86%|########5 | 12/14 [00:01<00:00,  8.31it/s]
          14/29     0.631G    0.05828    0.02315    0.01178         16        320:  93%|#########2| 13/14 [00:01<00:00,  8.23it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          14/29     0.631G    0.05799     0.0238     0.0117         14        320:  93%|#########2| 13/14 [00:01<00:00,  8.23it/s]
          14/29     0.631G    0.05799     0.0238     0.0117         14        320: 100%|##########| 14/14 [00:01<00:00,  8.29it/s]
          14/29     0.631G    0.05799     0.0238     0.0117         14        320: 100%|##########| 14/14 [00:01<00:00,  8.34it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.65it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.68it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.04it/s]
                       all         35         38      0.232      0.342      0.233     0.0966
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.06459    0.02582   0.008862         28        320:   0%|          | 0/14 [00:00<?, ?it/s]
          15/29     0.631G    0.06459    0.02582   0.008862         28        320:   7%|7         | 1/14 [00:00<00:01,  7.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.06137    0.02697    0.01103         27        320:   7%|7         | 1/14 [00:00<00:01,  7.11it/s]
          15/29     0.631G    0.06137    0.02697    0.01103         27        320:  14%|#4        | 2/14 [00:00<00:01,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.06015     0.0283    0.01244         29        320:  14%|#4        | 2/14 [00:00<00:01,  8.47it/s]
          15/29     0.631G    0.06015     0.0283    0.01244         29        320:  21%|##1       | 3/14 [00:00<00:01,  8.85it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05848    0.02686    0.01159         20        320:  21%|##1       | 3/14 [00:00<00:01,  8.85it/s]
          15/29     0.631G    0.05848    0.02686    0.01159         20        320:  29%|##8       | 4/14 [00:00<00:01,  9.07it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05776    0.02602    0.01249         24        320:  29%|##8       | 4/14 [00:00<00:01,  9.07it/s]
          15/29     0.631G    0.05776    0.02602    0.01249         24        320:  36%|###5      | 5/14 [00:00<00:00,  9.15it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05831     0.0257     0.0142         25        320:  36%|###5      | 5/14 [00:00<00:00,  9.15it/s]
          15/29     0.631G    0.05831     0.0257     0.0142         25        320:  43%|####2     | 6/14 [00:00<00:00,  9.21it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05808    0.02525    0.01328         20        320:  43%|####2     | 6/14 [00:00<00:00,  9.21it/s]
          15/29     0.631G    0.05808    0.02525    0.01328         20        320:  50%|#####     | 7/14 [00:00<00:00,  9.15it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05863    0.02499    0.01307         23        320:  50%|#####     | 7/14 [00:00<00:00,  9.15it/s]
          15/29     0.631G    0.05863    0.02499    0.01307         23        320:  57%|#####7    | 8/14 [00:00<00:00,  9.29it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05962    0.02446    0.01246         18        320:  57%|#####7    | 8/14 [00:01<00:00,  9.29it/s]
          15/29     0.631G    0.05962    0.02446    0.01246         18        320:  64%|######4   | 9/14 [00:01<00:00,  8.57it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05895    0.02403      0.012         19        320:  64%|######4   | 9/14 [00:01<00:00,  8.57it/s]
          15/29     0.631G    0.05895    0.02403      0.012         19        320:  71%|#######1  | 10/14 [00:01<00:00,  8.94it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05887    0.02415    0.01139         27        320:  71%|#######1  | 10/14 [00:01<00:00,  8.94it/s]
          15/29     0.631G    0.05887    0.02415    0.01139         27        320:  79%|#######8  | 11/14 [00:01<00:00,  9.00it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05874    0.02408    0.01093         23        320:  79%|#######8  | 11/14 [00:01<00:00,  9.00it/s]
          15/29     0.631G    0.05874    0.02408    0.01093         23        320:  86%|########5 | 12/14 [00:01<00:00,  9.19it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05838    0.02407    0.01116         23        320:  86%|########5 | 12/14 [00:01<00:00,  9.19it/s]
          15/29     0.631G    0.05838    0.02407    0.01116         23        320:  93%|#########2| 13/14 [00:01<00:00,  9.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          15/29     0.631G    0.05785    0.02366    0.01088          8        320:  93%|#########2| 13/14 [00:01<00:00,  9.18it/s]
          15/29     0.631G    0.05785    0.02366    0.01088          8        320: 100%|##########| 14/14 [00:01<00:00,  9.19it/s]
          15/29     0.631G    0.05785    0.02366    0.01088          8        320: 100%|##########| 14/14 [00:01<00:00,  9.01it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.93it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.69it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.02it/s]
                       all         35         38       0.42      0.238      0.257      0.132
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05888    0.02608   0.009274         25        320:   0%|          | 0/14 [00:00<?, ?it/s]
          16/29     0.631G    0.05888    0.02608   0.009274         25        320:   7%|7         | 1/14 [00:00<00:01,  9.41it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05603    0.02393   0.008379         21        320:   7%|7         | 1/14 [00:00<00:01,  9.41it/s]
          16/29     0.631G    0.05603    0.02393   0.008379         21        320:  14%|#4        | 2/14 [00:00<00:01,  9.16it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G     0.0583    0.02433     0.0094         26        320:  14%|#4        | 2/14 [00:00<00:01,  9.16it/s]
          16/29     0.631G     0.0583    0.02433     0.0094         26        320:  21%|##1       | 3/14 [00:00<00:01,  8.14it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05616    0.02339   0.008582         20        320:  21%|##1       | 3/14 [00:00<00:01,  8.14it/s]
          16/29     0.631G    0.05616    0.02339   0.008582         20        320:  29%|##8       | 4/14 [00:00<00:01,  8.72it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05462    0.02321   0.008751         21        320:  29%|##8       | 4/14 [00:00<00:01,  8.72it/s]
          16/29     0.631G    0.05462    0.02321   0.008751         21        320:  36%|###5      | 5/14 [00:00<00:01,  8.84it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05349     0.0235   0.008567         26        320:  36%|###5      | 5/14 [00:00<00:01,  8.84it/s]
          16/29     0.631G    0.05349     0.0235   0.008567         26        320:  43%|####2     | 6/14 [00:00<00:00,  9.04it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05391    0.02315   0.009108         20        320:  43%|####2     | 6/14 [00:00<00:00,  9.04it/s]
          16/29     0.631G    0.05391    0.02315   0.009108         20        320:  50%|#####     | 7/14 [00:00<00:00,  9.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05418    0.02274   0.009389         18        320:  50%|#####     | 7/14 [00:00<00:00,  9.18it/s]
          16/29     0.631G    0.05418    0.02274   0.009389         18        320:  57%|#####7    | 8/14 [00:00<00:00,  9.24it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05477    0.02293   0.009946         25        320:  57%|#####7    | 8/14 [00:00<00:00,  9.24it/s]
          16/29     0.631G    0.05477    0.02293   0.009946         25        320:  64%|######4   | 9/14 [00:00<00:00,  9.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05439    0.02303     0.0113         24        320:  64%|######4   | 9/14 [00:01<00:00,  9.27it/s]
          16/29     0.631G    0.05439    0.02303     0.0113         24        320:  71%|#######1  | 10/14 [00:01<00:00,  9.26it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05443    0.02312      0.011         23        320:  71%|#######1  | 10/14 [00:01<00:00,  9.26it/s]
          16/29     0.631G    0.05443    0.02312      0.011         23        320:  79%|#######8  | 11/14 [00:01<00:00,  8.54it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05437    0.02307    0.01084         22        320:  79%|#######8  | 11/14 [00:01<00:00,  8.54it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G    0.05393    0.02242    0.01069         13        320:  79%|#######8  | 11/14 [00:01<00:00,  8.54it/s]
          16/29     0.631G    0.05393    0.02242    0.01069         13        320:  93%|#########2| 13/14 [00:01<00:00,  9.07it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          16/29     0.631G     0.0538    0.02256    0.01065         11        320:  93%|#########2| 13/14 [00:01<00:00,  9.07it/s]
          16/29     0.631G     0.0538    0.02256    0.01065         11        320: 100%|##########| 14/14 [00:01<00:00,  9.19it/s]
          16/29     0.631G     0.0538    0.02256    0.01065         11        320: 100%|##########| 14/14 [00:01<00:00,  9.04it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.86it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.84it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.12it/s]
                       all         35         38      0.294      0.237      0.272      0.118
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05286     0.0217   0.006118         21        320:   0%|          | 0/14 [00:00<?, ?it/s]
          17/29     0.631G    0.05286     0.0217   0.006118         21        320:   7%|7         | 1/14 [00:00<00:01,  9.30it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05398    0.02224   0.007751         21        320:   7%|7         | 1/14 [00:00<00:01,  9.30it/s]
          17/29     0.631G    0.05398    0.02224   0.007751         21        320:  14%|#4        | 2/14 [00:00<00:01,  9.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05409     0.0218   0.007392         18        320:  14%|#4        | 2/14 [00:00<00:01,  9.37it/s]
          17/29     0.631G    0.05409     0.0218   0.007392         18        320:  21%|##1       | 3/14 [00:00<00:01,  9.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05521    0.02308   0.008191         27        320:  21%|##1       | 3/14 [00:00<00:01,  9.38it/s]
          17/29     0.631G    0.05521    0.02308   0.008191         27        320:  29%|##8       | 4/14 [00:00<00:01,  9.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05673    0.02304   0.007515         25        320:  29%|##8       | 4/14 [00:00<00:01,  9.38it/s]
          17/29     0.631G    0.05673    0.02304   0.007515         25        320:  36%|###5      | 5/14 [00:00<00:01,  8.42it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05668    0.02299   0.008473         23        320:  36%|###5      | 5/14 [00:00<00:01,  8.42it/s]
          17/29     0.631G    0.05668    0.02299   0.008473         23        320:  43%|####2     | 6/14 [00:00<00:00,  8.84it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05699     0.0221   0.009803         16        320:  43%|####2     | 6/14 [00:00<00:00,  8.84it/s]
          17/29     0.631G    0.05699     0.0221   0.009803         16        320:  50%|#####     | 7/14 [00:00<00:00,  9.00it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G     0.0548    0.02222   0.009517         21        320:  50%|#####     | 7/14 [00:00<00:00,  9.00it/s]
          17/29     0.631G     0.0548    0.02222   0.009517         21        320:  57%|#####7    | 8/14 [00:00<00:00,  9.13it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05437    0.02226   0.009502         24        320:  57%|#####7    | 8/14 [00:00<00:00,  9.13it/s]
          17/29     0.631G    0.05437    0.02226   0.009502         24        320:  64%|######4   | 9/14 [00:00<00:00,  9.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05351    0.02189    0.00949         18        320:  64%|######4   | 9/14 [00:01<00:00,  9.27it/s]
          17/29     0.631G    0.05351    0.02189    0.00949         18        320:  71%|#######1  | 10/14 [00:01<00:00,  9.30it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05346    0.02247   0.009734         28        320:  71%|#######1  | 10/14 [00:01<00:00,  9.30it/s]
          17/29     0.631G    0.05346    0.02247   0.009734         28        320:  79%|#######8  | 11/14 [00:01<00:00,  9.28it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05366    0.02281    0.01008         29        320:  79%|#######8  | 11/14 [00:01<00:00,  9.28it/s]
          17/29     0.631G    0.05366    0.02281    0.01008         29        320:  86%|########5 | 12/14 [00:01<00:00,  9.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05409    0.02319    0.01037         30        320:  86%|########5 | 12/14 [00:01<00:00,  9.37it/s]
          17/29     0.631G    0.05409    0.02319    0.01037         30        320:  93%|#########2| 13/14 [00:01<00:00,  8.55it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          17/29     0.631G    0.05425    0.02338    0.01001         12        320:  93%|#########2| 13/14 [00:01<00:00,  8.55it/s]
          17/29     0.631G    0.05425    0.02338    0.01001         12        320: 100%|##########| 14/14 [00:01<00:00,  8.88it/s]
          17/29     0.631G    0.05425    0.02338    0.01001         12        320: 100%|##########| 14/14 [00:01<00:00,  9.04it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.48it/s]
                       all         35         38      0.257      0.632       0.33      0.134
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05972    0.02132   0.007295         22        320:   0%|          | 0/14 [00:00<?, ?it/s]
          18/29     0.631G    0.05972    0.02132   0.007295         22        320:   7%|7         | 1/14 [00:00<00:01,  9.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05652    0.02428   0.008851         29        320:   7%|7         | 1/14 [00:00<00:01,  9.37it/s]
          18/29     0.631G    0.05652    0.02428   0.008851         29        320:  14%|#4        | 2/14 [00:00<00:01,  9.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05531    0.02354   0.007974         23        320:  14%|#4        | 2/14 [00:00<00:01,  9.38it/s]
          18/29     0.631G    0.05531    0.02354   0.007974         23        320:  21%|##1       | 3/14 [00:00<00:01,  9.19it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05511    0.02473    0.00821         28        320:  21%|##1       | 3/14 [00:00<00:01,  9.19it/s]
          18/29     0.631G    0.05511    0.02473    0.00821         28        320:  29%|##8       | 4/14 [00:00<00:01,  9.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05401    0.02387   0.007444         22        320:  29%|##8       | 4/14 [00:00<00:01,  9.18it/s]
          18/29     0.631G    0.05401    0.02387   0.007444         22        320:  36%|###5      | 5/14 [00:00<00:00,  9.28it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05496    0.02307   0.007748         22        320:  36%|###5      | 5/14 [00:00<00:00,  9.28it/s]
          18/29     0.631G    0.05496    0.02307   0.007748         22        320:  43%|####2     | 6/14 [00:00<00:00,  9.29it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05495    0.02239   0.008201         17        320:  43%|####2     | 6/14 [00:00<00:00,  9.29it/s]
          18/29     0.631G    0.05495    0.02239   0.008201         17        320:  50%|#####     | 7/14 [00:00<00:00,  8.52it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05484    0.02361   0.008966         30        320:  50%|#####     | 7/14 [00:00<00:00,  8.52it/s]
          18/29     0.631G    0.05484    0.02361   0.008966         30        320:  57%|#####7    | 8/14 [00:00<00:00,  8.87it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05453     0.0224   0.008381         14        320:  57%|#####7    | 8/14 [00:00<00:00,  8.87it/s]
          18/29     0.631G    0.05453     0.0224   0.008381         14        320:  64%|######4   | 9/14 [00:00<00:00,  9.04it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05489    0.02296   0.008011         26        320:  64%|######4   | 9/14 [00:01<00:00,  9.04it/s]
          18/29     0.631G    0.05489    0.02296   0.008011         26        320:  71%|#######1  | 10/14 [00:01<00:00,  9.03it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05481    0.02345   0.007984         32        320:  71%|#######1  | 10/14 [00:01<00:00,  9.03it/s]
          18/29     0.631G    0.05481    0.02345   0.007984         32        320:  79%|#######8  | 11/14 [00:01<00:00,  9.17it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05481    0.02382     0.0078         31        320:  79%|#######8  | 11/14 [00:01<00:00,  9.17it/s]
          18/29     0.631G    0.05481    0.02382     0.0078         31        320:  86%|########5 | 12/14 [00:01<00:00,  9.35it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G    0.05491    0.02393   0.007994         30        320:  86%|########5 | 12/14 [00:01<00:00,  9.35it/s]
          18/29     0.631G    0.05491    0.02393   0.007994         30        320:  93%|#########2| 13/14 [00:01<00:00,  9.28it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          18/29     0.631G     0.0544    0.02378   0.007831         10        320:  93%|#########2| 13/14 [00:01<00:00,  9.28it/s]
          18/29     0.631G     0.0544    0.02378   0.007831         10        320: 100%|##########| 14/14 [00:01<00:00,  9.35it/s]
          18/29     0.631G     0.0544    0.02378   0.007831         10        320: 100%|##########| 14/14 [00:01<00:00,  9.17it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.05it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.99it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.38it/s]
                       all         35         38      0.268      0.642      0.324      0.149
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05687    0.01744   0.008952         16        320:   0%|          | 0/14 [00:00<?, ?it/s]
          19/29     0.631G    0.05687    0.01744   0.008952         16        320:   7%|7         | 1/14 [00:00<00:01,  7.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05699    0.02028   0.009427         26        320:   7%|7         | 1/14 [00:00<00:01,  7.47it/s]
          19/29     0.631G    0.05699    0.02028   0.009427         26        320:  14%|#4        | 2/14 [00:00<00:01,  8.65it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05425     0.0194      0.011         18        320:  14%|#4        | 2/14 [00:00<00:01,  8.65it/s]
          19/29     0.631G    0.05425     0.0194      0.011         18        320:  21%|##1       | 3/14 [00:00<00:01,  8.90it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05308    0.02056   0.009396         25        320:  21%|##1       | 3/14 [00:00<00:01,  8.90it/s]
          19/29     0.631G    0.05308    0.02056   0.009396         25        320:  29%|##8       | 4/14 [00:00<00:01,  9.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05172    0.02076   0.008491         19        320:  29%|##8       | 4/14 [00:00<00:01,  9.11it/s]
          19/29     0.631G    0.05172    0.02076   0.008491         19        320:  36%|###5      | 5/14 [00:00<00:00,  9.02it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05185    0.02071   0.008936         21        320:  36%|###5      | 5/14 [00:00<00:00,  9.02it/s]
          19/29     0.631G    0.05185    0.02071   0.008936         21        320:  43%|####2     | 6/14 [00:00<00:00,  8.99it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05154    0.02104   0.008728         21        320:  43%|####2     | 6/14 [00:00<00:00,  8.99it/s]
          19/29     0.631G    0.05154    0.02104   0.008728         21        320:  50%|#####     | 7/14 [00:00<00:00,  9.14it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G     0.0511    0.02051   0.008356         17        320:  50%|#####     | 7/14 [00:00<00:00,  9.14it/s]
          19/29     0.631G     0.0511    0.02051   0.008356         17        320:  57%|#####7    | 8/14 [00:00<00:00,  9.25it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05133    0.02112   0.008351         31        320:  57%|#####7    | 8/14 [00:01<00:00,  9.25it/s]
          19/29     0.631G    0.05133    0.02112   0.008351         31        320:  64%|######4   | 9/14 [00:01<00:00,  8.53it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05187    0.02085   0.008723         19        320:  64%|######4   | 9/14 [00:01<00:00,  8.53it/s]
          19/29     0.631G    0.05187    0.02085   0.008723         19        320:  71%|#######1  | 10/14 [00:01<00:00,  8.83it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05191     0.0207   0.008909         20        320:  71%|#######1  | 10/14 [00:01<00:00,  8.83it/s]
          19/29     0.631G    0.05191     0.0207   0.008909         20        320:  79%|#######8  | 11/14 [00:01<00:00,  9.01it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05112    0.02065   0.009136         20        320:  79%|#######8  | 11/14 [00:01<00:00,  9.01it/s]
          19/29     0.631G    0.05112    0.02065   0.009136         20        320:  86%|########5 | 12/14 [00:01<00:00,  9.28it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05122    0.02063   0.009939         22        320:  86%|########5 | 12/14 [00:01<00:00,  9.28it/s]
          19/29     0.631G    0.05122    0.02063   0.009939         22        320:  93%|#########2| 13/14 [00:01<00:00,  9.36it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          19/29     0.631G    0.05148    0.02079    0.01016         12        320:  93%|#########2| 13/14 [00:01<00:00,  9.36it/s]
          19/29     0.631G    0.05148    0.02079    0.01016         12        320: 100%|##########| 14/14 [00:01<00:00,  9.41it/s]
          19/29     0.631G    0.05148    0.02079    0.01016         12        320: 100%|##########| 14/14 [00:01<00:00,  9.07it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.81it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.86it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.26it/s]
                       all         35         38      0.298      0.579      0.351      0.148
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04857    0.02237   0.009941         24        320:   0%|          | 0/14 [00:00<?, ?it/s]
          20/29     0.631G    0.04857    0.02237   0.009941         24        320:   7%|7         | 1/14 [00:00<00:01,  9.67it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.05035     0.0196   0.008955         18        320:   7%|7         | 1/14 [00:00<00:01,  9.67it/s]
          20/29     0.631G    0.05035     0.0196   0.008955         18        320:  14%|#4        | 2/14 [00:00<00:01,  9.52it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04914    0.02035    0.01137         20        320:  14%|#4        | 2/14 [00:00<00:01,  9.52it/s]
          20/29     0.631G    0.04914    0.02035    0.01137         20        320:  21%|##1       | 3/14 [00:00<00:01,  7.93it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04834    0.02021    0.01079         18        320:  21%|##1       | 3/14 [00:00<00:01,  7.93it/s]
          20/29     0.631G    0.04834    0.02021    0.01079         18        320:  29%|##8       | 4/14 [00:00<00:01,  8.22it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04848    0.01887   0.009486         12        320:  29%|##8       | 4/14 [00:00<00:01,  8.22it/s]
          20/29     0.631G    0.04848    0.01887   0.009486         12        320:  36%|###5      | 5/14 [00:00<00:01,  8.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04836    0.01936   0.009944         24        320:  36%|###5      | 5/14 [00:00<00:01,  8.18it/s]
          20/29     0.631G    0.04836    0.01936   0.009944         24        320:  43%|####2     | 6/14 [00:00<00:00,  8.20it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04806    0.01968    0.01056         20        320:  43%|####2     | 6/14 [00:00<00:00,  8.20it/s]
          20/29     0.631G    0.04806    0.01968    0.01056         20        320:  50%|#####     | 7/14 [00:00<00:00,  8.35it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04855    0.02071    0.01054         31        320:  50%|#####     | 7/14 [00:00<00:00,  8.35it/s]
          20/29     0.631G    0.04855    0.02071    0.01054         31        320:  57%|#####7    | 8/14 [00:00<00:00,  8.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G      0.049    0.02078   0.009829         21        320:  57%|#####7    | 8/14 [00:01<00:00,  8.38it/s]
          20/29     0.631G      0.049    0.02078   0.009829         21        320:  64%|######4   | 9/14 [00:01<00:00,  8.43it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G     0.0487    0.02074   0.009816         21        320:  64%|######4   | 9/14 [00:01<00:00,  8.43it/s]
          20/29     0.631G     0.0487    0.02074   0.009816         21        320:  71%|#######1  | 10/14 [00:01<00:00,  8.56it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04874    0.02077   0.009696         21        320:  71%|#######1  | 10/14 [00:01<00:00,  8.56it/s]
          20/29     0.631G    0.04874    0.02077   0.009696         21        320:  79%|#######8  | 11/14 [00:01<00:00,  7.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.04957    0.02079   0.009719         20        320:  79%|#######8  | 11/14 [00:01<00:00,  7.27it/s]
          20/29     0.631G    0.04957    0.02079   0.009719         20        320:  86%|########5 | 12/14 [00:01<00:00,  7.78it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.631G    0.05038     0.0206   0.009408         21        320:  86%|########5 | 12/14 [00:01<00:00,  7.78it/s]
          20/29     0.631G    0.05038     0.0206   0.009408         21        320:  93%|#########2| 13/14 [00:01<00:00,  7.95it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          20/29     0.633G    0.05062    0.02099    0.00914         13        320:  93%|#########2| 13/14 [00:01<00:00,  7.95it/s]
          20/29     0.633G    0.05062    0.02099    0.00914         13        320: 100%|##########| 14/14 [00:01<00:00,  7.94it/s]
          20/29     0.633G    0.05062    0.02099    0.00914         13        320: 100%|##########| 14/14 [00:01<00:00,  8.13it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.58it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.68it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.97it/s]
                       all         35         38      0.367       0.61      0.403      0.204
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05479    0.02306   0.008089         24        320:   0%|          | 0/14 [00:00<?, ?it/s]
          21/29     0.633G    0.05479    0.02306   0.008089         24        320:   7%|7         | 1/14 [00:00<00:01,  9.53it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05185    0.02145   0.007996         24        320:   7%|7         | 1/14 [00:00<00:01,  9.53it/s]
          21/29     0.633G    0.05185    0.02145   0.007996         24        320:  14%|#4        | 2/14 [00:00<00:01,  9.46it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05504    0.02236   0.009764         23        320:  14%|#4        | 2/14 [00:00<00:01,  9.46it/s]
          21/29     0.633G    0.05504    0.02236   0.009764         23        320:  21%|##1       | 3/14 [00:00<00:01,  8.99it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05314    0.02162   0.009774         20        320:  21%|##1       | 3/14 [00:00<00:01,  8.99it/s]
          21/29     0.633G    0.05314    0.02162   0.009774         20        320:  29%|##8       | 4/14 [00:00<00:01,  9.23it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G     0.0532    0.02275    0.01094         26        320:  29%|##8       | 4/14 [00:00<00:01,  9.23it/s]
          21/29     0.633G     0.0532    0.02275    0.01094         26        320:  36%|###5      | 5/14 [00:00<00:01,  8.39it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05196    0.02205    0.01035         21        320:  36%|###5      | 5/14 [00:00<00:01,  8.39it/s]
          21/29     0.633G    0.05196    0.02205    0.01035         21        320:  43%|####2     | 6/14 [00:00<00:00,  8.78it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05162    0.02171     0.0102         21        320:  43%|####2     | 6/14 [00:00<00:00,  8.78it/s]
          21/29     0.633G    0.05162    0.02171     0.0102         21        320:  50%|#####     | 7/14 [00:00<00:00,  8.98it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05151    0.02193    0.01113         23        320:  50%|#####     | 7/14 [00:00<00:00,  8.98it/s]
          21/29     0.633G    0.05151    0.02193    0.01113         23        320:  57%|#####7    | 8/14 [00:00<00:00,  9.07it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05213    0.02142    0.01082         20        320:  57%|#####7    | 8/14 [00:00<00:00,  9.07it/s]
          21/29     0.633G    0.05213    0.02142    0.01082         20        320:  64%|######4   | 9/14 [00:00<00:00,  9.15it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05191     0.0209    0.01021         15        320:  64%|######4   | 9/14 [00:01<00:00,  9.15it/s]
          21/29     0.633G    0.05191     0.0209    0.01021         15        320:  71%|#######1  | 10/14 [00:01<00:00,  9.24it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05218    0.02162    0.01016         33        320:  71%|#######1  | 10/14 [00:01<00:00,  9.24it/s]
          21/29     0.633G    0.05218    0.02162    0.01016         33        320:  79%|#######8  | 11/14 [00:01<00:00,  9.31it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05203    0.02161   0.009718         26        320:  79%|#######8  | 11/14 [00:01<00:00,  9.31it/s]
          21/29     0.633G    0.05203    0.02161   0.009718         26        320:  86%|########5 | 12/14 [00:01<00:00,  9.26it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05187    0.02194   0.009835         27        320:  86%|########5 | 12/14 [00:01<00:00,  9.26it/s]
          21/29     0.633G    0.05187    0.02194   0.009835         27        320:  93%|#########2| 13/14 [00:01<00:00,  8.52it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          21/29     0.633G    0.05173    0.02188   0.009332         11        320:  93%|#########2| 13/14 [00:01<00:00,  8.52it/s]
          21/29     0.633G    0.05173    0.02188   0.009332         11        320: 100%|##########| 14/14 [00:01<00:00,  8.83it/s]
          21/29     0.633G    0.05173    0.02188   0.009332         11        320: 100%|##########| 14/14 [00:01<00:00,  8.98it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.94it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.23it/s]
                       all         35         38      0.426      0.571      0.426      0.232
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G     0.0484    0.01952   0.005734         22        320:   0%|          | 0/14 [00:00<?, ?it/s]
          22/29     0.633G     0.0484    0.01952   0.005734         22        320:   7%|7         | 1/14 [00:00<00:01,  9.44it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04679    0.01909   0.005029         21        320:   7%|7         | 1/14 [00:00<00:01,  9.44it/s]
          22/29     0.633G    0.04679    0.01909   0.005029         21        320:  14%|#4        | 2/14 [00:00<00:01,  9.44it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04885     0.0204   0.005556         25        320:  14%|#4        | 2/14 [00:00<00:01,  9.44it/s]
          22/29     0.633G    0.04885     0.0204   0.005556         25        320:  21%|##1       | 3/14 [00:00<00:01,  9.39it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04657    0.01896    0.00594         15        320:  21%|##1       | 3/14 [00:00<00:01,  9.39it/s]
          22/29     0.633G    0.04657    0.01896    0.00594         15        320:  29%|##8       | 4/14 [00:00<00:01,  9.45it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04636     0.0192   0.007788         22        320:  29%|##8       | 4/14 [00:00<00:01,  9.45it/s]
          22/29     0.633G    0.04636     0.0192   0.007788         22        320:  36%|###5      | 5/14 [00:00<00:00,  9.40it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04611    0.01865    0.00768         18        320:  36%|###5      | 5/14 [00:00<00:00,  9.40it/s]
          22/29     0.633G    0.04611    0.01865    0.00768         18        320:  43%|####2     | 6/14 [00:00<00:00,  9.34it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04575    0.01899   0.007251         23        320:  43%|####2     | 6/14 [00:00<00:00,  9.34it/s]
          22/29     0.633G    0.04575    0.01899   0.007251         23        320:  50%|#####     | 7/14 [00:00<00:00,  8.48it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04642    0.01933   0.007276         22        320:  50%|#####     | 7/14 [00:00<00:00,  8.48it/s]
          22/29     0.633G    0.04642    0.01933   0.007276         22        320:  57%|#####7    | 8/14 [00:00<00:00,  8.91it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04647    0.01942   0.006982         24        320:  57%|#####7    | 8/14 [00:00<00:00,  8.91it/s]
          22/29     0.633G    0.04647    0.01942   0.006982         24        320:  64%|######4   | 9/14 [00:00<00:00,  9.10it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04636    0.01918   0.006627         17        320:  64%|######4   | 9/14 [00:01<00:00,  9.10it/s]
          22/29     0.633G    0.04636    0.01918   0.006627         17        320:  71%|#######1  | 10/14 [00:01<00:00,  9.20it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04566    0.01876   0.006213         15        320:  71%|#######1  | 10/14 [00:01<00:00,  9.20it/s]
          22/29     0.633G    0.04566    0.01876   0.006213         15        320:  79%|#######8  | 11/14 [00:01<00:00,  9.14it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04509     0.0188   0.006066         19        320:  79%|#######8  | 11/14 [00:01<00:00,  9.14it/s]
          22/29     0.633G    0.04509     0.0188   0.006066         19        320:  86%|########5 | 12/14 [00:01<00:00,  9.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04517    0.01898   0.005972         24        320:  86%|########5 | 12/14 [00:01<00:00,  9.37it/s]
          22/29     0.633G    0.04517    0.01898   0.005972         24        320:  93%|#########2| 13/14 [00:01<00:00,  9.39it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          22/29     0.633G    0.04543    0.01988   0.005897         19        320:  93%|#########2| 13/14 [00:01<00:00,  9.39it/s]
          22/29     0.633G    0.04543    0.01988   0.005897         19        320: 100%|##########| 14/14 [00:01<00:00,  9.47it/s]
          22/29     0.633G    0.04543    0.01988   0.005897         19        320: 100%|##########| 14/14 [00:01<00:00,  9.26it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.97it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.30it/s]
                       all         35         38      0.442      0.617      0.463      0.215
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.03735    0.01681   0.004068         18        320:   0%|          | 0/14 [00:00<?, ?it/s]
          23/29     0.633G    0.03735    0.01681   0.004068         18        320:   7%|7         | 1/14 [00:00<00:01,  7.11it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04232    0.01886   0.007987         22        320:   7%|7         | 1/14 [00:00<00:01,  7.11it/s]
          23/29     0.633G    0.04232    0.01886   0.007987         22        320:  14%|#4        | 2/14 [00:00<00:01,  8.35it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04706    0.01991   0.007592         21        320:  14%|#4        | 2/14 [00:00<00:01,  8.35it/s]
          23/29     0.633G    0.04706    0.01991   0.007592         21        320:  21%|##1       | 3/14 [00:00<00:01,  8.71it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04713    0.02018   0.006574         21        320:  21%|##1       | 3/14 [00:00<00:01,  8.71it/s]
          23/29     0.633G    0.04713    0.02018   0.006574         21        320:  29%|##8       | 4/14 [00:00<00:01,  9.02it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04677     0.0206   0.006313         24        320:  29%|##8       | 4/14 [00:00<00:01,  9.02it/s]
          23/29     0.633G    0.04677     0.0206   0.006313         24        320:  36%|###5      | 5/14 [00:00<00:00,  9.20it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04643    0.02042   0.006952         22        320:  36%|###5      | 5/14 [00:00<00:00,  9.20it/s]
          23/29     0.633G    0.04643    0.02042   0.006952         22        320:  43%|####2     | 6/14 [00:00<00:00,  9.21it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04832    0.02173   0.007232         30        320:  43%|####2     | 6/14 [00:00<00:00,  9.21it/s]
          23/29     0.633G    0.04832    0.02173   0.007232         30        320:  50%|#####     | 7/14 [00:00<00:00,  9.32it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04893    0.02285   0.007166         31        320:  50%|#####     | 7/14 [00:00<00:00,  9.32it/s]
          23/29     0.633G    0.04893    0.02285   0.007166         31        320:  57%|#####7    | 8/14 [00:00<00:00,  9.40it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04806    0.02223   0.006787         18        320:  57%|#####7    | 8/14 [00:01<00:00,  9.40it/s]
          23/29     0.633G    0.04806    0.02223   0.006787         18        320:  64%|######4   | 9/14 [00:01<00:00,  8.57it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04812    0.02193   0.006891         21        320:  64%|######4   | 9/14 [00:01<00:00,  8.57it/s]
          23/29     0.633G    0.04812    0.02193   0.006891         21        320:  71%|#######1  | 10/14 [00:01<00:00,  8.86it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04883    0.02181   0.006984         25        320:  71%|#######1  | 10/14 [00:01<00:00,  8.86it/s]
          23/29     0.633G    0.04883    0.02181   0.006984         25        320:  79%|#######8  | 11/14 [00:01<00:00,  9.03it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G     0.0484    0.02158   0.006947         21        320:  79%|#######8  | 11/14 [00:01<00:00,  9.03it/s]
          23/29     0.633G     0.0484    0.02158   0.006947         21        320:  86%|########5 | 12/14 [00:01<00:00,  9.26it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04886    0.02193   0.007466         28        320:  86%|########5 | 12/14 [00:01<00:00,  9.26it/s]
          23/29     0.633G    0.04886    0.02193   0.007466         28        320:  93%|#########2| 13/14 [00:01<00:00,  9.35it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          23/29     0.633G    0.04877    0.02193   0.007127         13        320:  93%|#########2| 13/14 [00:01<00:00,  9.35it/s]
          23/29     0.633G    0.04877    0.02193   0.007127         13        320: 100%|##########| 14/14 [00:01<00:00,  9.33it/s]
          23/29     0.633G    0.04877    0.02193   0.007127         13        320: 100%|##########| 14/14 [00:01<00:00,  9.06it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.02it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.90it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.27it/s]
                       all         35         38      0.446      0.625      0.472       0.25
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G      0.049      0.018   0.003116         18        320:   0%|          | 0/14 [00:00<?, ?it/s]
          24/29     0.633G      0.049      0.018   0.003116         18        320:   7%|7         | 1/14 [00:00<00:01,  9.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04508    0.01747   0.003527         17        320:   7%|7         | 1/14 [00:00<00:01,  9.47it/s]
          24/29     0.633G    0.04508    0.01747   0.003527         17        320:  14%|#4        | 2/14 [00:00<00:01,  9.29it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04646    0.02046   0.006308         25        320:  14%|#4        | 2/14 [00:00<00:01,  9.29it/s]
          24/29     0.633G    0.04646    0.02046   0.006308         25        320:  21%|##1       | 3/14 [00:00<00:01,  8.26it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04726    0.01959   0.007043         20        320:  21%|##1       | 3/14 [00:00<00:01,  8.26it/s]
          24/29     0.633G    0.04726    0.01959   0.007043         20        320:  29%|##8       | 4/14 [00:00<00:01,  8.74it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04619    0.02051   0.007264         26        320:  29%|##8       | 4/14 [00:00<00:01,  8.74it/s]
          24/29     0.633G    0.04619    0.02051   0.007264         26        320:  36%|###5      | 5/14 [00:00<00:01,  8.87it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04581     0.0202   0.006633         22        320:  36%|###5      | 5/14 [00:00<00:01,  8.87it/s]
          24/29     0.633G    0.04581     0.0202   0.006633         22        320:  43%|####2     | 6/14 [00:00<00:00,  8.80it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04711    0.02065    0.00679         26        320:  43%|####2     | 6/14 [00:00<00:00,  8.80it/s]
          24/29     0.633G    0.04711    0.02065    0.00679         26        320:  50%|#####     | 7/14 [00:00<00:00,  8.96it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04695    0.02054   0.006498         21        320:  50%|#####     | 7/14 [00:00<00:00,  8.96it/s]
          24/29     0.633G    0.04695    0.02054   0.006498         21        320:  57%|#####7    | 8/14 [00:00<00:00,  9.08it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04695    0.02058   0.006604         20        320:  57%|#####7    | 8/14 [00:01<00:00,  9.08it/s]
          24/29     0.633G    0.04695    0.02058   0.006604         20        320:  64%|######4   | 9/14 [00:01<00:00,  8.99it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04735      0.021   0.006803         24        320:  64%|######4   | 9/14 [00:01<00:00,  8.99it/s]
          24/29     0.633G    0.04735      0.021   0.006803         24        320:  71%|#######1  | 10/14 [00:01<00:00,  9.13it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04703    0.02068   0.006823         19        320:  71%|#######1  | 10/14 [00:01<00:00,  9.13it/s]
          24/29     0.633G    0.04703    0.02068   0.006823         19        320:  79%|#######8  | 11/14 [00:01<00:00,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G     0.0474    0.02148   0.006756         31        320:  79%|#######8  | 11/14 [00:01<00:00,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04723    0.02119   0.006832         17        320:  79%|#######8  | 11/14 [00:01<00:00,  8.47it/s]
          24/29     0.633G    0.04723    0.02119   0.006832         17        320:  93%|#########2| 13/14 [00:01<00:00,  8.97it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          24/29     0.633G    0.04736    0.02083   0.006971          8        320:  93%|#########2| 13/14 [00:01<00:00,  8.97it/s]
          24/29     0.633G    0.04736    0.02083   0.006971          8        320: 100%|##########| 14/14 [00:01<00:00,  9.07it/s]
          24/29     0.633G    0.04736    0.02083   0.006971          8        320: 100%|##########| 14/14 [00:01<00:00,  8.94it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.84it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.90it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.28it/s]
                       all         35         38      0.444      0.605      0.484      0.258
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.05604    0.01716   0.008194         17        320:   0%|          | 0/14 [00:00<?, ?it/s]
          25/29     0.633G    0.05604    0.01716   0.008194         17        320:   7%|7         | 1/14 [00:00<00:01,  9.72it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.05024    0.02048   0.007812         27        320:   7%|7         | 1/14 [00:00<00:01,  9.72it/s]
          25/29     0.633G    0.05024    0.02048   0.007812         27        320:  14%|#4        | 2/14 [00:00<00:01,  9.68it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.05402    0.02108   0.007029         25        320:  14%|#4        | 2/14 [00:00<00:01,  9.68it/s]
          25/29     0.633G    0.05402    0.02108   0.007029         25        320:  21%|##1       | 3/14 [00:00<00:01,  9.57it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04967    0.01933   0.006075         13        320:  21%|##1       | 3/14 [00:00<00:01,  9.57it/s]
          25/29     0.633G    0.04967    0.01933   0.006075         13        320:  29%|##8       | 4/14 [00:00<00:01,  9.53it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.05112    0.02097   0.006735         25        320:  29%|##8       | 4/14 [00:00<00:01,  9.53it/s]
          25/29     0.633G    0.05112    0.02097   0.006735         25        320:  36%|###5      | 5/14 [00:00<00:01,  8.59it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G      0.049    0.01998    0.00608         18        320:  36%|###5      | 5/14 [00:00<00:01,  8.59it/s]
          25/29     0.633G      0.049    0.01998    0.00608         18        320:  43%|####2     | 6/14 [00:00<00:00,  8.97it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04725    0.02036   0.006129         25        320:  43%|####2     | 6/14 [00:00<00:00,  8.97it/s]
          25/29     0.633G    0.04725    0.02036   0.006129         25        320:  50%|#####     | 7/14 [00:00<00:00,  9.15it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04647    0.02003   0.006049         20        320:  50%|#####     | 7/14 [00:00<00:00,  9.15it/s]
          25/29     0.633G    0.04647    0.02003   0.006049         20        320:  57%|#####7    | 8/14 [00:00<00:00,  9.26it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04693    0.01989   0.006106         23        320:  57%|#####7    | 8/14 [00:00<00:00,  9.26it/s]
          25/29     0.633G    0.04693    0.01989   0.006106         23        320:  64%|######4   | 9/14 [00:00<00:00,  9.33it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04729    0.02038   0.006204         26        320:  64%|######4   | 9/14 [00:01<00:00,  9.33it/s]
          25/29     0.633G    0.04729    0.02038   0.006204         26        320:  71%|#######1  | 10/14 [00:01<00:00,  9.38it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G     0.0468    0.02071   0.006646         28        320:  71%|#######1  | 10/14 [00:01<00:00,  9.38it/s]
          25/29     0.633G     0.0468    0.02071   0.006646         28        320:  79%|#######8  | 11/14 [00:01<00:00,  9.28it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04643    0.02025   0.006974         18        320:  79%|#######8  | 11/14 [00:01<00:00,  9.28it/s]
          25/29     0.633G    0.04643    0.02025   0.006974         18        320:  86%|########5 | 12/14 [00:01<00:00,  9.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04564    0.02001   0.006617         17        320:  86%|########5 | 12/14 [00:01<00:00,  9.27it/s]
          25/29     0.633G    0.04564    0.02001   0.006617         17        320:  93%|#########2| 13/14 [00:01<00:00,  8.56it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          25/29     0.633G    0.04629    0.02173    0.00748         21        320:  93%|#########2| 13/14 [00:01<00:00,  8.56it/s]
          25/29     0.633G    0.04629    0.02173    0.00748         21        320: 100%|##########| 14/14 [00:01<00:00,  8.87it/s]
          25/29     0.633G    0.04629    0.02173    0.00748         21        320: 100%|##########| 14/14 [00:01<00:00,  9.11it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.02it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.96it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.22it/s]
                       all         35         38      0.444       0.68       0.51      0.255
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G     0.0459    0.01856   0.004857         18        320:   0%|          | 0/14 [00:00<?, ?it/s]
          26/29     0.633G     0.0459    0.01856   0.004857         18        320:   7%|7         | 1/14 [00:00<00:01,  8.95it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04189    0.01926   0.004666         22        320:   7%|7         | 1/14 [00:00<00:01,  8.95it/s]
          26/29     0.633G    0.04189    0.01926   0.004666         22        320:  14%|#4        | 2/14 [00:00<00:01,  8.59it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04086    0.01955   0.004735         20        320:  14%|#4        | 2/14 [00:00<00:01,  8.59it/s]
          26/29     0.633G    0.04086    0.01955   0.004735         20        320:  21%|##1       | 3/14 [00:00<00:01,  8.47it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04278    0.01963   0.006052         20        320:  21%|##1       | 3/14 [00:00<00:01,  8.47it/s]
          26/29     0.633G    0.04278    0.01963   0.006052         20        320:  29%|##8       | 4/14 [00:00<00:01,  8.17it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04279    0.02047   0.005728         19        320:  29%|##8       | 4/14 [00:00<00:01,  8.17it/s]
          26/29     0.633G    0.04279    0.02047   0.005728         19        320:  36%|###5      | 5/14 [00:00<00:01,  7.89it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04479    0.02034   0.006095         24        320:  36%|###5      | 5/14 [00:00<00:01,  7.89it/s]
          26/29     0.633G    0.04479    0.02034   0.006095         24        320:  43%|####2     | 6/14 [00:00<00:01,  7.94it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04456     0.0211   0.006085         31        320:  43%|####2     | 6/14 [00:00<00:01,  7.94it/s]
          26/29     0.633G    0.04456     0.0211   0.006085         31        320:  50%|#####     | 7/14 [00:00<00:00,  7.37it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04434    0.02108   0.006186         24        320:  50%|#####     | 7/14 [00:01<00:00,  7.37it/s]
          26/29     0.633G    0.04434    0.02108   0.006186         24        320:  57%|#####7    | 8/14 [00:01<00:00,  7.49it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04445    0.02078   0.006846         23        320:  57%|#####7    | 8/14 [00:01<00:00,  7.49it/s]
          26/29     0.633G    0.04445    0.02078   0.006846         23        320:  64%|######4   | 9/14 [00:01<00:00,  7.66it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04553    0.02054    0.00668         22        320:  64%|######4   | 9/14 [00:01<00:00,  7.66it/s]
          26/29     0.633G    0.04553    0.02054    0.00668         22        320:  71%|#######1  | 10/14 [00:01<00:00,  7.81it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04543    0.02031   0.006452         18        320:  71%|#######1  | 10/14 [00:01<00:00,  7.81it/s]
          26/29     0.633G    0.04543    0.02031   0.006452         18        320:  79%|#######8  | 11/14 [00:01<00:00,  7.96it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04522    0.01977   0.006229         17        320:  79%|#######8  | 11/14 [00:01<00:00,  7.96it/s]
          26/29     0.633G    0.04522    0.01977   0.006229         17        320:  86%|########5 | 12/14 [00:01<00:00,  8.10it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04525    0.01978   0.005937         21        320:  86%|########5 | 12/14 [00:01<00:00,  8.10it/s]
          26/29     0.633G    0.04525    0.01978   0.005937         21        320:  93%|#########2| 13/14 [00:01<00:00,  8.07it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          26/29     0.633G    0.04515    0.02022   0.005911         16        320:  93%|#########2| 13/14 [00:01<00:00,  8.07it/s]
          26/29     0.633G    0.04515    0.02022   0.005911         16        320: 100%|##########| 14/14 [00:01<00:00,  8.25it/s]
          26/29     0.633G    0.04515    0.02022   0.005911         16        320: 100%|##########| 14/14 [00:01<00:00,  8.00it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.47it/s]
                       all         35         38      0.541      0.632      0.527      0.243
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.05165    0.01914   0.008714         21        320:   0%|          | 0/14 [00:00<?, ?it/s]
          27/29     0.633G    0.05165    0.01914   0.008714         21        320:   7%|7         | 1/14 [00:00<00:01,  6.85it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04451    0.01922   0.006492         20        320:   7%|7         | 1/14 [00:00<00:01,  6.85it/s]
          27/29     0.633G    0.04451    0.01922   0.006492         20        320:  14%|#4        | 2/14 [00:00<00:01,  8.23it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04693    0.01975   0.006959         22        320:  14%|#4        | 2/14 [00:00<00:01,  8.23it/s]
          27/29     0.633G    0.04693    0.01975   0.006959         22        320:  21%|##1       | 3/14 [00:00<00:01,  8.71it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04626    0.01916   0.006896         19        320:  21%|##1       | 3/14 [00:00<00:01,  8.71it/s]
          27/29     0.633G    0.04626    0.01916   0.006896         19        320:  29%|##8       | 4/14 [00:00<00:01,  9.00it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04526    0.01873   0.006432         19        320:  29%|##8       | 4/14 [00:00<00:01,  9.00it/s]
          27/29     0.633G    0.04526    0.01873   0.006432         19        320:  36%|###5      | 5/14 [00:00<00:01,  8.81it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04601    0.01884    0.00652         20        320:  36%|###5      | 5/14 [00:00<00:01,  8.81it/s]
          27/29     0.633G    0.04601    0.01884    0.00652         20        320:  43%|####2     | 6/14 [00:00<00:00,  8.90it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04614     0.0191   0.006454         25        320:  43%|####2     | 6/14 [00:00<00:00,  8.90it/s]
          27/29     0.633G    0.04614     0.0191   0.006454         25        320:  50%|#####     | 7/14 [00:00<00:00,  8.73it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G     0.0454    0.01879   0.006017         20        320:  50%|#####     | 7/14 [00:00<00:00,  8.73it/s]
          27/29     0.633G     0.0454    0.01879   0.006017         20        320:  57%|#####7    | 8/14 [00:00<00:00,  8.68it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04491    0.01846   0.006567         17        320:  57%|#####7    | 8/14 [00:01<00:00,  8.68it/s]
          27/29     0.633G    0.04491    0.01846   0.006567         17        320:  64%|######4   | 9/14 [00:01<00:00,  7.54it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04546     0.0189    0.00703         25        320:  64%|######4   | 9/14 [00:01<00:00,  7.54it/s]
          27/29     0.633G    0.04546     0.0189    0.00703         25        320:  71%|#######1  | 10/14 [00:01<00:00,  7.77it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04519     0.0192   0.007246         23        320:  71%|#######1  | 10/14 [00:01<00:00,  7.77it/s]
          27/29     0.633G    0.04519     0.0192   0.007246         23        320:  79%|#######8  | 11/14 [00:01<00:00,  7.78it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04551    0.01971   0.007657         28        320:  79%|#######8  | 11/14 [00:01<00:00,  7.78it/s]
          27/29     0.633G    0.04551    0.01971   0.007657         28        320:  86%|########5 | 12/14 [00:01<00:00,  7.77it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04563     0.0197   0.007818         20        320:  86%|########5 | 12/14 [00:01<00:00,  7.77it/s]
          27/29     0.633G    0.04563     0.0197   0.007818         20        320:  93%|#########2| 13/14 [00:01<00:00,  7.92it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          27/29     0.633G    0.04514    0.02034   0.008008         15        320:  93%|#########2| 13/14 [00:01<00:00,  7.92it/s]
          27/29     0.633G    0.04514    0.02034   0.008008         15        320: 100%|##########| 14/14 [00:01<00:00,  8.07it/s]
          27/29     0.633G    0.04514    0.02034   0.008008         15        320: 100%|##########| 14/14 [00:01<00:00,  8.17it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.66it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.80it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.16it/s]
                       all         35         38      0.519      0.675      0.579       0.26
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G     0.0419    0.02712   0.007633         24        320:   0%|          | 0/14 [00:00<?, ?it/s]
          28/29     0.633G     0.0419    0.02712   0.007633         24        320:   7%|7         | 1/14 [00:00<00:01,  9.80it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04483    0.02122   0.005066         19        320:   7%|7         | 1/14 [00:00<00:01,  9.80it/s]
          28/29     0.633G    0.04483    0.02122   0.005066         19        320:  14%|#4        | 2/14 [00:00<00:01,  9.02it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G     0.0446    0.02166   0.005438         24        320:  14%|#4        | 2/14 [00:00<00:01,  9.02it/s]
          28/29     0.633G     0.0446    0.02166   0.005438         24        320:  21%|##1       | 3/14 [00:00<00:01,  8.22it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04759    0.01987   0.005097         16        320:  21%|##1       | 3/14 [00:00<00:01,  8.22it/s]
          28/29     0.633G    0.04759    0.01987   0.005097         16        320:  29%|##8       | 4/14 [00:00<00:01,  8.75it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04722    0.01973   0.005133         22        320:  29%|##8       | 4/14 [00:00<00:01,  8.75it/s]
          28/29     0.633G    0.04722    0.01973   0.005133         22        320:  36%|###5      | 5/14 [00:00<00:01,  8.39it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04664    0.01948   0.005679         20        320:  36%|###5      | 5/14 [00:00<00:01,  8.39it/s]
          28/29     0.633G    0.04664    0.01948   0.005679         20        320:  43%|####2     | 6/14 [00:00<00:00,  8.53it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04546    0.01959     0.0056         23        320:  43%|####2     | 6/14 [00:00<00:00,  8.53it/s]
          28/29     0.633G    0.04546    0.01959     0.0056         23        320:  50%|#####     | 7/14 [00:00<00:00,  8.84it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G     0.0451    0.01954   0.005303         21        320:  50%|#####     | 7/14 [00:00<00:00,  8.84it/s]
          28/29     0.633G     0.0451    0.01954   0.005303         21        320:  57%|#####7    | 8/14 [00:00<00:00,  9.02it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04474    0.01959   0.005211         21        320:  57%|#####7    | 8/14 [00:01<00:00,  9.02it/s]
          28/29     0.633G    0.04474    0.01959   0.005211         21        320:  64%|######4   | 9/14 [00:01<00:00,  9.12it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04485    0.01959   0.005229         21        320:  64%|######4   | 9/14 [00:01<00:00,  9.12it/s]
          28/29     0.633G    0.04485    0.01959   0.005229         21        320:  71%|#######1  | 10/14 [00:01<00:00,  9.18it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04472    0.01994    0.00537         24        320:  71%|#######1  | 10/14 [00:01<00:00,  9.18it/s]
          28/29     0.633G    0.04472    0.01994    0.00537         24        320:  79%|#######8  | 11/14 [00:01<00:00,  8.49it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04418    0.01966   0.005803         19        320:  79%|#######8  | 11/14 [00:01<00:00,  8.49it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04402    0.01941   0.006114         18        320:  79%|#######8  | 11/14 [00:01<00:00,  8.49it/s]
          28/29     0.633G    0.04402    0.01941   0.006114         18        320:  93%|#########2| 13/14 [00:01<00:00,  9.00it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          28/29     0.633G    0.04475    0.01923   0.005973          9        320:  93%|#########2| 13/14 [00:01<00:00,  9.00it/s]
          28/29     0.633G    0.04475    0.01923   0.005973          9        320: 100%|##########| 14/14 [00:01<00:00,  8.91it/s]
          28/29     0.633G    0.04475    0.01923   0.005973          9        320: 100%|##########| 14/14 [00:01<00:00,  8.84it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  5.04it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.99it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  6.40it/s]
                       all         35         38      0.479      0.711      0.566      0.282
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    
      0%|          | 0/14 [00:00<?, ?it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04498    0.01733   0.004694         18        320:   0%|          | 0/14 [00:00<?, ?it/s]
          29/29     0.633G    0.04498    0.01733   0.004694         18        320:   7%|7         | 1/14 [00:00<00:01,  7.68it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04338    0.01746    0.00662         19        320:   7%|7         | 1/14 [00:00<00:01,  7.68it/s]
          29/29     0.633G    0.04338    0.01746    0.00662         19        320:  14%|#4        | 2/14 [00:00<00:01,  8.73it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04394    0.01703    0.00943         17        320:  14%|#4        | 2/14 [00:00<00:01,  8.73it/s]
          29/29     0.633G    0.04394    0.01703    0.00943         17        320:  21%|##1       | 3/14 [00:00<00:01,  8.81it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04345    0.01815   0.007937         25        320:  21%|##1       | 3/14 [00:00<00:01,  8.81it/s]
          29/29     0.633G    0.04345    0.01815   0.007937         25        320:  29%|##8       | 4/14 [00:00<00:01,  8.99it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04277    0.01807   0.007044         19        320:  29%|##8       | 4/14 [00:00<00:01,  8.99it/s]
          29/29     0.633G    0.04277    0.01807   0.007044         19        320:  36%|###5      | 5/14 [00:00<00:01,  8.27it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04378    0.01892   0.007549         28        320:  36%|###5      | 5/14 [00:00<00:01,  8.27it/s]
          29/29     0.633G    0.04378    0.01892   0.007549         28        320:  43%|####2     | 6/14 [00:00<00:00,  8.70it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04242    0.01816   0.007818         15        320:  43%|####2     | 6/14 [00:00<00:00,  8.70it/s]
          29/29     0.633G    0.04242    0.01816   0.007818         15        320:  50%|#####     | 7/14 [00:00<00:00,  8.92it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04235    0.01773   0.008158         16        320:  50%|#####     | 7/14 [00:00<00:00,  8.92it/s]
          29/29     0.633G    0.04235    0.01773   0.008158         16        320:  57%|#####7    | 8/14 [00:00<00:00,  9.10it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G     0.0411    0.01745   0.009001         16        320:  57%|#####7    | 8/14 [00:01<00:00,  9.10it/s]
          29/29     0.633G     0.0411    0.01745   0.009001         16        320:  64%|######4   | 9/14 [00:01<00:00,  9.12it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04032    0.01778   0.008369         23        320:  64%|######4   | 9/14 [00:01<00:00,  9.12it/s]
          29/29     0.633G    0.04032    0.01778   0.008369         23        320:  71%|#######1  | 10/14 [00:01<00:00,  9.02it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04026    0.01816   0.007981         29        320:  71%|#######1  | 10/14 [00:01<00:00,  9.02it/s]
          29/29     0.633G    0.04026    0.01816   0.007981         29        320:  79%|#######8  | 11/14 [00:01<00:00,  9.15it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G    0.04033    0.01806   0.007745         19        320:  79%|#######8  | 11/14 [00:01<00:00,  9.15it/s]
          29/29     0.633G    0.04033    0.01806   0.007745         19        320:  86%|########5 | 12/14 [00:01<00:00,  8.94it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G     0.0408    0.01826   0.007592         23        320:  86%|########5 | 12/14 [00:01<00:00,  8.94it/s]
          29/29     0.633G     0.0408    0.01826   0.007592         23        320:  93%|#########2| 13/14 [00:01<00:00,  7.93it/s]C:\Users\sunny\졸업작품\yolov5\train.py:412: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(amp):
    
          29/29     0.633G     0.0414    0.01818   0.007211         11        320:  93%|#########2| 13/14 [00:01<00:00,  7.93it/s]
          29/29     0.633G     0.0414    0.01818   0.007211         11        320: 100%|##########| 14/14 [00:01<00:00,  8.22it/s]
          29/29     0.633G     0.0414    0.01818   0.007211         11        320: 100%|##########| 14/14 [00:01<00:00,  8.62it/s]
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.45it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  4.47it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  5.68it/s]
                       all         35         38      0.525      0.711      0.602      0.298
    
    30 epochs completed in 0.023 hours.
    Optimizer stripped from runs\train\exp2\weights\last.pt, 14.3MB
    Optimizer stripped from runs\train\exp2\weights\best.pt, 14.3MB
    
    Validating runs\train\exp2\weights\best.pt...
    Fusing layers... 
    Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    
                     Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/3 [00:00<?, ?it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|###3      | 1/3 [00:00<00:00,  4.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|######6   | 2/3 [00:00<00:00,  3.43it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  4.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|##########| 3/3 [00:00<00:00,  4.06it/s]
                       all         35         38      0.524      0.711      0.602      0.297
                clear-road         35         19      0.686      0.474      0.625      0.291
                 iced-road         35         19      0.362      0.947       0.58      0.303
    Results saved to [1mruns\train\exp2[0m
    


```python
# 테스트 이미지 경로
test_data_dir = film['val']

# train_exp_num 지정
train_exp_num = '2'
```

# 테스트
detect.py 파일 실행시켜서 실제로 yolov5 엔진이 동작했는지, 이미지에 annotation이 정상적으로 남았는지 확인한다.


```python
!python detect.py --weights runs/train/exp{train_exp_num}/weights/best.pt --img 416 --conf 0.1 --source {test_data_dir}
```

    [34m[1mdetect: [0mweights=['runs/train/exp2/weights/best.pt'], source=C:/Users/sunny//BLACKICE-segmentation-4/test/images, data=data\coco128.yaml, imgsz=[416, 416], conf_thres=0.1, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_format=0, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs\detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
    YOLOv5  v7.0-397-gde62f93c Python-3.12.7 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6144MiB)
    
    Fusing layers... 
    Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
    image 1/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\3edv_png.rf.74915f2fd5b96807778919bb2f9f6034.jpg: 416x416 1 iced-road, 4.0ms
    image 2/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\745_png.rf.232792c988a71d978ce6d14c05e541d9.jpg: 416x416 5 iced-roads, 6.0ms
    image 3/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_000322_jpg.rf.b56ea9da152c2e8b70bdf3df19d4642e.jpg: 416x416 2 iced-roads, 12.0ms
    image 4/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_000579_jpg.rf.9c08db9a1e336cf41b49eab85688fcc7.jpg: 416x416 1 clear-road, 1 iced-road, 9.0ms
    image 5/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001155_jpg.rf.eb190cab2f73bce900672fe1475d13b0.jpg: 416x416 1 clear-road, 2 iced-roads, 6.1ms
    image 6/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001285_jpg.rf.aa1a3fe091973c5abe7dc57da19983e3.jpg: 416x416 3 iced-roads, 6.0ms
    image 7/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001345_jpg.rf.c1e538da8dfa02c3a6fcf540481b6b45.jpg: 416x416 2 iced-roads, 5.0ms
    image 8/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001370_jpg.rf.72f592f58724c543d7d2aaee58915543.jpg: 416x416 1 iced-road, 4.0ms
    image 9/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001519_jpg.rf.2e1ee5849b2f5ca7167df60323b95e17.jpg: 416x416 1 clear-road, 2 iced-roads, 5.0ms
    image 10/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001564_jpg.rf.a19aa1aef6072da3e2ffeec88127e67f.jpg: 416x416 3 iced-roads, 4.1ms
    image 11/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001623_jpg.rf.881ee4ccff4ce0199381fa6bb0934942.jpg: 416x416 2 iced-roads, 4.5ms
    image 12/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001634_jpg.rf.5d0f7188b14e1b07141a3044a4e0dd5c.jpg: 416x416 2 iced-roads, 6.5ms
    image 13/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001798_jpg.rf.27ab99f3f2f3515be40272c98d75a185.jpg: 416x416 1 iced-road, 7.0ms
    image 14/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\Norway_001946_jpg.rf.85ed70c74aeedc4b90d3050c5aec766d.jpg: 416x416 1 iced-road, 4.5ms
    image 15/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\b5thg_png.rf.7c00e373d12997cbd8857890ae7fe505.jpg: 416x416 3 iced-roads, 8.5ms
    image 16/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\f1_png.rf.0b5d5756ea1658766bebcca4d0168b74.jpg: 416x416 1 iced-road, 6.0ms
    image 17/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-11-_png.rf.9bcd5e884f9a2ce50df5145a453943f5.jpg: 416x416 2 iced-roads, 5.0ms
    image 18/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-2023-04-20T185933-957_png.rf.6204ee789b6acebfc9b4ff071e422bab.jpg: 416x416 1 clear-road, 2 iced-roads, 7.0ms
    image 19/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-2023-04-20T185933-960_png.rf.809a9b9a69cb7c4ae8ce473509b41023.jpg: 416x416 1 iced-road, 9.0ms
    image 20/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-2023-04-20T190808-903_png.rf.8dce50bace16cee28d5337c0292378c7.jpg: 416x416 1 iced-road, 3.3ms
    image 21/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-2023-04-20T190808-919_png.rf.ecfdd52f1bc57996c61aa166f5599ae1.jpg: 416x416 5 iced-roads, 4.1ms
    image 22/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-2023-04-20T190808-920_png.rf.339fda0d4fa2de4f6c77e16d805b9a90.jpg: 416x416 3 iced-roads, 6.0ms
    image 23/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-25-_png.rf.6d11ee5e0b92492cc912374068b31148.jpg: 416x416 3 iced-roads, 5.2ms
    image 24/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-35-_png.rf.2a6d7f522025d8792f2305a437b9791c.jpg: 416x416 1 iced-road, 4.1ms
    image 25/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-39-_png.rf.6778781d5ba5987ee301a2116b12d34f.jpg: 416x416 1 iced-road, 8.5ms
    image 26/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-41-_png.rf.ba693e299601472472722abbdd7c88ea.jpg: 416x416 3 iced-roads, 6.0ms
    image 27/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-49-_png.rf.beecb04d3a8ac6a8c7e5eb408ffcde18.jpg: 416x416 3 iced-roads, 5.5ms
    image 28/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-50-_png.rf.e69f0f4eede5e36e85b20a2a54849e72.jpg: 416x416 2 iced-roads, 6.0ms
    image 29/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-51-_png.rf.78af658698e3ad032b8d047319e6aa0a.jpg: 416x416 1 iced-road, 5.0ms
    image 30/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-55-_png.rf.561ba752249d6d6d91f242ff44f6b747.jpg: 416x416 4 iced-roads, 9.5ms
    image 31/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-58-_png.rf.4eb70e642dc3f6da1b5696cd6d8e065a.jpg: 416x416 3 iced-roads, 10.5ms
    image 32/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-71-_png.rf.cde3d0e50bfbcf36d80357c0b5853ead.jpg: 416x416 2 iced-roads, 7.5ms
    image 33/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\th-77-_png.rf.ee96fe735d9a5f65652800ec0b9ac389.jpg: 416x416 2 iced-roads, 10.5ms
    image 34/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\training-AI-CROPPED_mp4-798_jpg.rf.de16da0ed3d8e9908ba442eab22646a1.jpg: 416x416 2 clear-roads, 10.0ms
    image 35/35 C:\Users\sunny\\BLACKICE-segmentation-4\test\images\xcvoij_png.rf.945ae300801754bbf9deb86219d73a73.jpg: 416x416 1 iced-road, 13.0ms
    Speed: 0.2ms pre-process, 6.7ms inference, 3.7ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns\detect\exp3[0m
    

# 테스트 결과 확인
검증 데이터를 확인할 수 있다.

test_exp_num에 exp 순서를 적어넣으면 그 exp번째 폴더에 있는 사진을 아래 결과창에서 보여준다.


```python
import glob
from IPython.display import Image, display

test_exp_num = '3'

if not os.path.exists('C:/Users/sunny/졸업작품/yolov5/runs/detect/exp' + str(test_exp_num) + '/') :
  raise Exception('test_exp_num 을 다시 확인하세요.')

for imageName in glob.glob('C:/Users/sunny/졸업작품/yolov5/runs/detect/exp' + str(test_exp_num) + '/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")
```


    
![jpeg](output_23_0.jpg)
    


    
    
    


    
![jpeg](output_23_2.jpg)
    


    
    
    


    
![jpeg](output_23_4.jpg)
    


    
    
    


    
![jpeg](output_23_6.jpg)
    


    
    
    


    
![jpeg](output_23_8.jpg)
    


    
    
    


    
![jpeg](output_23_10.jpg)
    


    
    
    


    
![jpeg](output_23_12.jpg)
    


    
    
    


    
![jpeg](output_23_14.jpg)
    


    
    
    


    
![jpeg](output_23_16.jpg)
    


    
    
    


    
![jpeg](output_23_18.jpg)
    


    
    
    


    
![jpeg](output_23_20.jpg)
    


    
    
    


    
![jpeg](output_23_22.jpg)
    


    
    
    


    
![jpeg](output_23_24.jpg)
    


    
    
    


    
![jpeg](output_23_26.jpg)
    


    
    
    


    
![jpeg](output_23_28.jpg)
    


    
    
    


    
![jpeg](output_23_30.jpg)
    


    
    
    


    
![jpeg](output_23_32.jpg)
    


    
    
    


    
![jpeg](output_23_34.jpg)
    


    
    
    


    
![jpeg](output_23_36.jpg)
    


    
    
    


    
![jpeg](output_23_38.jpg)
    


    
    
    


    
![jpeg](output_23_40.jpg)
    


    
    
    


    
![jpeg](output_23_42.jpg)
    


    
    
    


    
![jpeg](output_23_44.jpg)
    


    
    
    


    
![jpeg](output_23_46.jpg)
    


    
    
    


    
![jpeg](output_23_48.jpg)
    


    
    
    


    
![jpeg](output_23_50.jpg)
    


    
    
    


    
![jpeg](output_23_52.jpg)
    


    
    
    


    
![jpeg](output_23_54.jpg)
    


    
    
    


    
![jpeg](output_23_56.jpg)
    


    
    
    


    
![jpeg](output_23_58.jpg)
    


    
    
    


    
![jpeg](output_23_60.jpg)
    


    
    
    


    
![jpeg](output_23_62.jpg)
    


    
    
    


    
![jpeg](output_23_64.jpg)
    


    
    
    


    
![jpeg](output_23_66.jpg)
    


    
    
    


    
![jpeg](output_23_68.jpg)
    


    
    
    


```python

```
