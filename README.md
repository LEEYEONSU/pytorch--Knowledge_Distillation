### Knowledge distillation (pytorch)

- For training

  ```
  python main.py 
  # argparser Default 
  --print_freq 32 --save_dir ./save_model/ --save_every 10
  --lr 0.1 --weight_decay 1e-4 --momentum 0.9 
  --Epoch 80 --batch_size 128 --test_batch_size 100 
  --KD True 
  ```

- Result 

  - KD Teacher model = Student model 
    - SENet + resnet32 
    - KD_loss ([./model/SE.py](https://github.com/LEEYEONSU/pytorch--Knowledge_Distillation/blob/main/model/SE.py)) 

  |               | SE + resnet | KD   |
  | ------------- | ----------- | ---- |
  | top - 1 error | 94.6        | 95.1 |

  

- Dataset = CIFAR-10

  

##### Preprocessing

---

**<Data augmentation>**

- 4pixels padded
- Randomly 32 x 32 crop
- Horizontal Flip
- Normalization with mean and standard deviation



**<Parameter>**

- Weight_initialization - kaiming_normal
- Optimizer
  - SGD
    - Learning_rate : 0.1
    - Milestones [250, 375]
    - gamma : 0.1
  - Weight_decay : 0.4
  - momentum : 0.9

