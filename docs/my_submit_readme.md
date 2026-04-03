docker run -it \
    --gpus all \
    --entrypoint /bin/bash \
    --network host \
    -e HF_HUB_OFFLINE=1 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /home/nvidia/maoz/ckpts:/checkpoints \
    -v /home/nvidia/maoz/lehome-challenge/Datasets:/opt/lehome-challenge/Datasets \
    -v /home/nvidia/maoz/lehome-challenge/Assets:/opt/lehome-challenge/Assets \
    lehome-challenge

source .venv/bin/activate

python -m ensurepip && python -m pip install git+https://ghfast.top/https://github.com/huggingface/transformers.git@fix/lerobot_openpi

xvfb-run python -m scripts.eval \
      --policy_type lerobot \
      --policy_path /checkpoints/lehome_pi05_step62k_reject \
      --garment_type "top_long" \
      --dataset_root Datasets/example/top_long_merged \
      --num_episodes 2 \
      --enable_cameras \
      --device cpu \
      --headless


## 防止重装
docker commit $(docker ps -q --latest) lehome-challenge:with-transformers

docker run -it --gpus all --shm-size=16gb --network host \                                                              
    -e HF_HUB_OFFLINE=1 \                                                                                                 
    -e HF_ENDPOINT=https://hf-mirror.com \                                                                                
    -v ~/.cache/huggingface:/root/.cache/huggingface \                                                                    
    -v /home/nvidia/maoz/ckpts:/checkpoints \                                                                             
    -v /home/nvidia/maoz/lehome-challenge/Datasets:/opt/lehome-challenge/Datasets \                                       
    -v /home/nvidia/maoz/lehome-challenge/Assets:/opt/lehome-challenge/Assets \                                           
    --entrypoint bash lehome-challenge:with-transformers 