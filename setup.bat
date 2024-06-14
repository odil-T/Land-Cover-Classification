echo Setting up anaconda environment
conda env create -f environment.yml

mkdir best_models
cd best_models

echo Downloading SegFormer semantic segmentation model
curl -o "segformer_sem_seg_checkpoint_epoch35.pt" "https://huggingface.co/odil111/segformer-fine-tuned-on-openearthmap/resolve/main/segformer_sem_seg_2024-06-05--16-54-31/segformer_sem_seg_checkpoint_epoch35.pt?download=true"

echo Downloading YOLO instance segmentation model
curl -o "best.pt" "https://huggingface.co/odil111/yolov8m-seg-fine-tuned-on-spacenetv2/resolve/main/yolov8m_inst_seg_2024-06-11--15-57-15/weights/best.pt?download=true"

conda init