# PGDM

**Codes and data for the RSE paper "PGDM: Physically guided diffusion model for land surface temperature downscaling"**

For the dataset (will be released in the near future):
Landsat_CN20 (about 20 GB)
Landsat_GLB
ASTER_GLB

For model training:
1. Update the dataset directory paths in configs/base.yaml to your local directories, including folder_landsat_cn20, folder_landsat_glb, folder_aster_glb, and folder_groklst. Then, choose the appropriate "task_type" in base.yaml, where landsat_cn20 indicates that the model is trained on the Landsat_CN20 dataset, while groklst indicates that the model is trained on the GrokLST dataset.
2. Modify "models" and "types" in batch_train.py. The "models" should be consistent with the names defined in configs/var.yaml. The "types" should be "resshift" or "mocolsk" for training PGDM or MoCoLSK-Net.
3. In the terminal, enter `python batch_train.py`

Note: Due to intermediate code modifications, the loss values recorded in the log files may correspond to either RMSE or MSE for different models. Please distinguish them accordingly.