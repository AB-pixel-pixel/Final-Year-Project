source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai2thor_bin
cd deploy_framework
uvicorn framework_server_tdw:app --port 8013
# 8013 for ex1
# 8015 for ex2
