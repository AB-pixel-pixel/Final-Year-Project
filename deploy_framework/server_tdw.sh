source ~/anaconda3/etc/profile.d/conda.sh
conda activate ai2thor_bin
uvicorn framework_server_tdw:app --port 8000