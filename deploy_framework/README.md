# Code

uvicorn framework_server_tdw:app --port 8000
./scripts/sending_msg_test.sh

uvicorn framework_server_tdw:app --port 8001
./scripts/sending_msg_test1.sh

uvicorn framework_server_tdw:app --port 8002
./scripts/sending_msg_test2.sh

uvicorn framework_server_tdw:app --port 8003
./scripts/sending_msg_test3.sh

uvicorn framework_server_tdw:app --port 8004
# 运行可视化网页
enter 


# 代码结构
framework_structure.py 包含所有头文件,连同着框架中的数据结构
common_imports.py 引用了framework_structure,并添加了一些公共变量
framework_tdw_server.py is 程序入口


ln -s /media/airs/BIN/code_base/cb/communication_protocol.py communication_protocol.py 

gpt-4o-mini_20250328_161121