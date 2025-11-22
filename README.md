# LLada-sft
LLada-sft
### data——process
请将文件转化为jsonl的格式，同时注意，因为使用的是多卡流式加载，所以文件数要大于等于卡的数量
请自行拆分

### custom_data
对于数据的特殊处理，请重写utils/process_data

### config
配置文件再config中

### 使用框架
使用的框架为lightning
