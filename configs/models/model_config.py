import pandas as pd

df = pd.read_csv("model_config.csv")

for i in range(df.shape[0]):
    model = {}
    model_name = df.iloc[i]["model_name"]
    path = df.iloc[i]["model_path"]
    if path.endswith('/'):
        # 如果是的话，去掉最后一个字符
        path = path[:-1]
    step = path.split("/")[-1]
    abbr = df.iloc[i]["model_name"]+"_"+step
    abbr = abbr.replace(".","_")
    model["abbr"]=abbr
    model["type"]='opencompass.models.internal.InternLM'
    model["path"]=df.iloc[i]["model_path"]
    model["tokenizer_path"]=df.iloc[i]["tokenizer_path"]
    tokenizer_path = df.iloc[i]["tokenizer_path"]
    if "v7" in tokenizer_path or "V7" in tokenizer_path:
        model["tokenizer_type"]="v7"
    elif "v4" in tokenizer_path or "V4" in tokenizer_path:
        model["tokenizer_type"]="v4"
    elif "llama" in tokenizer_path:
        model["tokenizer_type"]="llama"
    else:
        model["tokenizer_type"]="v7"
    model["module_path"]=df.iloc[i]["module_path"]
    model["model_config"]=df.iloc[i]["model_config"]
    model["model_type"]=df.iloc[i]["model_type"]
    model["max_out_len"]=100
    model["max_seq_len"]=2048
    model["batch_size"]=16
    run_cfg = dict(num_gpus=1, num_procs=1)
    model["run_cfg"] = run_cfg

    formatted_str = "dict(\n\
        abbr='{}',\n\
        type='{}',\n\
        path='{}',\n\
        tokenizer_path='{}',\n\
        tokenizer_type='{}',\n\
        module_path='{}',\n\
        model_config='{}',\n\
        model_type='{}',\n\
        max_out_len={},\n\
        max_seq_len={},\n\
        batch_size={},\n\
        run_cfg=dict(num_gpus={}, num_procs={})\n),".format(
        model['abbr'], model['type'], model['path'],
        model['tokenizer_path'], model['tokenizer_type'],
        model['module_path'], model['model_config'],
        model['model_type'], model['max_out_len'],
        model['max_seq_len'], model['batch_size'],
        model['run_cfg']['num_gpus'], model['run_cfg']['num_procs']
    )
    print(formatted_str)
