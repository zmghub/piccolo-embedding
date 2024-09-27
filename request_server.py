# 导入所需的库
import argparse
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import json
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


class MiniCPM(torch.nn.Module):
    support_models = ["MiniCPM"]
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
        self.model.eval()

    def weighted_mean_pooling(self, hidden, attention_mask):
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

    @torch.no_grad()
    def encode(self, input_texts, normalize_embeddings=True):
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to("cuda")
        
        outputs = self.model(**batch_dict)
        attention_mask = batch_dict["attention_mask"]
        hidden = outputs.last_hidden_state

        reps = self.weighted_mean_pooling(hidden, attention_mask)
        if normalize_embeddings:
            embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        else:
            embeddings = reps.detach().cpu().numpy()
        return embeddings

class TextInferenceService:
    def __init__(self, model_path, port):
        # 实例化模型
        minicpm_flag = False
        for key in MiniCPM.support_models:
            if key in model_path:
                minicpm_flag = True
                break
        if minicpm_flag:
            self.model = MiniCPM(model_path)
        else:
            self.model = SentenceTransformer(model_path)
        # 创建Flask应用实例
        self.app = Flask(__name__)
        # 设置服务器监听的端口
        self.port = port
        self.model_path = model_path
        
        # 定义POST请求的处理函数
        self.app.add_url_rule('/infer', "infer", self.infer, methods=['POST'])

    def infer(self,):
        # 获取请求的JSON数据
        data = json.loads(request.data)
        
        # 假设模型需要的输入格式是列表，这里需要根据实际情况调整
        list_result = True
        if "text_list" in data:
            input_data = data["text_list"]
        elif "text" in data:
            input_data = [data["text"]]
            list_result = False
        else:
            return jsonify({'error': 'invalid input data'})
        
        # 调用模型进行推理
        result = self.model.encode(input_data, normalize_embeddings=True)
        print(f"model path: {self.model_path}, port: {self.port}", flush=True)
        print(f"input num: {len(input_data)}, result shape: {result.shape}, input sample: {input_data[0]}", flush=True)
        result = result.tolist()
        if not list_result:
            result = result[0]
            
        # 返回推理结果
        return jsonify({'result': result})
    
    def start_service(self):
        # 启动服务器
        self.app.run(host='0.0.0.0', port=self.port)
        

class ArgsReader:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="简单的参数读取类")
        self._setup_arguments()

    def _setup_arguments(self):
        # 添加测试结果文件路径参数
        self.parser.add_argument(
            "model_path",
            type=str,
            nargs="?",
            default='lier007/xiaobu-embedding-v2',
            help="模型文件路径"
        )
        # 添加输出指标保存路径参数
        self.parser.add_argument(
            "--port",
            type=int,
            default=5000,
            help="服务端口指定"
        )

    def parse_args(self):
        # 解析命令行参数
        args = self.parser.parse_args()
        return args


if __name__ == '__main__':
    args_reader = ArgsReader()
    args = args_reader.parse_args()
    print(f"加载的模型的路径: {args.model_path}")
    print(f"服务端口: {args.port}")
    # 使用类创建服务实例，指定模型路径和服务端口
    service = TextInferenceService(args.model_path, args.port)
    # 启动服务
    service.start_service()
