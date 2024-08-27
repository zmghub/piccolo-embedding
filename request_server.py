# 导入所需的库
import argparse
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import json

class TextInferenceService:
    def __init__(self, model_path, port):
        # 实例化模型
        self.model = SentenceTransformer(model_path)
        # 创建Flask应用实例
        self.app = Flask(__name__)
        # 设置服务器监听的端口
        self.port = port
        
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
