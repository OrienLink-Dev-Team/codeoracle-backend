from flask import Flask, request, jsonify
import fire
from llama import Llama
from Example.Input import Input
from Example.Output import Output
import os
import torch.distributed as dist

# Initialize the Flask app
app = Flask(__name__)

generator = None

def initialize_generator():
    global generator
    generator = Llama.build(
        ckpt_dir="CodeLlama-7b",
        tokenizer_path="CodeLlama-7b/tokenizer.model",
        max_seq_len=192,
        max_batch_size=4,
    )

@app.route("/codeinterpreter", methods=["POST"])
def run_code_interpreter():
    json_data = request.json
    if not json_data:
        return jsonify({"error": "No JSON data provided"}), 400

    input_type = json_data.get('type')
    input_text = json_data.get('code')

    print(f"Input Type: {input_type}")
    print(f"Input Code: {input_text}")

    if input_type == "code":
        print("The type is 'code', performing the logic.")
        prompts = [input_text]
        prefixes = [input_text]
        suffixes = [''' ''']
        
        print(f"Input Code: {prefixes}")
        print(f"Input Code: {suffixes}")
        print(f"Input Code: {prompts}")
        

        results = generator.text_infilling(
            prefixes=prefixes,
            suffixes=suffixes,
            max_gen_len=128,
            temperature=0.0,
            top_p=0.9,
        )
        print(f"Generation: {results[0]['generation']}")
        jsonResponse = {
            "type": "code",
            "result": results
        }

        return jsonify(jsonResponse)
    else:
        return jsonify({"error": "Invalid input type"}), 400

def main(
    ckpt_dir: str = "CodeLlama-7b",
    tokenizer_path: str = "CodeLlama-7b/tokenizer.model",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 192*2,
    max_gen_len: int = 128*2,
    max_batch_size: int = 4,
):
    initialize_generator()
   
    try:
        dist.init_process_group(backend='nccl', init_method='env://')
        generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            
    app.run(host='127.0.0.1', port=8899, debug=True, threaded=True)

if __name__ == "__main__":
    fire.Fire(main)
