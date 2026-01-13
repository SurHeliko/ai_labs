import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="Путь к Article.txt")
    parser.add_argument("--model", required=True, help="Путь к папке модели")
    parser.add_argument("--encoding", default="cp1251")
    parser.add_argument("--input_chars", type=int, default=10000)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args(argv)

    article_path: str = args.article
    model_path: str = args.model
    encoding: str = args.encoding
    input_chars: int = args.input_chars
    max_new_tokens: int = args.max_new_tokens

    if not torch.cuda.is_available():
        print("В системе нет CUDA")
        return -1
        
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    with open(article_path, "r", encoding=encoding) as f:
        data = f.read().replace("\n", " ")

    print("Chars:", len(data))
    print("Snippet:", data[90:160])
    print("Words:", len(data.split()))

    messages = [
        {
            "role": "system",
            "content": "Найди в тексте кто предложил цепное правило дифференцирования и в каком году?",
        },
        {"role": "user", "content": data[:input_chars]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )

    gen_only_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    main()