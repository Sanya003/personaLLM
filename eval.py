from transformers import AutoTokenizer, AutoModelForCausalLM

def predict_personality(input_text):
    tokenizer=AutoTokenizer.from_pretrained("./llm-personality-model")
    model=AutoModelForCausalLM.from_pretrained("./llm-personality-model")

    input_ids=tokenizer.encode(f"Input: {input_text}", padding=True, return_tensors="pt")
    input_ids=input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids  

    output=model.generate(
        input_ids, 
        max_length=512
    )
    # return tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_input = "Openness: 0.9, Conscientiousness: 0.4, Extraversion: 0.2, Agreeableness: 0.8, Neuroticism: 0.6"
print(predict_personality(test_input))