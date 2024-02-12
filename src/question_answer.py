import torch


def get_answer(tokenizer, model, question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_start_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    return answer


