import torch


def get_answer(tokenizer, model, question, context):
    inputs = tokenizer(question, context, return_tensors="pt") # encode the question and context in tokens, and returns it in pytorch tensors.
    #print(inputs)
    with torch.no_grad():
        outputs = model(**inputs) # calculate the response passing the input to the pre-trained model
    #print(outputs)
    answer_start_index = outputs.start_logits.argmax() # get the inputs index that starts the answer
    answer_end_index = outputs.end_logits.argmax() # get the inputs index that ends the answer

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1] # get the inputs slice that contains the answer
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True) # decode the answer to text
    return answer


