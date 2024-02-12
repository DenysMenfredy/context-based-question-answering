from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from question_answer import get_answer
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("atharvamundada99/bert-large-question-answering-finetuned-legal")
    model = AutoModelForQuestionAnswering.from_pretrained("atharvamundada99/bert-large-question-answering-finetuned-legal")
    
    context = input("Type your context:\n")
    question = ""
    while question != "\\q":
        question = input("Type your question or type \\q to quit\n")

        answer = get_answer(tokenizer, model, question, context)
        print(answer)
