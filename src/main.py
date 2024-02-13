from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from question_answer import get_answer
import streamlit as st

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("atharvamundada99/bert-large-question-answering-finetuned-legal")
    model = AutoModelForQuestionAnswering.from_pretrained("atharvamundada99/bert-large-question-answering-finetuned-legal")

    return tokenizer, model

if __name__ == '__main__':
    tokenizer, model = get_model()
    st.title("Write a context and then ask questions about it")
    st.text_area("Type your context", key="context", height=400)
    question = st.chat_input("Type your question")
    if question:
        answer = get_answer(tokenizer, model, question, st.session_state.context)
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(f"Answer: {answer}")
        #st.write(st.session_state.question)


