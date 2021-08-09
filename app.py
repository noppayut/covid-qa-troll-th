import glob
import time

import logging
import logging.handlers

import numpy as np
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
#===========================================#
# Load model and functions for inference
#===========================================#

max_len = 500

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
model_name_local = "./model_checkpoint/"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name_local).eval()

def norm_with_softmax(inputs, x):
    x = x.detach().numpy()
    
    # remove non-context part    
    p_mask = [tok != 1 for tok in inputs.sequence_ids(0)]
    
    undesired_tokens = np.abs(np.array(p_mask) - 1) & inputs['attention_mask'].detach().numpy()
    undesired_tokens_mask = undesired_tokens == 0.0
    
    x = np.where(undesired_tokens_mask, -10000.0, x)
    
    # calculate softmax
    sm = np.exp(x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True)))
    return sm

def get_answer_and_score(context, question, print_qa=False):
    model.eval()
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, 
                                   return_tensors="pt", truncation=True, 
                                   max_length=max_len)
    
    input_ids = inputs["input_ids"].tolist()[0]
    

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    qa_output = model(**inputs)
    answer_start_scores = norm_with_softmax(inputs, qa_output['start_logits'])[0]
    answer_end_scores = norm_with_softmax(inputs, qa_output['end_logits'])[0]

    answer_start = np.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = np.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score    
    score = answer_start_scores[answer_start] * answer_end_scores[answer_end]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    if print_qa:
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")
      
    return answer, score

#===========================================#
# Load documents and set up BM25 retriever
#===========================================#
doc_dir = 'line_grp_articles/'

filenames = glob.glob(doc_dir + '*.txt')

corpus = []
corpus_tkn = []

def chunk(doc, chunk_max_char=800):
    doc_len = len(doc)
    n_chunks = int(np.ceil(doc_len / chunk_max_char))
    
    chunks = []
    for i in range(n_chunks):
        begin = chunk_max_char * i
        end = begin + chunk_max_char
        chunks.append(doc[begin:end])
    return chunks

for fn in filenames:
    with open(fn, 'r') as f:
        doc = f.read().replace('\n', '  ')   # remove linebreak
        doc_chunks = chunk(doc)
        corpus.extend(doc_chunks)   

corpus_tkn = list(map(word_tokenize, corpus))

bm25 = BM25Okapi(corpus_tkn)

def get_best_matches(query, n=3):
    query_tkn = word_tokenize(query)
    docs = bm25.get_top_n(query_tkn, corpus, n=n)
    return docs

#===========================================#
# Putting parts together
#===========================================#

def ask_model(query, n=3):
    # retrive most relevant n*1.5 documents (generate many candidates, then filter blank answers out)
    docs = get_best_matches(query, n=n)
    
    # ask q question on each chunk
    ans_and_scores = sorted([get_answer_and_score(doc, query) for doc in docs], key=lambda x: x[1], reverse=True)
    # remove blank answers
    ans_and_scores = list(filter(lambda x: len(x[0]) > 0, ans_and_scores))
    # get most confident one
    best_ans = ans_and_scores[0] if ans_and_scores else []
    
    return best_ans, ans_and_scores        

#===========================================#
# Setup log
#===========================================#

log_name = 'covid-qa-th-log.log'

# Set up a specific logger with our desired output level
logger = logging.getLogger('MyLogger')
logger.setLevel(logging.INFO)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(log_name, maxBytes=1e5, backupCount=5)

logger.addHandler(handler)


#===========================================#
# Streamlit UI
#===========================================#
import streamlit as st
from PIL import Image

image = Image.open('img/mairoo.jpeg')

disclaimer = "เว็บนี้เป็นเว็บที่ทำขึ้นมาเพื่อการศึกษา คำตอบเป็นข้อมูลเท็จและไม่มีการพิสูจน์ ห้ามนำไปใช้เด็ดขาด!"

st.title('ตอบคำถามเกี่ยวกับโควิด-19 (การรักษา วัคซีน ข่าววงใน ฯลฯ) (คำตอบเป็นข้อมูลเท็จ)')
st.title('Ask me about (fake) COVID-19 info (treatment, vaccine, insider news, etc.)')
st.write(disclaimer)

n_ans = st.number_input('จำนวนคำตอบ (อาจได้น้อยกว่านี้)', min_value=1, max_value=20, value=3)
query = st.text_input('อยากรู้อะไรถามเลย (ตัวอย่าง ติดโควิดควรกินอะไร, วัคซีนอะไรดีที่สุด)')

too_short_msg = "ไม่รู้ คำถามสั้นไป"
no_ans_msg = "ไม่รู้ ถามใหม่"

if st.button('ดูคำตอบ') or query:
    logger.info(f"{time.ctime()}::: Question: {query}, N_ans: {n_ans}")
    if len(query.strip()) >= 5:
        best_ans, ans_and_scores = ask_model(query, n_ans)
        if best_ans:
            st.header("คำตอบที่ดีที่สุด:")
            st.write(best_ans[0])
            st.header("คำตอบอื่นๆ:")
            for i, (ans, _) in enumerate(ans_and_scores):
                st.write(f"{i+1}. {ans}")
        else:
            st.image(image)
            st.subheader(no_ans_msg)
    else:

        st.image(image)
        st.subheader(too_short_msg)
