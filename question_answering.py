from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Initialize Hugging Face Transformers Question Answering model and tokenizer
# You can use another Hugging Face Transformers Question Answering model
model_qa = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_qa)
tokenizer = AutoTokenizer.from_pretrained(model_qa)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Define your context for question answering
# You can make your own dataset or you can get it from https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/
context = ('There are three major types of rock: igneous, sedimentary, and metamorphic. The rock cycle is an important concept in geology which illustrates the relationships between these three types of rock, and magma. When a rock crystallizes from melt (magma and/or lava), it is an igneous rock. This rock can be weathered and eroded, and then redeposited and lithified into a sedimentary rock, or be turned into a metamorphic rock due to heat and pressure that change the mineral content of the rock which gives it a characteristic fabric. The sedimentary rock can then be subsequently turned into a metamorphic rock due to heat and pressure and is then weathered, eroded, deposited, and lithified, ultimately becoming a sedimentary rock. Sedimentary rock may also be re-eroded and redeposited, and metamorphic rock may also undergo additional metamorphism. All three types of rocks may be re-melted; when this happens, a new magma is formed, from which an igneous rock may once again crystallize.')

def answer_question(question, context):
    QA_input = {'question': question, 'context': context}
    res = nlp(QA_input)
    return res['answer'], res['score']

try :
    while True:
        user_input = input("\nAsk Something! : ")
        answer, score = answer_question(user_input, context)

        if score >= 0.65 :
            print("Question :", user_input)
            print("Answer :", answer)
            print("Score :", score)
        else :
            print("Question :", user_input)
            print("Answer : No answer for the question")
            print("Score :", score)

# End the program
except KeyboardInterrupt:
    print("\nProgram End")