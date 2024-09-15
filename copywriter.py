from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.prompts.prompt import PromptTemplate
from langsmith import traceable


def get_comments_prompt(query, draft):
    system_message = SystemMessage(
        content="""
            You are an AI text reviewer with a keen eye for detail and a deep understanding of language, style, and grammar. 
            Your task is to refine and improve the draft content provided by the writers, offering advanced copyediting techniques and suggestions to enhance the overall quality of the text. 
            When a user submits a piece of writing, follow these steps:
            1. Read the orginal query from the user so you understand clearly the request that was given to the writer.
            2. Read through the draft text carefully, identifying areas that need improvement in terms of grammar, punctuation, spelling, syntax, and style.
            3. Provide specific, actionable suggestions for refining the text, explaining the rationale behind each suggestion.
            4. Offer alternatives for word choice, sentence structure, and phrasing to improve clarity, concision, and impact.
            5. Ensure the tone and voice of the writing are consistent and appropriate for the intended audience and purpose.
            6. Check for logical flow, coherence, and organization, suggesting improvements where necessary.
            7. Provide feedback on the overall effectiveness of the writing, highlighting strengths and areas for further development.
            
            Your suggestions should be constructive, insightful, and designed to help the user elevate the quality of their writing.
            You never generate the corrected text by itself. *Only* give the comment.
        """
    )
    human_message = HumanMessage(
        content=f"""
        Original query: {query}
        ------------------------
        Draft text: {draft}
        """
    )
    return [system_message, human_message]

@traceable(run_type="llm", name="generate_comments")
def generate_comments(chat_llm, query, draft, callbacks=[]):
    messages = get_comments_prompt(query, draft)
    response = chat_llm.invoke(messages, config={"callbacks": callbacks})
    return response.content

def get_final_text_prompt(query, draft, comments):
    system_message = SystemMessage(
        content="""
            You are an AI copyeditor with a keen eye for detail and a deep understanding of language, style, and grammar.
            Your role is to elevate the quality of the writing.
            You are given:
                1. The orginal query from the user
                2. The draft text from the writer
                3. The comments from the reviewer
            Your task is to refine and improve draft text taking into account the comments from the reviewer.
            Output a fully edited version that takes into account the original query, the draft text, and the comments from the reviewer.
            Keep the references of the draft untouched!
        """
    )
    human_message = HumanMessage(
        content=f"""
        Original query: {query}
        -------------------------------------
        Draft text: {draft}
        -------------------------------------
        Comments from the reviewer: {comments}
        -------------------------------------
        Final text:
        """
    )
    return [system_message, human_message]


@traceable(run_type="llm", name="generate_final_text")
def generate_final_text(chat_llm, query, draft, comments, callbacks=[]):
    messages = get_final_text_prompt(query, draft, comments)
    response = chat_llm.invoke(messages, config={"callbacks": callbacks})
    return response.content    


def get_compare_texts_prompts(query, draft_text, final_text):
    system_message = SystemMessage(
        content="""
        I want you to act as a writing quality evaluator. 
        I will provide you with the original user request and four texts. 
        Your task is to carefully analyze, compare the two texts across the following dimensions and grade each text 0 to 10:
            1. Grammar and spelling - Which text has fewer grammatical errors and spelling mistakes?
            2. Clarity and coherence - Which text is easier to understand and has a more logical flow of ideas? Evaluate how well each text conveys its main points.
            3. Tone and style - Which text has a more appropriate and engaging tone and writing style for its intended purpose and audience?
            4. Sticking to the request - Which text is more successful responding to the original user request. Consider the request, the style, the length, etc.
            5. Overall effectiveness - Considering the above factors, which text is more successful overall at communicating its message and achieving its goals?

        After comparing the texts on these criteria, clearly state which text you think is better and summarize the main reasons why. 
        Provide specific examples from each text to support your evaluation.
        """
    )
    human_message = HumanMessage(
        content=f"""
        Original query: {query}
        ------------------------
        Text 1: {draft_text}
        ------------------------
        Text 2: {final_text}
        ------------------------
        Summary:
        """
    )
    return [system_message, human_message]


def compare_text(chat_llm, query, draft, final, callbacks=[]):
    messages = get_compare_texts_prompts(query, draft_text=draft, final_text=final)
    response = chat_llm.invoke(messages, config={"callbacks": callbacks})
    return response.content