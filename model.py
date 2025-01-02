from dotenv import load_dotenv

from langchain_cohere import ChatCohere
from langchain.schema import StrOutputParser
from prompting import chat_prompt_template


load_dotenv()

model = ChatCohere(
    model="command-r-plus",
    streaming=True
)

chain = chat_prompt_template | model | StrOutputParser()

