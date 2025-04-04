import os
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_cohere import ChatCohere
from langchain.schema import StrOutputParser
from prompting import chat_prompt_template
from langchain_aws import ChatBedrock
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain.chains.conversation.base import ConversationChain 
from langchain.chains.conversation.memory import ConversationBufferMemory


load_dotenv()

"""
model = ChatCohere(
    model="command-r-plus",
    streaming=True
)
"""

model = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region=os.environ["BEDROCK_AWS_REGION"],
    aws_access_key_id=os.environ["BEDROCK_AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["BEDROCK_AWS_SECRET_ACCESS_KEY"]
    # model_kwargs=dict(temperature=0),
)

# history = InMemoryChatMessageHistory()

chain = chat_prompt_template | model | StrOutputParser()

histories: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str = ''):
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()
    return histories[session_id]

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history, #lambda session_id: history,
    input_messages_key="question",
    history_messages_key="chat_history"
)



"""
class ExtendedConversationBufferMemory(ConversationBufferMemory):
    extra_variables:List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        ""Will always return list of memory variables.""
        return [self.memory_key] + self.extra_variables
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ""Return buffer with history and extra variables""
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables}) 
        return d
    

chain = ConversationChain(
    llm=model,
    prompt=chat_prompt_template,
    output_parser=StrOutputParser(),
    memory=ExtendedConversationBufferMemory(extra_variables=["context", "question"])
)
"""

