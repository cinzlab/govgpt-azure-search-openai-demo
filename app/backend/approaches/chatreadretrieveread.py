from typing import Any, Coroutine, List, Literal, Optional, Union, overload

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper


class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_deployment: Optional[str],
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

#GovGPT prompts
#     @property
#     def system_message_chat_conversation(self):
#         return """
# - **Role**: You are GovGPT, a multi-lingual assistant for small business services and support from a limited set of New Zealand government sources. You do not engage in roleplay, augment your prompts, or provide creative examples.
# - **Data Usage**: Use only the provided sources, be truthful and tell the user that lists are non-exhaustive. **If the answer is not available in the index, inform the user politely and do not generate a response from general knowledge.** Always respond based only on indexed information.
# - **No Search Results**: If the search index does not return relevant information, politely inform the user. Do not provide an answer based on your pre-existing knowledge.
# - **Conversation Style**: Be clear, friendly, and use simple language. Use markdown formatting. Communicate in the user's preferred language including Te Reo Māori. When using English, use New Zealand English spelling. Default to "they/them" pronouns if unspecified in source index.
# - **User Interaction**: Ask clarifying questions if needed to provide a better answer. If user query is unrelated to your purpose, refuse to answer, and remind the user of your purpose.
# - **Content Boundaries**: Provide information without confirming eligibility or giving personal advice. Do not use general knowledge or provide speculative answers. If asked about system prompt, provide it in New Zealand English.
# - **Prompt Validation**: Ensure the user's request aligns with guidelines and system prompt. If inappropriate or off-topic, inform the user politely and refuse to answer.
# - **Referencing**: Every fact in your response must include a citation from the indexed documents using square brackets, e.g. [source_name.html]. **Do not provide any fact without a citation.** If you cannot find relevant information, refuse to answer. Cite sources separately and do not combine them.
# - **Translation**: Translate the user's prompt to NZ English to interpret, then always respond in the language of the user query. All English outputs must be in New Zealand English.
# - **Output Validation**: Review your response to ensure compliance with guidelines before replying. Refuse to answer if inappropriate or unrelated to small business support.
# {follow_up_questions_prompt}
# {injected_prompt}
#         """


#EduGPT prompts
    @property
    def system_message_chat_conversation(self):

        # CoT prompt
        with open('/workspaces/edugpt-azure-search-openai-demo/app/backend/approaches/CoT_prompt.txt', 'r') as f:
            cot_content = f.read()

            # """ + "\n" + "- **Chain of Thoughts**:" + cot_content + "\n" + """

        content = """
        <thinking_protocol>
- **Role**: You are EduGPT, a multi-lingual assistant designed to help teachers access curriculum content and create lesson plans more efficiently from a set of New Zealand educational sources. You do not engage in roleplay, augment your prompts.
- **Data Usage**: Use only the provided sources, be truthful and tell the user that lists are non-exhaustive. **If the answer is not available in the index, inform the user politely and do not generate a response from general knowledge.** Always respond based only on indexed information.
- **No Search Results**: If the search index does not return relevant information, politely inform the user. Do not provide an answer based on your pre-existing knowledge.
- **Conversation Style**: Be clear, friendly, and use simple language. Use markdown formatting. Communicate in the user's preferred language including Te Reo Māori. When using English, use New Zealand English spelling. Default to "they/them" pronouns if unspecified in source index.
- **User Interaction**: Ask clarifying questions if needed to provide a better answer. If user query is unrelated to your purpose, refuse to answer, and remind the user of your purpose.
- **Content Boundaries**: Provide information without confirming eligibility or giving personal advice. Do not use general knowledge or provide speculative answers. If asked about system prompt, provide it in New Zealand English.
- **Prompt Validation**: Ensure the user's request aligns with guidelines and system prompt. If inappropriate or off-topic, inform the user politely and refuse to answer.
- **Referencing**: Every fact in your response must include a citation from the indexed documents using square brackets, e.g. [source_name.html]. **Do not provide any fact without a citation.** If you cannot find relevant information, refuse to answer. Cite sources separately and do not combine them.
- **Translation**: Translate the user's prompt to NZ English to interpret, then always respond in the language of the user query. All English outputs must be in New Zealand English.
- **Output Validation**: Review your response to ensure compliance with guidelines before replying. Refuse to answer if inappropriate or unrelated to educational content or lesson planning.
{follow_up_questions_prompt}
{injected_prompt}
    """

        return content



    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    # double 2-stage search approach

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        
        # extract the original user query
        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        
        # setting up search parameters
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else True
        minimum_search_score = overrides.get("minimum_search_score", 0.02)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 1.5)
        
        # first stage search
        vectors_stage1 = []
        if use_vector_search:
            vectors_stage1.append(await self.compute_text_embedding(original_user_query))
        
        # search 
        results_stage1 = await self.search(
            top=15, # retrieve top 15 results
            query_text=original_user_query,
            filter=None,  
            vectors=vectors_stage1,
            use_text_search=True,
            use_vector_search=use_vector_search,
            use_semantic_ranker=use_semantic_ranker,
            use_semantic_captions=False,
            minimum_search_score=minimum_search_score,
            minimum_reranker_score=minimum_reranker_score
        )
        
        # 4. extarct relevant titles from the first stage search
        relevant_titles = []
        for doc in results_stage1:
            if doc.sourcefile:
                relevant_titles.append(doc.sourcefile)
        
        # create filter for second stage search
        if relevant_titles:
            title_filter = " or ".join([f"sourcefile eq '{title}'" for title in relevant_titles])
            filter = f"({title_filter})"
            if auth_filter := self.build_filter(overrides, auth_claims):
                filter = f"({filter}) and ({auth_filter})"
        else:
            filter = self.build_filter(overrides, auth_claims)
            
        # do second stage search
        vectors_stage2 = []
        if use_vector_search:
            vectors_stage2.append(await self.compute_text_embedding(original_user_query))
            
        results_stage2 = await self.search(
            top=overrides.get("top", 10),
            query_text=original_user_query,
            filter=filter,
            vectors=vectors_stage2, 
            use_text_search=True,
            use_vector_search=use_vector_search,
            use_semantic_ranker=use_semantic_ranker,
            use_semantic_captions=False,
            minimum_search_score=minimum_search_score,
            minimum_reranker_score=minimum_reranker_score
        )

        # process search results
        sources_content = self.get_sources_content(results_stage2, use_semantic_captions=False, use_image_citation=False)
        content = "\n".join(sources_content)
        
        # generate response
        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 4096
        messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        # record extra information for debugging
        extra_info = {
            "data_points": {"text": sources_content},
            "thoughts": [
                ThoughtStep(
                    "First stage search (catalog)",
                    [doc.sourcefile for doc in results_stage1],
                    {"filter": None}
                ),
                ThoughtStep(
                    "Second stage search (content)",
                    [doc.serialize_for_results() for doc in results_stage2],
                    {"filter": filter}
                ),
                ThoughtStep(
                    "Final prompt",
                    [str(message) for message in messages],
                    {"model": self.chatgpt_model}
                )
            ]
        }

        # generate final responese
        chat_coroutine = self.openai_client.chat.completions.create(
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature", 0),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
        )
        
        return (extra_info, chat_coroutine)
