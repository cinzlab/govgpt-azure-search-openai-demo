from typing import Any, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """
# GovGPT prompts
#     system_chat_template = (
#         "You are an intelligent assistant helping Contoso Inc employees with their healthcare plan questions and employee handbook questions. "
#         + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
#         + "Answer the following question using only the data provided in the sources below. "
#         + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
#         + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
#     )

#     # shots/sample conversation
#     question = """
# 'What is the deductible for the employee plan for a visit to Overlake in Bellevue?'

# Sources:
# info1.txt: deductibles depend on whether you are in-network or out-of-network. In-network deductibles are $500 for employee and $1000 for family. Out-of-network deductibles are $1000 for employee and $2000 for family.
# info2.pdf: Overlake is in-network for the employee plan.
# info3.pdf: Overlake is the name of the area that includes a park and ride near Bellevue.
# info4.pdf: In-network institutions include Overlake, Swedish and others in the region
# """
#     answer = "In-network deductibles are $500 for employee and $1000 for family [info1.txt] and Overlake is in-network for the employee plan [info2.pdf][info4.pdf]."


#EduGPT prompts

    # CoT prompt
    with open('/workspaces/edugpt-azure-search-openai-demo/app/backend/approaches/CoT_prompt.txt', 'r') as f:
        cot_content = f.read()

    system_chat_template = (
        "<thinking_protocol>"
        + "You are EduGPT, an intelligent assistant helping teachers access curriculum content and create lesson plans more efficiently from a limited set of New Zealand educational sources. "
        + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
        + "Answer the following question using only the data provided in the sources below. "
        + "Each source has a name followed by a colon and the actual information; always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. Use the example below to guide your answer."
        + "Every fact in your response must include a citation from the indexed documents using square brackets, e.g. [source_name.html]. **Do not provide any fact without a citation.** If you cannot find relevant information, refuse to answer. Cite sources separately and do not combine them."
    )

    # shots/sample conversation
    question = """
    'Can you help me create a detailed lesson plan on persuasive writing for my Year 8 English class, including specific activities, timelines, and resources?'

    Sources:
    curriculum1.txt: Persuasive writing involves stating a clear position, supporting it with evidence, and convincing the reader.
    curriculum2.pdf: Year 8 English curriculum focuses on developing writing skills, including persuasive and argumentative texts.
    curriculum3.pdf: Lesson planning should include activities like debates, essay writing, and peer review sessions.
    curriculum4.pdf: Key concepts for Year 8 include understanding audience, purpose, and the use of persuasive language techniques.
    """

    answer = """
    Certainly! Here's a comprehensive lesson plan for a Year 8 English class on persuasive writing:

    ---

    **Lesson Title:** Introduction to Persuasive Writing

    **Objective:** 
    - Students will understand the components of persuasive writing.
    - Students will be able to state a clear position and support it with evidence.
    - Students will apply persuasive language techniques to convince their audience.

    **Materials Needed:** 
    - Examples of persuasive essays/articles
    - Debate topics list
    - Writing materials (paper, pens, computers)
    - Peer review checklists
    - Projector and whiteboard

    **Lesson Duration:** 90 minutes

    **Lesson Outline:**

    1. **Introduction (10 minutes)**
    - **Activity:** Begin with a brief discussion on what persuasive writing is and its importance.
    - **Reference:** [curriculum1.txt][curriculum2.pdf]

    2. **Understanding Components (15 minutes)**
    - **Lecture:** Explain the key components: clear position, supporting evidence, and persuasive techniques.
    - **Example Analysis:** Review excerpts from persuasive texts to identify these components.
    - **Reference:** [curriculum1.txt][curriculum2.pdf]

    3. **Audience and Purpose (10 minutes)**
    - **Discussion:** How knowing the audience and purpose shapes the writing.
    - **Activity:** Identify the audience and purpose in given examples.
    - **Reference:** [curriculum4.pdf]

    4. **Persuasive Language Techniques (15 minutes)**
    - **Lecture:** Introduce techniques such as rhetorical questions, emotive language, repetition, and strong evidence.
    - **Activity:** Highlight these techniques in sample texts.
    - **Reference:** [curriculum4.pdf]

    5. **Structured Debate (20 minutes)**
    - **Activity:** Divide the class into groups and assign debate topics.
    - **Task:** Each group prepares arguments supporting their position using persuasive techniques.
    - **Reference:** [curriculum3.pdf]

    6. **Essay Writing Workshop (20 minutes)**
    - **Activity:** Students begin drafting their own persuasive essays on a chosen topic.
    - **Guidance:** Provide a basic outline to structure their essays.
    - **Reference:** [curriculum3.pdf][curriculum2.pdf]

    7. **Peer Review Introduction (5 minutes)**
    - **Explanation:** Introduce the peer review process and distribute checklists.
    - **Task:** Explain how to give constructive feedback.
    - **Reference:** [curriculum3.pdf]

    **Homework:**
    - Complete the first draft of the persuasive essay.
    - Exchange essays with a peer for review using the provided checklist.

    **Assessment:**
    - Participation in debates.
    - Quality and clarity of the essay drafts.
    - Use of persuasive techniques and supporting evidence.
    - Effectiveness of peer feedback.

    **Extensions:**
    - Incorporate multimedia elements into persuasive presentations.
    - Explore persuasive writing in different formats (e.g., letters to the editor, advertisements).

    ---

    This lesson plan provides a structured approach to teaching persuasive writing, incorporating interactive activities and ensuring alignment with the Year 8 English curriculum standards.

    [References: curriculum1.txt, curriculum2.pdf, curriculum3.pdf, curriculum4.pdf]
    """



    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        q = messages[-1]["content"]
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")
        overrides = context.get("overrides", {})
        seed = overrides.get("seed", None)
        auth_claims = context.get("auth_claims", {})
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        results = await self.search(
            top,
            q,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"

        response_token_limit = 1024
        updated_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=overrides.get("prompt_template", self.system_chat_template),
            few_shots=[{"role": "user", "content": self.question}, {"role": "assistant", "content": self.answer}],
            new_user_content=user_content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        chat_completion = await self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=updated_messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            seed=seed,
        )

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    q,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in updated_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        return {
            "message": {
                "content": chat_completion.choices[0].message.content,
                "role": chat_completion.choices[0].message.role,
            },
            "context": extra_info,
            "session_state": session_state,
        }
