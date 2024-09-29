###################
###################

import json
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models.provider_factory import ProviderFactory

###################
###################

# Parse the arguments
user_args = ProviderFactory.parse_provider_arg()
provider_class = ProviderFactory.get_provider(user_args.inference_provider)
provider_class.initialize_provider()

llm = provider_class.get_llm_instance()

###################
###################

prompt_template_force_structure = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
			Your task is to process survey data and return a JSON array with objects/dicts that hold the following keys:
			"text": The text of the segment,
			"category": The category of the section. Possible values include but are not limited to: "NVBugs", "JIRA", "Nvidia Help", "Support", "Devices", "OS",
			"clarity": If the segments sentiment is Negative and the text is clear enough for a human to understand the pain point, mark this field as "Clear". Otherwise, mark it as "Unclear".
			"sentiment": The user sentiment of the segment text (Positive / Negative / Neutral),
			"summary": A summary of the segment text in less than 10 words, using words from the text and skipping connecting words if necessary.
			Instructions:
			1. Break the user survey into multiple segments based on the main topic.
			2. Popular topics include: NVBugs, JIRA, Nvidia Help, Support, Devices, OS.
			3. The topic can be from the popular topics list or another topic you define. If unsure, use the topic "Other".
			4. Note that Nvidia Help is the unified ticketing system for IT and HR.
			5. Note that Devices include laptops and mobile devices.
			Ensure your response is a JSON array which holds objects/dicts each comprised of the mentioned keys. Here is sn example -
			[
				(
					"text": "This is the first segments text...",
					"category": "NVHelp",
					"clarity": "UnClear",
					"sentiment": "Positive",
					"summary": "Do this and that..."
				),
			]
			""",
            # Once again, the response must be ONLY the JSON array with objects/dicts and nothing else.
            # """,
        ),
        (
            "user",
            """
			Survey of multiple users:
			{survey}
			""",
        ),
    ]
)

###################
###################


# TypedDict
class StructuredSurvey(BaseModel):
    text: str = Field(default="", description="The text of the segment")
    category: Literal["NVBugs", "JIRA", "Nvidia Help", "Support", "Devices", "OS"] = (
        Field(description="The category of the section")
    )
    clarity: Literal["Clear", "Unclear"] = Field(
        description="If the segments sentiment is Negative and the text is clear enough "
        'for a human to understand the pain point, mark this field as "Clear". '
        'Otherwise, mark it as "Unclear"',
    )
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="The user sentiment of the segment text"
    )
    summary: str = Field(
        description="A summary of the segment text in less than 10 words, using words from "
        "the text and skipping connecting words if necessary",
    )


class StructuredSurveys(BaseModel):
    surveys: list[StructuredSurvey] = Field(
        default=[], description="A list of all the structured serveys"
    )


survey = (
    "User 1: To save time when changing devices (LT or DT), it would be really great to have a step-by-step email "
    "or tutorial reminding you how to transfer all your datas, including reinstalling softwares, without "
    "wasting too much time. "
    "User 2: Im very angry, i didnt get any help from the support team. "
)

chain_prompt_structured = prompt_template_force_structure | llm | StrOutputParser()
answer_prompt_structured = chain_prompt_structured.invoke({"survey": survey})

print("----------------------------------------")
print(answer_prompt_structured)
print("----------------------------------------")

###################
###################

if provider_class.supports_structured_output():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Your task is to process survey data and return the survey after putting it in a structured format (JSON)
                Instructions:
                1. Break the user survey into multiple segments based on the main topic.
                2. Popular topics include: NVBugs, JIRA, Nvidia Help, Support, Devices, OS.
                3. The topic can be from the popular topics list or another topic you define. If unsure, use the topic "Other".
                4. Note that Nvidia Help is the unified ticketing system for IT and HR.
                5. Note that Devices include laptops and mobile devices.
                """,
            ),
            (
                "user",
                """
                Survey:
                {survey}
                """,
            ),
        ]
    )

    chain_forced_structure = prompt_template | llm.with_structured_output(
        StructuredSurveys
    )
    answer_forced_structure = chain_forced_structure.invoke({"survey": survey})
    assert isinstance(answer_forced_structure, BaseModel)

    print(json.dumps(answer_forced_structure.model_dump(), indent=4))
    print("----------------------------------------")
else:
    print("Provider does not support structured output")
