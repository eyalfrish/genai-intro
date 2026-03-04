import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_classic.chains import TransformChain
from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain
from PIL import Image
from pydantic import BaseModel, Field

from models.provider_factory import ProviderFactory

############################################
# Component #1 - Load Image
############################################


class ModelImageInformationInput(TypedDict):
    image_path: str
    prompt: str
    formatting_instruction: str


def load_image(
    inputs: ModelImageInformationInput, max_image_size: int = 5_000_000
) -> dict[str, str]:
    """Load image from file and encode it as base64."""
    image_path_str = inputs["image_path"]

    def encode_image(image_path: Path) -> tuple[bytes, bytes]:
        with image_path.open("rb") as image_file:
            buffer = image_file.read()
            return buffer, base64.b64encode(buffer)

    image_buffer, image_base64 = encode_image(Path(image_path_str))

    # get image format
    img_format = Path(image_path_str).suffix[1:]

    # compress image if its size is too large
    if len(image_base64) > max_image_size:
        # compress image
        img = Image.open(io.BytesIO(image_buffer))
        compressed_format = "JPEG"
        with io.BytesIO() as output:
            img.save(output, format=compressed_format, quality=85)
            contents = output.getvalue()
        image_base64 = base64.b64encode(contents)
        img_format = compressed_format.lower()

    return {"image": image_base64.decode("utf-8"), "image_format": img_format}


load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image", "image_format"],
    transform=load_image,
)

############################################
# Component #2 - Image Structured data
############################################

vision_prompt = """
    Given the image, provide the following information:
    - A count of how many people are in the image
    - A list of the main objects present in the image
    - A description of the image
    """


class ImageInformation(BaseModel):
    """Information about an image."""

    image_description: str = Field(description="a short description of the image")
    people_count: int = Field(description="number of humans on the picture")
    main_objects: list[str] = Field(
        description="list of the main objects on the picture"
    )


############################################
# Component #3 - Image Model
############################################


# Extend the ModeImageInformationInput by adding the image field
class ModelImageInformationInputExt(ModelImageInformationInput):
    image: str
    image_format: Literal["jpg", "jpeg", "png", "bmp", "gif"]


@chain
def image_model(
    inputs: ModelImageInformationInputExt,
) -> str | list[str | dict[Any, Any]]:
    """Invoke model with image and prompt."""
    model = PROVIDER_CLASS.get_vision_instance(
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
    )
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {
                        "type": "text",
                        "text": inputs["formatting_instruction"],
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{inputs['image_format']};base64,{inputs['image']}"
                        },
                    },
                ]
            )
        ]
    )
    return msg.content


############################################
# Component #4 - Create chain
############################################


def get_image_information(image_path: Path, vision_prompt: str) -> ImageInformation:
    output_parser = JsonOutputParser(pydantic_object=ImageInformation)
    vision_chain = load_image_chain | image_model | output_parser
    return ImageInformation.model_validate(
        vision_chain.invoke(
            {
                "image_path": f"{image_path}",
                "prompt": vision_prompt,
                "formatting_instruction": output_parser.get_format_instructions(),
            }
        )
    )


############################################
# Main
############################################


def main(image_path: Path) -> None:
    print(f"Processing image: {image_path}")
    result = get_image_information(image_path, vision_prompt)

    # Convert the result to dict and save to a JSON file
    result_dict = result.model_dump()
    print(json.dumps(result_dict, indent=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generating structured data from an image",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the image file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser = ProviderFactory.add_provider_arg(parser)

    return parser.parse_args()


if __name__ == "__main__":
    global PROVIDER_CLASS

    args = parse_args()
    if args.debug:
        set_debug(True)
    PROVIDER_CLASS = ProviderFactory.get_provider(args.inference_provider)
    PROVIDER_CLASS.initialize_provider()

    main(args.input)
