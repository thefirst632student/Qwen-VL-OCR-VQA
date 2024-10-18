# Qwen2-VL: Vision-Language Model for OCR and VQA
This project demonstrates how to use the Qwen2-VL model from Hugging Face for Optical Character Recognition (OCR) and Visual Question Answering (VQA). The model combines vision and language capabilities, enabling users to analyze images and generate context-based responses. In this project, we'll specifically explore its application in analyzing a football image and providing tactical insights.

## Overview
Qwen2-VL is a state-of-the-art vision-language model that can handle various multimodal tasks, such as:

- **Visual Question Answering (VQA)**: Answering questions based on an image.
- **Optical Character Recognition (OCR)**: Extracting text from images.
  
This example showcases how to load the model in a Google Colab notebook, analyze a football image, and generate a tactical analysis to assist a coach.

## Setup Instructions
### Prerequisites
To run the notebook, you need to have the following installed:

- Python 3.7+
- `transformers` library
- `Pillow` for image handling
- Google Colab (recommended for easy execution in GPU environment)

### Installation
First, install the required dependencies:

```bash
!pip install git+https://github.com/huggingface/transformers
!pip install Pillow
```

### Model Setup
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
```

### Example: Football Image Analysis
1. Upload your image of a football game:

```python
from PIL import Image
image = Image.open("football.jpg")  # Replace with your image file
```

2. Prepare your question and input:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {
                "type": "text",
                "text": "What is wrong with the tactics in the image? Your analysis should help the coach."
            }
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=[text_prompt],
    images=[image],
    padding=True,
    return_tensors="pt"
)
```

3. Generate the response:

```python
inputs = inputs.to("cuda")  # Move to GPU if available

output_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]

output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)
```

### Output
The model will generate text-based analysis based on the football image, helping to inform tactical adjustments that the coach can make.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
