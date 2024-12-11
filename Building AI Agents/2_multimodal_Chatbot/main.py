from openai import OpenAI
import base64
import chainlit as cl
import os

os.environ['OPENAI_APT_KEY']='OPENAI_APT_KEY'

client = OpenAI(api_key='OPENAI_APT_KEY')


def append_messages(image_url=None, query=None, audio_transcript=None):
    message_list = []

    if image_url:
        message_list.append({"type": "image_url", "image_url": {"url": image_url}})

    if query and not audio_transcript:
        message_list.append({"type": "text", "text": query})

    if audio_transcript:
        message_list.append({"type": "text", "text": query + "\n" + audio_transcript})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message_list}],
        max_tokens=1024,
    )

    return response.choices[0]


def image2base64(image_path):
    with open(image_path, "rb") as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode("utf-8")


def audio_process(audio_path):
    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return transcription.text


@cl.on_message
async def chat(msg: cl.Message):

    images = [file for file in msg.elements if "image" in file.mime]
    audios = [file for file in msg.elements if "audio" in file.mime]

    if len(images) > 0:
        base64_image = image2base64(images[0].path)
        image_url = f"data:image/png;base64,{base64_image}"

    elif len(audios) > 0:
        text = audio_process(audios[0].path)

    response_msg = cl.Message(content="")

    if len(images) == 0 and len(audios) == 0:
        response = append_messages(query=msg.content)

    elif len(audios) == 0:
        response = append_messages(image_url=image_url, query=msg.content)

    else:
        response = append_messages(query=msg.content, audio_transcript=text)

    response_msg.content = response.message.content

    await response_msg.send()