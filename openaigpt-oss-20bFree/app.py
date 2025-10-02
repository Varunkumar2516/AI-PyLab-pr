import os 
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client=OpenAI(base_url="https://openrouter.ai/api/v1",
              api_key=os.environ.get("OPENROUTER_API_KEY"))

system_prompt='You Are a Friendly and supportive Teaching Assitant for technical . Your are Also A Cute Teddy'
messages=[
                {'role':'system','content':system_prompt}
                    ]

while True:
    user_input=input("\n\nPress (Y/N) Y: More Questions N: Exit").lower()
    if user_input=='y':
        user_prompt=input("ENter your Question ?")
        messages.append({'role':'user','content':user_prompt})
        chat_completion = client.chat.completions.create(
            messages=messages,
            model='openai/gpt-oss-20b:free'
        )

        response_text=chat_completion.choices[0].message.content

        print(response_text)
    elif user_input=='n':
       break
    else:
        print("Enter Valid Input")