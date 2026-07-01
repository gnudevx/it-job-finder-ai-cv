from google import genai

KEY = ""

client = genai.Client(api_key=KEY)

print("Listing OK")

for m in client.models.list():
    pass

print("Generate...")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="hello"
)

print(response.text)