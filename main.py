from transformers import AutoTokenizer, AutoModelForCausalLM
from github import Github
from datetime import datetime

# Replace these with your GitHub credentials and repository information
g = Github("YOUR_GITHUB_USERNAME", "YOUR_GITHUB_PASSWORD_OR_ACCESS_TOKEN")
repo_name = "YOUR_REPOSITORY_NAME_HERE"

# Set up the GPT-4All-J model and tokenizer
model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")
tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")

# Set the prompt for the GPT-4All-J API to generate the topic and title
prompt = "Generate a random computing topic and a title for a blog post."

# Generate the topic and title using GPT-4All-J
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=100, num_beams=5, early_stopping=True)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
topic, title = generated.split("\n\n", maxsplit=1)

# Set the prompt for the GPT-4All-J API to generate the article
prompt = f"Write a blog post about {topic}. Must be at least 4 paragraphs, no more than 6."

# Generate the article using GPT-4All-J
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=1000, num_beams=5, early_stopping=True)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
article = generated.strip()

# Format the date for the filename
date_str = datetime.now().strftime("%Y-%m-%d")

# Format the markdown content for the article
content = f"# {title}\n\n{topic}\n\n{article}"

# Create the filename
filename = f"{date_str}-{title.replace(' ', '-').lower()}.md"

# Get the repository to commit to
repo = g.get_user().get_repo(repo_name)

# Create a new file with the content
file = repo.create_file(filename, f"Add {filename}", content)

# Output the URL of the created file
print(f"File created at {file.html_url}")
