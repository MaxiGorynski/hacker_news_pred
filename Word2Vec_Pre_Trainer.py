import os
import requests
import tempfile
from gensim.corpora.wikicorpus import WikiCorpus

# URL of the latest Wikipedia dump
wiki_dump_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

# Output directory for processed text
output_dir = "wiki_texts"
os.makedirs(output_dir, exist_ok=True)

# Create a temporary file to store the Wikipedia dump
with tempfile.NamedTemporaryFile(delete=False, suffix=".bz2") as tmp_file:
    print(f"Downloading Wikipedia dump to temp file: {tmp_file.name}")

    with requests.get(wiki_dump_url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            tmp_file.write(chunk)

# Now use Gensim's WikiCorpus on the downloaded file
print("Processing Wikipedia dump...")
wiki = WikiCorpus(tmp_file.name, dictionary={})

with open(os.path.join(output_dir, "wiki_texts.txt"), 'w', encoding='utf-8') as output:
    for i, text in enumerate(wiki.get_texts()):
        output.write(' '.join(text) + '\n')
        if i % 10000 == 0:
            print(f"Processed {i} articles")

# Optionally delete the temporary file after processing
os.remove(tmp_file.name)
print("Done. Temporary dump file deleted.")
