# unruly-nightmares
Using NLP techniques and LLMs to mimic style and conversation
## Synopsis
This project is an exploration of Natural Language Processing techniques, pretrained large language models, and how they can be used to mimic writing style and interact with a literary work.
Here’s what the code does:
* Have a “conversation” with the poet Rubén Darío through his book Autobiografía ([Retrieval-Augmented Generation](https://github.com/cepedayan/unruly-nightmares/blob/main/retrieval_augmented_generation/Retrieval_Augmented_Generation.ipynb)).
* Find the most similar sentence in any given book and use it to mimic conversation ([Sentence Similarity](https://github.com/cepedayan/unruly-nightmares/blob/main/sentence_similarity/Sentence_Similarity.ipynb)).
* Replicate some of Gustavo Adolfo Becquer’s most prominent stylistic attributes ([Finetuning & encoding methods for text generation](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)).
This project was developed for the KaggleX program. Special thanks to my mentor, Jonathan Schlosser, who not only guided me with their expertise in NLP, but also changed my perspective regarding my career in Data Science and had a tremendous impact on my own personal outlook and capabilities.
Although a lot more can be done —measure quality and compare model performance with linguistic KPIs, prompt engineering, experiment with number of epochs and track training loss—, this is the start of my journey with NLP.

### Example outputs
If you prefer to skip directly to the results, see here #pending.

## My experience
My starting point was a basic knowledge of Python and NLP concepts, and a very keen interest in understanding this marvelous technology. I wanted to learn how LLMs understand natural language, how they process it, and discover practical applications in creative spaces.
As a Localization Editor for Spanish in the Games industry, I seek to understand how LLMs can aid creative roles such as writers, translators, and editors in their daily tasks, in particular, those that relate to maintaining the voice and style of the brand and ensuring the continuity of characters and lore as valuable intellectual property of a company.
The questions that prompted this project were:
* How can LLMs learn stylistic attributes present in reference documents and replicate them in newly generated text?
* Can style be measured quantitatively with the help of an LLM, removing the editor’s subjective evaluation from the equation?
* What are the techniques used to make LLMs learn style vs learn facts present in any given document?

## The datasets
The following books were downloaded, cleaned, and preprocessed for using in various NLP techniques. (Public domain for their use in the US, gutemberg.org) 
* Leyendas, cuentos y poemas, by Gustavo Adolfo Bécquer
* Obras escogidas, by Gustavo Adolfo Bécquer
* Autobiografía, by Rubén Darío
  
## What I learned:
* About style ([See code here](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)):
  * Stylistic attributes can be replicated by finetuning with a dataset in the target writing style. Attributes such as replication of imagery, specific typography not common in modern languages, average sentence length, register and tone, can all be learned by an open-source model with less than 1M characters.
  * The output can be greatly improved using various decoding methods—the best performance in this case was obtained with Beam search and Sampling.
    * I used DeepESP/gpt2-spanish, available in Hugging Face, and a dataset of ~750K characters.
* About having a “conversation” with a poet through their work:
  * Models that are pretrained for the Sentence Similarity task can be used to retrieve the source text that is most similar to the sentence fed in the prompt. This by no means resemble a conversation, but it’s a starting point. ([See code here](https://github.com/cepedayan/unruly-nightmares/tree/main/sentence_similarity))
  * Retrieval-Augmented Generation is a really powerful way to answer questions based on documents that are specific to any given task or industry. In combination with a powerful LLM, such as gpt-3.5-turbo, we get a really cool text generator, conversation partner, assistant—whatever we need, just give the instruction in the prompt!— with access to our knowledge repository. ([See code here](https://github.com/cepedayan/unruly-nightmares/tree/main/retrieval_augmented_generation))
    * Although we should beware of hallucinations—this can be somewhat offset by changing the temperature—, the advantage of RAG is that it allows us to track answers back to their original source (if present in the source documents). This is an advantage vs using the typical LLM chat platforms. The cost of embedding source documents and queries is an important consideration that should be weighed against the increase in productivity and resources required.
* About myself:
  * I possess the ability to ride steep learning curves—it’s quite amazing to think that the knowledge I started with was a vague idea of what a token was and that ChatGPT was a LLM.
  * Resilience is the biggest asset in the face of complicated problems—my ability to find answers and alternative solutions was put to the test after so many errors I had to solve. (See the error log here for your amusement.) #pending
  * NLP requires a combination of creative and logical approach to problems—my preferred type of challenge.
  * I learned to value my skills over the knowledge I have at any given time—problem-solving and communication is crucial.

## Libraries and techniques
Python libraries:
* numpy
* operating system (os)
* regex (re)
* scikit-learn
NLP techniques:
* Tokenization
* Finetuning
* Embeddings
* Decoding methods for text generation
* Prompt engineering
* Retrieval-augmented generation
NLP libraries and tools:
* Transformers (Hugging Face)
* chromadb
Deep learning and LLM frameworks:
* Torch
* TensorFlow
* LangChain
* OpenAI API
Others:
* Handling character encoding standards, ISO-8859-1

## Error log
#pending

